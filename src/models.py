import torch as th
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from dgl.nn import MaxPooling, AvgPooling, GlobalAttentionPooling
from dgl import function as fn
from dgl.nn.pytorch.softmax import edge_softmax
from dgllife.model import GATPredictor, GIN, load_pretrained, GAT, WLN

"""
- in general, for DGLLife, GNN is just a graph layer while GNNPredictor includes a graph layer, a readout layer, and an mlp layer
- 4 types of pretrained models: moleculenet, generative_models, property_prediction, reaction 
- property_prediction are GIN supervised (contextpred,infomax,edgepred,masking) and AttentiveFP, they do not contain reaodut and output MLP
- moleculenet models are a variety of architectures fine-tuned on moleculenet benchmark tasks (Tox21, PCBA, etc), the do contain readout and output MLP
- reaction and generative_models are not relevant
"""


def get_model(args, data_feats):
	
	model_type = args["model_type"]

	if model_type == "gat_p":
		model = GATP(data_feats["n_feats"],**args)

	elif model_type == "gin":
		model = GIN1(data_feats["n_feats"],data_feats["e_feats"],**args)

	elif model_type == "gat":
		model = GAT1(data_feats["n_feats"], data_feats["grover_fp_feats"], **args)

	elif model_type == "mlp_fingerprint":
		model = MLP(data_feats["finger_feats"], data_feats["grover_fp_feats"], **args)

	elif model_type == "gin_pt1":
		pretrain_name = "gin_supervised_contextpred_Tox21"
		model = GINPT(
			data_feats["n_feats"],
			data_feats["e_feats"],
			pretrain_name,
			data_feats["grover_fp_feats"],
			**args)
	elif model_type == "gin_pt2":
		pretrain_name = "gin_supervised_infomax_Tox21"
		model = GINPT(
			data_feats["n_feats"],
			data_feats["e_feats"],
			pretrain_name,
			data_feats["grover_fp_feats"],
			**args)
	elif model_type == "gin_pt3":
		pretrain_name = "gin_supervised_infomax_ToxCast"
		model = GINPT(
			data_feats["n_feats"],
			data_feats["e_feats"],
			pretrain_name,
			data_feats["grover_fp_feats"],
			**args)
	elif model_type == "gin_pt4":
		pretrain_name = "gin_supervised_infomax"
		model = GINPT(
			data_feats["n_feats"],
			data_feats["e_feats"],
			pretrain_name,
			data_feats["grover_fp_feats"],
			**args)
	elif model_type == "cnn":
		model = CNN(data_feats["c_feats"], data_feats["l_feats"],**args)
	elif model_type == "grover_pt":
		model = GROVERPT(data_feats["grover_fp_feats"],**args)
	elif model_type == "wln":
		model = WLN1(data_feats["n_feats"],data_feats["e_feats"],**args)
	else:
		print(model_type)
		raise ValueError(f"invalid model_type {model_type}")
	return model

class Identity(nn.Module):
	""" 
	for deleting layers in pretrained models
	from https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648 
	"""

	def __init__(self):
		super(Identity,self).__init__()

	def forward(self,x):
		return x

class GNIdentity(Identity):
	"""
	identity for operation that takes graph, node attributes (for example, graph pool layer)
	"""

	def forward(self,g,nh):
		return nh


class GNEIdentity(Identity):
	"""
	identity for operation that takes graph, node, edge attributes (for example. message passing layer)
	"""

	def forward(self,g,nh,eh):
		return nh


class LinearBlock(nn.Module):
	""" basic block in an MLP, with dropout and batch norm """

	def __init__(self, in_feats, out_feats, dropout=0.1):
		super(LinearBlock, self).__init__()
		self.linear = nn.Linear(in_feats, out_feats)
		self.bn = nn.BatchNorm1d(out_feats)
		self.dropout = nn.Dropout(dropout)
			
	def forward(self, x):
		return self.bn(self.dropout(F.relu(self.linear(x))))


class GATP(nn.Module):
	""" silly wrapper around GATPredictor1 """

	def __init__(self,n_feats,**kwargs):

		super(GATP,self).__init__()
		model_d = {}
		model_d['in_feats'] = n_feats
		model_d['hidden_feats'] = [kwargs['gnn_hidden_feats'] for x in range(kwargs['gnn_num_layers'])]
		model_d['num_heads'] = [kwargs['gat_num_heads'] for x in range(kwargs['gnn_num_layers'])]
		model_d['feat_drops'] = [kwargs['gnn_dropout'] for x in range(kwargs['gnn_num_layers'])]
		model_d['attn_drops'] = [kwargs['gnn_dropout'] for x in range(kwargs['gnn_num_layers'])]
		model_d["alphas"] = [0.2 for _ in range(kwargs['gnn_num_layers'])]
		# residuals list of bool for number of layers
		model_d['residuals'] = [True for _ in range(kwargs['gnn_num_layers'])]
		"""
		The way to aggregate multi-head attention results for each GAT layer, which can be either
		'flatten' for concatenating all-head results or 'mean' for averaging all-head results.
		``agg_modes[i]`` gives the way to aggregate multi-head attention results for the i-th
		GAT layer. ``len(agg_modes)`` equals the number of GAT layers. By default, we flatten
		multi-head results for intermediate GAT layers and compute mean of multi-head results
		for the last GAT layer.
		"""
		model_d['agg_modes'] = ['flatten' for _ in range(kwargs['gnn_num_layers'] - 1)] + ['mean']
		# Apply Elu to all the layers besides the last one
		model_d['activations'] = [nn.ELU(model_d['alphas'][i]) for i in range(kwargs['gnn_num_layers'] - 1)] + [None]
		model_d['n_tasks'] = kwargs['out_feats']
		model_d['predictor_hidden_feats'] = kwargs['ff_hidden_feats']
		model_d['predictor_dropout'] = kwargs['ff_dropout']
		print(model_d)
		self.gnn_p = GATPredictor(**model_d)

	def forward(self,input_d):
		""" this is compatible with canonical and pretrafsefin featurization """

		graph = input_d["graph"]
		node_h = graph.ndata["h"]
		return self.gnn_p(graph,node_h)


class MLP(nn.Module):
	def __init__(self, n_feats, grover_fp_feats, **kwargs):

		super(MLP, self).__init__()
		self.n_feats = n_feats
		self.grover_fp_feats = grover_fp_feats # can be 0
		for k,v in kwargs.items():
			setattr(self, k, v)

		self.ff_layers = construct_mlp(
			self.n_feats + self.grover_fp_feats,
			self.out_feats,
			self.ff_hidden_feats,
			self.ff_num_layers,
			self.ff_dropout
		)

	def forward(self, input_d):
		fingerprints = input_d["fingerprints"]

		if self.grover_fp in ['base', 'large']:
			output = self.ff_layers(th.cat([fingerprints, input_d["grover_fp"]], dim=1))
		else:
			output = self.ff_layers(fingerprints)

		return output


class GAT1(nn.Module):

	def __init__(self, n_feats, grover_fp_feats, **kwargs):

		super(GAT1,self).__init__()
		self.n_feats = n_feats
		self.grover_fp_feats = grover_fp_feats # can be 0
		for k,v in kwargs.items():
			setattr(self,k,v)
		self.gnn_layers = GAT(
			in_feats=self.n_feats,
			hidden_feats=[self.gnn_hidden_feats for i in range(self.gnn_num_layers)],
			num_heads=[self.gat_num_heads for i in range(self.gnn_num_layers)],
			feat_drops=[self.gnn_dropout for i in range(self.gnn_num_layers)],
			attn_drops=[self.gnn_dropout for i in range(self.gnn_num_layers)],
			alphas=[0.2 for i in range(self.gnn_num_layers)],
			residuals=[True for i in range(self.gnn_num_layers)],
			agg_modes=["flatten" for i in range(self.gnn_num_layers-1)] + ["mean"],
			activations=[nn.ELU(0.2) for i in range(self.gnn_num_layers-1)] + [None]
		)
		self.pool = construct_pool(self.gnn_hidden_feats,self.gnn_pool_type)
		if not self.separate_heads:
			self.ff_layers = construct_mlp(
				self.gnn_hidden_feats + self.grover_fp_feats,
				self.out_feats,
				self.ff_hidden_feats,
				self.ff_num_layers,
				self.ff_dropout
			)
		else:
			assert self.out_feats % 2 == 0
			self.ff_layers1 = construct_mlp(
				self.gnn_hidden_feats + self.grover_fp_feats,
				int(self.out_feats / 2),
				self.ff_hidden_feats,
				self.ff_num_layers,
				self.ff_dropout
			)
			self.ff_layers2 = construct_mlp(
				self.gnn_hidden_feats + self.grover_fp_feats,
				int(self.out_feats/2),
				self.ff_hidden_feats,
				self.ff_num_layers,
				self.ff_dropout
			)

	def forward(self, input_d):

		graph = input_d["graph"]
		node_h = graph.ndata["h"]
		node_h = self.gnn_layers(graph, node_h)
		global_h = self.pool(graph, node_h)

		if not self.separate_heads:
			if self.grover_fp in ['base', 'large']:
				output = self.ff_layers(th.cat([global_h, input_d["grover_fp"]], dim=1))
			else:
				output = self.ff_layers(global_h)

		else:
			if self.grover_fp in ['base', 'large']:
				output1 = self.ff_layers1(th.cat([global_h, input_d["grover_fp"]], dim=1))
				output2 = self.ff_layers2(th.cat([global_h, input_d["grover_fp"]], dim=1))
				output = th.cat([output1, output2], dim=1)
			else:
				output1 = self.ff_layers1(global_h)
				output2 = self.ff_layers2(global_h)
				output = th.cat([output1, output2], dim=1)

		return output

	def get_attention(self, graph, layer_idx=0):

		assert not self.training
		with th.no_grad():
			# run the model forward until reaching layer layer_idx
			graph = graph.local_var()
			feat = graph.ndata["h"]
			skip_gats = self.gnn_layers.gnn_layers[:layer_idx]
			for skip_gat in skip_gats:
				feat = skip_gat(graph,feat)
			# get attention weights of target GAT layer
			targ_gat = self.gnn_layers.gnn_layers[layer_idx].gat_conv
			h_src = h_dst = targ_gat.feat_drop(feat)
			feat_src = feat_dst = targ_gat.fc(h_src).view(
				-1, targ_gat._num_heads, targ_gat._out_feats)
			# NOTE: GAT paper uses "first concatenation then linear projection"
			# to compute attention scores, while ours is "first projection then
			# addition", the two approaches are mathematically equivalent:
			# We decompose the weight vector a mentioned in the paper into
			# [a_l || a_r], then
			# a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
			# Our implementation is much efficient because we do not need to
			# save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
			# addition could be optimized with DGL's built-in function u_add_v,
			# which further speeds up computation and saves memory footprint.
			el = (feat_src * targ_gat.attn_l).sum(dim=-1).unsqueeze(-1)
			er = (feat_dst * targ_gat.attn_r).sum(dim=-1).unsqueeze(-1)
			graph.srcdata.update({'ft': feat_src, 'el': el})
			graph.dstdata.update({'er': er})
			# compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
			graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
			e = targ_gat.leaky_relu(graph.edata.pop('e'))
			# compute softmax
			e_attn = targ_gat.attn_drop(edge_softmax(graph, e))
		# average attention over bidirectional edges (not vectorized)
		attn_dict = {}
		e_attn = e_attn.cpu()
		e_tups = zip(graph.edges()[0],graph.edges()[1])
		for e_idx, e_tup in enumerate(e_tups):
			if e_tup[0] <= e_tup[1]:
				e_tup = (int(e_tup[0]),int(e_tup[1]))
			else:
				e_tup = (int(e_tup[1]),int(e_tup[0]))
			if e_tup in attn_dict:
				attn_dict[e_tup] += 0.5*e_attn[e_idx].squeeze(1)
			else:
				attn_dict[e_tup] = 0.5*e_attn[e_idx].squeeze(1)
		return attn_dict



class GIN1(nn.Module):
	""" GIN model """

	def __init__(self,n_feats,e_feats,**kwargs):

		super(GIN1,self).__init__()
		self.n_feats = n_feats
		self.e_feats = e_feats
		for k,v in kwargs.items():
			setattr(self,k,v)
		self.gnn_layers = GIN(
			num_node_emb_list=self.n_feats,
			num_edge_emb_list=self.e_feats,
			num_layers=self.gnn_num_layers,
			emb_dim=self.gnn_hidden_feats,
			dropout=self.gnn_dropout
		)
		self.pool = construct_pool(self.gnn_hidden_feats,self.gnn_pool_type)
		self.ff_layers = construct_mlp(self.gnn_hidden_feats,self.out_feats,self.ff_hidden_feats,self.ff_num_layers,self.ff_dropout)

	def forward(self,input_d):

		graph = input_d["graph"]
		ndata_tensors = [ncol.data for ncol in graph.ndata.values()]
		edata_tensors = [ecol.data for ecol in graph.edata.values()]
		# import pdb; pdb.set_trace()
		node_h = self.gnn_layers(graph,ndata_tensors,edata_tensors)
		global_h = self.pool(graph,node_h)
		output = self.ff_layers(global_h)
		return output


class WLN1(nn.Module):
	""" WLN model """

	def __init__(self,n_feats,e_feats,**kwargs):

		super(WLN1,self).__init__()
		self.n_feats = n_feats
		self.e_feats = e_feats
		for k,v in kwargs.items():
			setattr(self,k,v)
		self.gnn_layers = WLN(
			self.n_feats,
			self.e_feats,
			self.gnn_hidden_feats,
			self.gnn_num_layers
		)
		self.pool = construct_pool(self.gnn_hidden_feats,self.gnn_pool_type)
		self.ff_layers = construct_mlp(self.gnn_hidden_feats,self.out_feats,self.ff_hidden_feats,self.ff_num_layers,self.ff_dropout)

	def forward(self,input_d):

		graph = input_d["graph"]
		node_h = graph.ndata["h"]
		edge_h = graph.edata["e"]
		# import pdb; pdb.set_trace()
		node_h = self.gnn_layers(graph,node_h,edge_h)
		global_h = self.pool(graph,node_h)
		output = self.ff_layers(global_h)
		return output


class GINPT(nn.Module):
	""" gin_supervised_contextpred_Tox21 """

	def __init__(self, n_feats, e_feats, pretrain_name, grover_fp_feats, **kwargs):

		super(GINPT,self).__init__()
		self.grover_fp_feats = grover_fp_feats # can be 0
		self.n_feats = n_feats
		self.e_feats = e_feats
		self.pretrain_name = pretrain_name
		for k,v in kwargs.items():
			setattr(self,k,v)
		assert len(n_feats) == 2 and n_feats[0] == 120 and n_feats[1] == 3, n_feats
		assert len(e_feats) == 2 and e_feats[0] == 6 and e_feats[1] == 3, e_feats
		self.gnn_layers = load_pretrained(self.pretrain_name)
		# remove the output layers
		if hasattr(self.gnn_layers,"readout"):
			self.gnn_layers.readout = GNIdentity()
		if hasattr(self.gnn_layers,"predict"):
			self.gnn_layers.predict = Identity()
		# fix pretrained weights
		if self.fix_pt_weights:
			# note: this is not the same as putting it in eval() mode
			# batchnorm/dropout still works as it does in train() mode
			for param in self.gnn_layers.parameters():
				param.requires_grad = False
		if self.pretrain_name in ["gin_supervised_contextpred_Tox21", "gin_supervised_infomax_Tox21"]:
			self.gnn_out_dim = 1800
		elif self.pretrain_name in ["gin_supervised_infomax_ToxCast", "gin_supervised_infomax"]:
			self.gnn_out_dim = 300
		else:
			raise NotImplementedError
		self.pool = construct_pool(self.gnn_out_dim,self.gnn_pool_type)
		if not self.separate_heads:
			self.ff_layers = construct_mlp(
				self.gnn_out_dim + self.grover_fp_feats,
				self.out_feats,
				self.ff_hidden_feats,
				self.ff_num_layers,
				self.ff_dropout
			)
		else:
			assert self.out_feats % 2 == 0
			self.ff_layers1 = construct_mlp(
				self.gnn_out_dim + self.grover_fp_feats,
				int(self.out_feats / 2),
				self.ff_hidden_feats,
				self.ff_num_layers,
				self.ff_dropout
			)
			self.ff_layers2 = construct_mlp(
				self.gnn_out_dim + self.grover_fp_feats,
				int(self.out_feats/2),
				self.ff_hidden_feats,
				self.ff_num_layers,
				self.ff_dropout
			)


	def forward(self,input_d):
		""" assumes graph is using pretrain featurization """

		graph = input_d["graph"]
		ndata_tensors = [ncol.data for ncol in graph.ndata.values()]
		edata_tensors = [ecol.data for ecol in graph.edata.values()]
		node_h = self.gnn_layers(graph, ndata_tensors, edata_tensors)
		global_h = self.pool(graph, node_h)

		if not self.separate_heads:
			if self.grover_fp in ['base', 'large']:
				output = self.ff_layers(th.cat([global_h, input_d["grover_fp"]], dim=1))
			else:
				output = self.ff_layers(global_h)

		else:
			if self.grover_fp in ['base', 'large']:
				output1 = self.ff_layers1(th.cat([global_h, input_d["grover_fp"]], dim=1))
				output2 = self.ff_layers2(th.cat([global_h, input_d["grover_fp"]], dim=1))
				output = th.cat([output1, output2], dim=1)
			else:
				output1 = self.ff_layers1(global_h)
				output2 = self.ff_layers2(global_h)
				output = th.cat([output1, output2], dim=1)

		return output


class GROVERPT(nn.Module):

	def __init__(self,grover_fp_feats,**kwargs):
		""" just an MLP on the grover embedding """

		super(GROVERPT,self).__init__()
		self.grover_fp_feats = grover_fp_feats
		assert self.grover_fp_feats > 0, self.grover_fp_feats
		for k,v in kwargs.items():
			setattr(self,k,v)
		self.ff_layers = construct_mlp(self.grover_fp_feats,self.out_feats,self.ff_hidden_feats,self.ff_num_layers,self.ff_dropout)

	def forward(self,input_d):
		""" ignores graph """

		grover_fp = input_d["grover_fp"]
		output = self.ff_layers(grover_fp)
		return output

# class GINPT1(nn.Module):
# 	""" gin_supervised_contextpred_Tox21 """

# 	def __init__(self,n_feats,e_feats,**kwargs):

# 		super(GINPT1,self).__init__()
# 		self.n_feats = n_feats
# 		self.e_feats = e_feats
# 		for k,v in kwargs.items():
# 			setattr(self,k,v)
# 		assert len(n_feats) == 2 and n_feats[0] == 120 and n_feats[1] == 3, n_feats
# 		assert len(e_feats) == 2 and e_feats[0] == 6 and e_feats[1] == 3, e_feats
# 		self.gnn_layers = load_pretrained("gin_supervised_contextpred_Tox21")
# 		# remove the output layers
# 		self.gnn_layers.readout = GNIdentity()
# 		self.gnn_layers.predict = Identity()
# 		# fix pretrained weights
# 		if self.fix_pt_weights:
# 			# note: this is not the same as putting it in eval() mode
# 			# batchnorm/dropout still works as it does in train() mode
# 			for param in self.gnn_layers.parameters():
# 				param.requires_grad = False
# 		self.gnn_out_dim = 1800
# 		self.pool = construct_pool(self.gnn_out_dim,self.gnn_pool_type)
# 		self.ff_layers = construct_mlp(self.gnn_out_dim,self.out_feats,self.ff_hidden_feats,self.ff_num_layers,self.ff_dropout)

# 	def forward(self,graph):
# 		""" assumes graph is using pretrain featurization """

# 		ndata_tensors = [ncol.data for ncol in graph.ndata.values()]
# 		edata_tensors = [ecol.data for ecol in graph.edata.values()]
# 		# import pdb; pdb.set_trace()
# 		node_h = self.gnn_layers(graph,ndata_tensors,edata_tensors)
# 		global_h = self.pool(graph,node_h)
# 		output = self.ff_layers(global_h)
# 		return output


class Conv1dBlock(nn.Module):

	def __init__(self,in_channels,out_channels,kernel_size=5,stride=1,pool_type="avg",pool_size=4,batch_norm=True,dropout=0.1,activation="relu"):
		"""
		padding is always the same
		order of ops:
		1. conv
		2. batch norm (if applicable)
		3. activation
		4. pool (if applicable)
		5. dropout (if applicable)
		"""

		super(Conv1dBlock,self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		ops_list = []
		assert kernel_size % 2 == 1, kernel_size
		padding = kernel_size // 2
		ops_list.append(nn.Conv1d(in_channels,out_channels,kernel_size,stride=stride,padding=padding))
		if batch_norm:
			ops_list.append(nn.BatchNorm1d(out_channels))
		if activation == "relu":
			ops_list.append(nn.ReLU())
		else:
			raise ValueError
		if pool_type == "max":
			ops_list.append(nn.MaxPool1d(pool_size,ceil_mode=True))
		elif pool_type == "avg":
			ops_list.append(nn.AvgPool1d(pool_size,ceil_mode=True))
		else:
			assert pool_type == "none", pool_type
		if dropout > 0.:
			ops_list.append(nn.Dropout(dropout))
		self.ops = nn.Sequential(*ops_list)

	def forward(self,b_x):

		assert b_x.ndim == 3, b_x.ndim
		assert b_x.shape[1] == self.in_channels, b_x.shape
		b_y = self.ops(b_x)
		return b_y


class CNN(nn.Module):

	def __init__(self,c_feats,l_feats,**kwargs):

		super(CNN,self).__init__()
		self.c_feats = c_feats
		self.l_feats = l_feats
		for k,v in kwargs.items():
			setattr(self,k,v)

		conv_hp = dict(
			kernel_size=self.conv_kernel_size,
			stride=self.conv_stride,
			pool_type=self.conv_pool_type,
			pool_size=self.conv_pool_size,
			dropout=self.ff_dropout,
			activation="relu"
		)

		def compute_conv_out_length(in_length,kernel_size,stride):
			padding = kernel_size // 2
			dilation = 1
			out_length = np.floor((in_length + 2*padding - dilation * (kernel_size-1) - 1) / stride + 1)
			return int(out_length)

		def compute_pool_out_length(in_length,kernel_size):
			if self.conv_pool_type == "none":
				return in_length
			else:
				padding = 0
				dilation = 1
				stride = kernel_size
				out_length = np.ceil((in_length + 2*padding - dilation * (kernel_size-1) - 1) / stride + 1)
				return int(out_length)
		
		conv_list = []
		conv_list.append(Conv1dBlock(self.c_feats,self.conv_num_kernels,**conv_hp))
		conv_out_length = compute_conv_out_length(self.l_feats,self.conv_kernel_size,self.conv_stride)
		pool_out_length = compute_pool_out_length(conv_out_length,self.conv_pool_size)
		# print(0,conv_out_length,pool_out_length)
		for i in range(self.conv_num_layers-1):
			conv_list.append(Conv1dBlock(self.conv_num_kernels,self.conv_num_kernels,**conv_hp))
			conv_out_length = compute_conv_out_length(pool_out_length,self.conv_kernel_size,self.conv_stride)
			pool_out_length = compute_pool_out_length(conv_out_length,self.conv_pool_size)
			# print(i+1,conv_out_length,pool_out_length)
		self.conv_layers = nn.Sequential(*conv_list)

		self.ff_layers = construct_mlp(pool_out_length*self.conv_num_kernels,self.out_feats,self.ff_hidden_feats,self.ff_num_layers,self.ff_dropout)

	def forward(self, input_d):

		seq = input_d["seq"]
		# assumes seq is padded
		batch_size = seq.shape[0]
		ch = self.conv_layers(seq)
		ch = ch.view(batch_size,-1)
		fh = self.ff_layers(ch)
		return fh

def construct_mlp(in_dim, out_dim, hidden_dim, hidden_layers, dropout, different_heads=False):
	""" 
	total number of layers = hidden_layers + 1 (the + 1 is for the output linear)
	no activation/batch norm/dropout on output layer
	"""

	assert hidden_layers >= 1, hidden_layers
	mlp_list = []
	mlp_list.append(LinearBlock(in_dim,hidden_dim,dropout=dropout))
	for i in range(hidden_layers-1):
		mlp_list.append(LinearBlock(hidden_dim,hidden_dim,dropout=dropout))
	mlp_list.append(nn.Linear(hidden_dim,out_dim))
	mlp = nn.Sequential(*mlp_list)
	return mlp

def construct_pool(node_dim,pool_type):

	if pool_type == "max":
		return MaxPooling()
	elif pool_type == "avg":
		return AvgPooling()
	elif pool_type == "attn":
		pool_gate_nn = nn.Linear(node_dim, 1)
		return GlobalAttentionPooling(pool_gate_nn)


def test_forward_backward(model_type):

	from data_loader import get_datasets
	from arguments import get_args
	args = get_args()
	print(model_type)
	args["model_type"] = model_type
	if model_type == "grover_pt":
		args["grover_fp"] = "base"
	ds = get_datasets(args)[0]
	data_feats_d = ds.get_data_feats()
	model = get_model(args, data_feats_d)
	collate_fn = ds.get_collate_fn()
	d0 = ds[0]
	d1 = ds[1]
	d2 = ds[2]
	bd = collate_fn([d0, d1, d2])
	# print(g)
	o = model(bd)
	print(o)
	assert pd.Series(bd["mut_label"]).isnull().sum() == 0
	l = th.mean((bd["mut_label"]-o[:, 0])**2)
	print(l)
	l.backward()
	import ipdb;
	ipdb.set_trace()


def test_attention_viz():

	from data_loader import get_datasets
	from arguments import get_args
	from plot_utils import viz_attention
	args = get_args()
	args["model_type"] = "gat"
	ds = get_datasets(args)[0]
	data_feats_d = ds.get_data_feats()
	model = get_model(args, data_feats_d)
	collate_fn = ds.get_collate_fn()
	d0 = ds[0]
	model.eval()
	attn_dict = model.get_attention(d0["graph"])
	print(attn_dict)
	im = viz_attention(smiles,attn_dict)
	im.save("test.png")
	

if __name__ == "__main__":

	# test_attention_viz()
	test_forward_backward("mlp_fingerprint")
	test_forward_backward("grover_pt")
	test_forward_backward("gat")
	# test_forward_backward("gin_pt1")
	test_forward_backward("gin_pt2")
	# test_forward_backward("gin_pt3")
	# test_forward_backward("gin_pt4")
	test_forward_backward("cnn")
	test_forward_backward("wln")
