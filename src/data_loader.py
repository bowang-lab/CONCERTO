from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

from rdkit.Chem import Draw
import pandas as pd
from datetime import datetime
import dgllife
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer, PretrainAtomFeaturizer, PretrainBondFeaturizer
from dgllife.utils import ScaffoldSplitter
from dgllife.utils.splitters import prepare_mols
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import selfies as sf
import re
from dgllife.utils import smiles_to_bigraph, EarlyStopping
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import warnings

from utils import suppress_output, np_temp_seed, th_temp_seed


def get_datasets(args):

	model_type = args["model_type"]
	kwargs = {}
	kwargs["grover_fp"] = args["grover_fp"]
	if model_type in ["gat_p","gat","gin","gin_pt1","gin_pt2","gin_pt3","gin_pt4","grover_pt","wln"]:
		if model_type in ["grover_pt"]:
			assert args["grover_fp"] != "none"
		if model_type in ["gin","gin_pt1","gin_pt2","gin_pt3","gin_pt4"]:
			kwargs["featurizer"] = "pretrain"
		elif model_type in ["gat_p","gat","wln"]:
			kwargs["featurizer"] = "canonical"
		model_class = GraphCancerMolecules

	elif model_type in ["cnn", 'mlp_fingerprint']:
		kwargs["max_seq_len"] = args["max_seq_len"]
		kwargs['atom_pairs_fingerprints'] = args["atom_pairs_fingerprints"]
		kwargs['torsion_fingerprints'] = args["torsion_fingerprints"]
		kwargs['fp_nbits'] = args['fp_nbits']
		model_class = SelfiesCancerMolecules

	else:
		raise ValueError(f"invalid model_type: {model_type}")

	# Use carc prob for training data based on argparse
	train_data = model_class(
		carc_datasets=args['training_carc_datasets'],
		mut_datasets=args['training_mut_datasets'],
		out_feats=args['out_feats'],
		fraction_of_data=args['fraction_of_data'],
		min_carbon_count=args['min_carbon_count'],
		use_carc_prob=args['use_carc_prob'],
		carc_percentile_to_drop=args['carc_percentile_to_drop'],
		**kwargs
	)

	if isinstance(train_data, SelfiesCancerMolecules):
		kwargs["stoi"] = train_data.stoi

	# don't use carc prob on held out test data, don't drop the top probabilities
	held_out_test_data = model_class(
		carc_datasets=args['held_out_test_carc_datasets'],
		mut_datasets=args['held_out_test_mut_datasets'],
		out_feats=args['out_feats'],
		fraction_of_data=args['fraction_of_data'],
		min_carbon_count=args['min_carbon_count'],
		**kwargs
	)
	return train_data, held_out_test_data


def get_explain_dataset(args,smileses,base_fp_df,large_fp_df):

	model_type = args["model_type"]
	kwargs = {}
	if model_type in ["gin","gin_pt1","gin_pt2","gin_pt3","gin_pt4"]:
		kwargs["featurizer"] = "pretrain"
	elif model_type in ["gat_p","gat","wln"]:
		kwargs["featurizer"] = "canonical"
	kwargs["max_seq_len"] = args["max_seq_len"]
	kwargs['atom_pairs_fingerprints'] = args["atom_pairs_fingerprints"]
	kwargs['torsion_fingerprints'] = args["torsion_fingerprints"]
	kwargs['fp_nbits'] = args['fp_nbits']

	kwargs["smileses"] = smileses
	if args["grover_fp"] == "base":
		kwargs["grover_fp_df"] = base_fp_df
		kwargs["use_grover"] = True
	elif args["grover_fp"] == "large":
		kwargs["grover_fp_df"] = large_fp_df
		kwargs["use_grover"] = True
	else:
		kwargs["use_grover"] = False
	kwargs["use_graph"] = args["model_type"] in ["gat_p","gat","gin","gin_pt1","gin_pt2","gin_pt3","gin_pt4","grover_pt","wln"]
	kwargs["use_seq"] = args["model_type"] in ["cnn"]
	kwargs["use_fingerprint"] = args["model_type"] in ["mlp_fingerprint"]
	kwargs["drop_ionic"] = True # not an argument
	kwargs["min_carbon_count"] = args["min_carbon_count"]

	ds = ExplainMolecules(**kwargs)

	return ds

class CancerMolecules(Dataset):
	
	def __init__(
			self,
			carc_datasets=('carc_cpdb',),
			mut_datasets=('mut_hansen',),
			out_feats=1,
			drop_ionic=True,
			min_carbon_count=0,
			fraction_of_data=1,
			use_carc_prob=False,
			carc_percentile_to_drop=0,
	):
		self.allowed_carc_datasets = ['carc_cpdb', 'carc_ccris', 'all_data', 'carc_caps_pred_el']
		self.allowed_mut_datasets = ['mut_hansen', 'mut_ccris', 'mut_li', 'all_data']
		assert all(dataset in self.allowed_carc_datasets for dataset in carc_datasets), \
			f"carcinogenic datasets should all be one of the following values: {self.allowed_carc_datasets}" \
			f" not {carc_datasets}"
		assert all(dataset in self.allowed_mut_datasets for dataset in mut_datasets), \
			f"mutagenic datasets should all be one of the following values: {self.allowed_mut_datasets}" \
			f" not {mut_datasets}"

		if 'all_data' in mut_datasets:
			mut_datasets = tuple(self.allowed_mut_datasets)
		if 'all_data' in carc_datasets:
			carc_datasets = tuple(self.allowed_carc_datasets)

		super(CancerMolecules, self).__init__()
		self.carc_datasets = carc_datasets
		self.mut_datasets = mut_datasets
		self.out_feats = out_feats
		self.drop_ionic = drop_ionic
		self.min_carbon_count = min_carbon_count
		self.fraction_of_data = fraction_of_data
		self.use_carc_prob = use_carc_prob
		self.carc_percentile_to_drop = carc_percentile_to_drop

	@classmethod
	def count_num_carbons(cls, smile):
		mol = Chem.MolFromSmiles(smile)
		num_c = 0
		for atom in mol.GetAtoms():
			if atom.GetSymbol().upper() == 'C':
				num_c += 1
		return num_c

	@classmethod
	def load_carc_ccris(cls):
		df = pd.read_csv('../data/ccri_smiles.csv')
		carc_neg = df[(df['ratio_of_positive_carc'] <= 0) & (df['number_of_experiments_carc'] >= 1)][['smiles']]
		carc_neg['carc_class'] = 0
		carc_pos = df[(df['ratio_of_positive_carc'] >= 0.6) & (df['number_of_experiments_carc'] >= 2)][['smiles']]
		carc_pos['carc_class'] = 1
		carc_ccris = pd.concat([carc_neg, carc_pos])
		carc_ccris['source'] = 'ccris'
		carc_ccris['smiles'] = cls.smiles_standardize(carc_ccris['smiles'].values)
		carc_ccris = carc_ccris[carc_ccris['smiles'].notnull()]
		carc_ccris['source'] = 'carc_ccris'
		return carc_ccris

	@classmethod
	def load_mut_ccris(cls):
		df = pd.read_csv('../data/ccri_smiles.csv')
		mut_pos = df[(df['ratio_of_positive_mut'] <= 0) & (df['number_of_experiments_mut'] >= 1)][['smiles']]
		mut_pos['mut_class'] = 0
		mut_neg = df[(df['ratio_of_positive_mut'] >= 0.54) & (df['number_of_experiments_mut'] >= 6)][['smiles']]
		mut_neg['mut_class'] = 1
		mut_ccris = pd.concat([mut_pos, mut_neg])
		mut_ccris['source'] = 'ccris'
		mut_ccris['smiles'] = cls.smiles_standardize(mut_ccris['smiles'].values)
		mut_ccris = mut_ccris[mut_ccris['smiles'].notnull()]
		mut_ccris['source'] = 'mut_ccris'
		return mut_ccris

	@classmethod
	def load_mut_li(cls):
		df = pd.read_csv('../data/mutagenicity_li_2021.csv', usecols=['SMILES', 'Mutagenicity'])
		df.rename(columns={'SMILES': 'smiles', 'Mutagenicity': 'mut_class'}, inplace=True)
		df['smiles'] = cls.smiles_standardize(df['smiles'])
		df = df[df['smiles'].notnull()]
		df['source'] = 'li'
		return df

	@classmethod
	def load_carc_cpdb(cls):
		cpdb = pd.read_csv(
			'../data/cpdb_aggergated.csv',
			usecols=['smiles', 'td50_log_harmonic', 'cas', 'carc_class', 'carc_class_multi']
		)
		cpdb['smiles'] = cls.smiles_standardize(cpdb['smiles'].values)
		cpdb = cpdb[cpdb['smiles'].notnull()]
		cpdb.rename(columns={'td50_log_harmonic': 'td50'}, inplace=True)
		assert cpdb['smiles'].duplicated().sum() == 0, \
			cpdb[cpdb['smiles'].duplicated(keep=False)].sort_values('smiles').to_string()
		cpdb['source'] = 'cpdb'
		return cpdb

	@classmethod
	def load_mut_hansen(cls):
		ames_df = pd.read_csv('../data/hansen_2009_ames.smi', sep='\t', names=['smiles', 'cas', 'class'])
		ames_df['smiles'] = cls.smiles_standardize(ames_df['smiles'].values)
		ames_df = ames_df[ames_df['smiles'].notnull()]
		ames_df.rename(columns={'class': 'mut_class'}, inplace=True)
		ames_df.drop(columns='cas', inplace=True)
		ames_df = ames_df[~ames_df.duplicated(['smiles', 'mut_class'])]
		assert ames_df['smiles'].duplicated().sum() == 0, \
			ames_df[ames_df['smiles'].duplicated(keep=False)].sort_values('smiles').to_string()
		ames_df['source'] = 'hansen'
		return ames_df

	@classmethod
	def load_carcino_pred_el_data(cls):
		carcino_df = pd.read_csv('../data/carcino_pred_el_extracted_smiles.csv')
		carcino_df['smiles'] = cls.smiles_standardize(carcino_df['smiles'].values)
		carcino_df = carcino_df[carcino_df['smiles'].notnull()]
		conversion_dict = {'Carcinogen':1, 'Non-Carcinogen':0}
		carcino_df['carc_class'] = [conversion_dict[x] for x in carcino_df['status'].values]
		carcino_df = carcino_df[~carcino_df.duplicated(['smiles', 'carc_class'])]
		assert carcino_df['smiles'].duplicated().sum() == 0, \
			carcino_df[carcino_df['smiles'].duplicated(keep=False)].sort_values('smiles').to_string()
		carcino_df['source'] = 'caps_pred_el'
		return carcino_df[['smiles', 'carc_class', 'source']]

	def load_data(self):
		# Loaded data should not overlap mut_hansn and carc_cpdb since they are the two primary training data sources
		mut = self.load_mut_hansen()
		carc = self.load_carc_cpdb()
		carc2 = self.load_carcino_pred_el_data()

		carc_datasets = []
		mut_datasets = []
		if 'carc_cpdb' in self.carc_datasets:
			carc_datasets.append(carc)

		if 'carc_caps_pred_el' in self.carc_datasets:
			carc_datasets.append(carc2)

		if 'carc_ccris' in self.carc_datasets:
			temp = self.load_carc_ccris()
			# drop samples that are in training data by defaul cpdb & carc_pred_el
			temp = temp[~temp['smiles'].isin(carc['smiles'])]
			temp = temp[~temp['smiles'].isin(carc2['smiles'])]
			carc_datasets.append(temp)

		if 'mut_hansen' in self.mut_datasets:
			mut_datasets.append(mut)

		if 'mut_li' in self.mut_datasets:
			temp = self.load_mut_li()
			# drop samples that are in training data by defaul cpdb
			temp = temp[~temp['smiles'].isin(mut['smiles'])]
			mut_datasets.append(temp)

		if 'mut_ccris' in self.mut_datasets:
			temp = self.load_mut_ccris()
			# drop samples that are in training data by defaul cpdb
			temp = temp[~temp['smiles'].isin(mut['smiles'])]
			mut_datasets.append(temp)

		if mut_datasets:
			mut = pd.concat(mut_datasets)
		else:
			mut = pd.DataFrame(columns=mut.columns)

		if carc_datasets:
			carc = pd.concat(carc_datasets)
		else:
			carc = pd.DataFrame(columns=carc.columns)

		if mut['smiles'].duplicated().sum() != 0:
			warnings.warn(f"duplicated samples {mut['smiles'].duplicated().sum()} contained in mutagenicity data "
						  f"from {self.mut_datasets}\n"
						  f" {mut[mut['smiles'].duplicated(keep=False)].sort_values('smiles').head().to_string()}")
			mut.sort_values(['smiles', 'mut_class'], ascending=False)
			mut = mut[~mut.duplicated('smiles')]

		if carc['smiles'].duplicated().sum() != 0:
			warnings.warn(f"duplicated samples {carc['smiles'].duplicated().sum()} contained in carcinogenicity data "
						  f"from {self.carc_datasets}\n"
						  f" {carc[carc['smiles'].duplicated(keep=False)].sort_values('smiles').head().to_string()}")
			carc = carc[~carc.duplicated('smiles')]

		df = pd.merge(mut, carc, how='outer', on='smiles')
		df['source'] = df['source_x'].fillna('') + ',' + df['source_y'].fillna('')
		df.drop(columns=['source_x', 'source_y'], inplace=True)

		required_columns = ['td50', 'carc_class', 'mut_class', 'carc_class_multi']
		for column in required_columns:
			if column not in df.columns:
				df[column] = np.nan

		if self.drop_ionic:
			df = df[~df['smiles'].str.contains('\.')]

		df['carbon_count'] = [self.count_num_carbons(x) for x in df['smiles'].values]
		if self.min_carbon_count:
			df = df[df['carbon_count'] >= self.min_carbon_count]

		# Shuffle the data
		index = np.arange(len(df))
		with np_temp_seed(1337):
			np.random.shuffle(index)
		df = df.iloc[index].reset_index(drop=True)

		# Log of Betas that were fitted using a cox regression model
		if self.carc_percentile_to_drop > 0:
			# cleave off the top percentile
			lowest_td_50_val = np.nanpercentile(df['td50'].values, self.carc_percentile_to_drop)
			mask = df[df['td50'] <= lowest_td_50_val].index
			df.loc[mask, 'td50'] = lowest_td_50_val

		betas = np.log((np.log(2) / df["td50"].values))

		# standardize
		beta_standardized = (betas - np.nanmean(betas)) / np.nanstd(betas)
		# normalize
		beta_normalized = betas - np.nanmin(betas)
		beta_normalized = beta_normalized / np.nanmax(beta_normalized)

		df['beta_standardized'] = beta_standardized
		df['beta_normalized'] = beta_normalized

		if self.fraction_of_data < 1:
			new_data_len = int(len(df) * self.fraction_of_data)
			df = df.iloc[:new_data_len]

		self.df = df

	@classmethod
	def smiles_standardize(cls, smiles):
		new_smiles = []
		for smile in smiles:
			if pd.isnull(smile):
				new_smiles.append(None)
				continue
			# Generate the molecule from smile string
			mol = Chem.MolFromSmiles(smile)
			# If the smile string is null then continue
			if pd.isnull(mol):
				new_smiles.append(None)
			else:
				new_smiles.append(Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False,allBondsExplicit=False))
		return new_smiles

	def load_grover_fps(self):

		# get grover stuff
		if self.grover_fp != "none":
			if self.grover_fp == "base":
				grover_fp_df = pd.read_pickle('../data/grover_base_fp.pkl')
			elif self.grover_fp == "large":
				grover_fp_df = pd.read_pickle('../data/grover_large_fp.pkl')
			len_old_df = len(self.df)
			grover_fp_df = grover_fp_df[~grover_fp_df['smiles'].duplicated()]
			self.df = self.df.merge(grover_fp_df, how="left", on=["smiles"])
			assert len(self.df) == len_old_df
			self.df = self.df.rename(columns={"fps": "grover_fp"})

	def process(self):
		raise NotImplementedError

	def __getitem__(self, i):
		raise NotImplementedError

	def __len__(self):
		raise NotImplementedError

	def get_collate_fn(self):
		raise NotImplementedError


class GraphCancerMolecules(CancerMolecules):

	def __init__(
			self,
			carc_datasets=('carc_cpdb',),
			mut_datasets=('mut_hansen',),
			featurizer='canonical',
			out_feats=1,
			grover_fp="none",
			drop_ionic=True,
			min_carbon_count=0,
			fraction_of_data=1,
			use_carc_prob=False,
			carc_percentile_to_drop=0,
			**kwargs
	):

		super(GraphCancerMolecules, self).__init__(
			carc_datasets=carc_datasets,
			mut_datasets=mut_datasets,
			out_feats=out_feats,
			drop_ionic=drop_ionic,
			min_carbon_count=min_carbon_count,
			fraction_of_data=fraction_of_data,
			use_carc_prob=use_carc_prob,
			carc_percentile_to_drop=carc_percentile_to_drop,
		)
		self.featurizer = featurizer
		self.grover_fp = grover_fp
		self.process()
		self.out_feats = out_feats
		self.use_carc_prob = use_carc_prob

	def process(self):
		"""
		Inputs:
			df is a pd.DataFrame with columns 'smiles' and 'class'
		"""

		self.load_data()

		if self.featurizer == "canonical":
			node_featurizer = CanonicalAtomFeaturizer()
			edge_featurizer = CanonicalBondFeaturizer(self_loop=True)

		elif self.featurizer == "pretrain":
			node_featurizer = PretrainAtomFeaturizer(atomic_number_types=list(range(120)))
			edge_featurizer = PretrainBondFeaturizer()

		else:
			raise ValueError

		with suppress_output(stdout=True, stderr=False):
			bigraphs = []
			for smile in self.df['smiles'].values:
				bigraphs.append(
					dgllife.utils.smiles_to_bigraph(
						smile,
						add_self_loop=True,
						node_featurizer=node_featurizer,
						edge_featurizer=edge_featurizer,
						canonical_atom_order=False,
					))
			index_to_keep = np.array([pd.notnull(bigraph) for bigraph in bigraphs])

		self.load_grover_fps()

		# instantiate
		if self.grover_fp != "none":
			index_to_keep = self.df["grover_fp"].notnull().values & index_to_keep

		print(self.df.shape, index_to_keep.sum())

		self.df = self.df.iloc[index_to_keep].reset_index(drop=True)
		self.smiles = self.df["smiles"].values
		self.graphs = bigraphs

		self.carc_continuous = torch.tensor(self.df['beta_standardized']).to(torch.float)
		self.carc_mask_continuous = torch.tensor(self.df['beta_standardized'].notnull().values)

		self.carc_prob = torch.tensor(self.df['beta_normalized']).to(torch.float)
		self.carc_labels = torch.tensor(self.df["carc_class"]).to(torch.float)
		self.carc_mask = torch.tensor(self.df['carc_class'].notnull().values)

		self.carc_labels_multi = torch.tensor(self.df["carc_class_multi"]).to(torch.long)
		self.carc_mask_multi = torch.tensor(self.df['carc_class_multi'].notnull().values)

		self.mut_labels = torch.tensor(self.df["mut_class"]).to(torch.float)
		self.mut_mask = torch.tensor(self.df['mut_class'].notnull().values)

		if self.grover_fp != "none":
			self.grover_fps = torch.tensor(self.df["grover_fp"], dtype=torch.float)

	def __getitem__(self, i):
		item_d = {}
		item_d["graph"] = self.graphs[i]
		item_d["smiles"] = self.smiles[i]

		item_d["mut_label"] = self.mut_labels[i]
		item_d["mut_mask"] = self.mut_mask[i]

		item_d["carc_mask_continuous"] = self.carc_mask_continuous[i]
		item_d["carc_continuous"] = self.carc_continuous[i]

		item_d["carc_prob"] = self.carc_prob[i]
		item_d["carc_label"] = self.carc_labels[i]
		item_d["carc_mask"] = self.carc_mask[i]

		item_d["carc_label_multi"] = self.carc_labels_multi[i]
		item_d["carc_mask_multi"] = self.carc_mask_multi[i]

		if self.grover_fp != "none":
			item_d["grover_fp"] = self.grover_fps[i]
		return item_d

	def __len__(self):
		return self.df.shape[0]

	def get_collate_fn(self):
		def _collate(data_list):
			batch_d = {}
			graphs, smileses, grover_fps = [], [], []
			mut_labels, mut_mask, carc_mask_multi, carc_labels_multi = [], [], [], []
			carc_mask, carc_labels, carc_continuous, carc_prob, carc_mask_continuous = [], [], [], [], []

			for d in data_list:
				graphs.append(d["graph"])
				smileses.append(d["smiles"])

				mut_labels.append(d["mut_label"])
				mut_mask.append(d['mut_mask'])

				carc_labels_multi.append(d["carc_label_multi"])
				carc_mask_multi.append(d['carc_mask_multi'])

				carc_labels.append(d["carc_label"])
				carc_prob.append(d['carc_prob'])
				carc_mask.append(d['carc_mask'])

				carc_continuous.append(d['carc_continuous'])
				carc_mask_continuous.append(d['carc_mask_continuous'])

				if self.grover_fp != "none":
					grover_fps.append(d["grover_fp"])

			batch_d["batch_size"] = len(graphs)
			batch_d["graph"] = dgl.batch(graphs)
			batch_d["smiles"] = smileses

			batch_d["mut_label"] = torch.stack(mut_labels, dim=0)
			batch_d["mut_mask"] = torch.stack(mut_mask, dim=0)

			batch_d["carc_label"] = torch.stack(carc_labels, dim=0)
			batch_d["carc_prob"] = torch.stack(carc_prob, dim=0)
			batch_d["carc_mask"] = torch.stack(carc_mask, dim=0)

			batch_d["carc_label_multi"] = torch.stack(carc_labels_multi, dim=0)
			batch_d["carc_mask_multi"] = torch.stack(carc_mask_multi, dim=0)

			batch_d["carc_continuous"] = torch.stack(carc_continuous, dim=0)
			batch_d["carc_mask_continuous"] = torch.stack(carc_mask_continuous, dim=0)

			if self.grover_fp != "none":
				batch_d["grover_fp"] = torch.stack(grover_fps, dim=0)
			return batch_d
		return _collate

	def get_data_feats(self):

		if self.featurizer == "canonical":
			graph = self.graphs[0]
			n_feats = graph.ndata["h"].shape[1]
			e_feats = graph.edata["e"].shape[1]
		else:
			n_feats = [120, 3]
			e_feats = [6, 3]
		data_feats_d = dict(n_feats=n_feats,e_feats=e_feats)
		if self.grover_fp != "none":
			data_feats_d["grover_fp_feats"] = self.grover_fps.shape[1]
		else:
			data_feats_d["grover_fp_feats"] = 0
		return data_feats_d


class SelfiesCancerMolecules(CancerMolecules):

	def __init__(
			self,
			carc_datasets=('carc_cpdb',),
			mut_datasets=('mut_hansen',),
			max_seq_len=None,
			stoi=None,
			drop_ionic=True,
			min_carbon_count=0,
			grover_fp="none",
			out_feats=1,
			fraction_of_data=1,
			use_carc_prob=False,
			carc_percentile_to_drop=0,
			torsion_fingerprints=True,
			atom_pairs_fingerprints=True,
			fp_nbits=1024,
			**kwargs
	):

		super(SelfiesCancerMolecules, self).__init__(
			carc_datasets=carc_datasets,
			mut_datasets=mut_datasets,
			out_feats=out_feats,
			drop_ionic=drop_ionic,
			min_carbon_count=min_carbon_count,
			fraction_of_data=fraction_of_data,
			use_carc_prob=use_carc_prob,
			carc_percentile_to_drop=carc_percentile_to_drop
		)
		self.max_seq_len = max_seq_len
		self.stoi = stoi
		self.grover_fp = grover_fp
		self.torsion_fingerprints = torsion_fingerprints
		self.atom_pairs_fingerprints = atom_pairs_fingerprints
		self.fp_nbits = fp_nbits
		self.process()

	def convert_smile_to_fp_bit_string(self, smile):
		"""
		RDFKIT Morgan and MACCSS are default fingerprints. Torsion and atom pairs are optional
		"""
		# RDFKIT
		x = Chem.MolFromSmiles(smile)
		fp1 = Chem.RDKFingerprint(x, fpSize=self.fp_nbits)
		# MACCSS substructure
		fp2 = MACCSkeys.GenMACCSKeys(x)
		# Morgan
		fp_hashes = []
		fp3 = AllChem.GetHashedMorganFingerprint(x, 2, nBits=self.fp_nbits)
		fp3_array = np.zeros((0,), dtype=np.int8)
		DataStructs.ConvertToNumpyArray(fp3, fp3_array)
		fp_hashes.append(fp3_array)

		# Hashed atom pairs
		if self.atom_pairs_fingerprints:
			fp4 = AllChem.GetHashedAtomPairFingerprint(x, nBits=self.fp_nbits)
			fp4_array = np.zeros((0,), dtype=np.int8)
			DataStructs.ConvertToNumpyArray(fp4, fp4_array)
			fp_hashes.append(fp4_array)

		# Torsion fingerprints?
		if self.torsion_fingerprints:
			fp5 = AllChem.GetHashedTopologicalTorsionFingerprint(x, nBits=self.fp_nbits)
			fp5_array = np.zeros((0,), dtype=np.int8)
			DataStructs.ConvertToNumpyArray(fp5, fp5_array)
			fp_hashes.append(fp5_array)

		fp = fp1.ToBitString() + fp2.ToBitString()
		fp = np.array(list(fp)).astype(np.int8)
		fp = np.concatenate([fp] + fp_hashes)
		fp = torch.tensor(fp).to(torch.float32)
		return fp

	def process(self):
		"""
		Inputs:
			df is a pd.DataFrame with columns 'smiles' and 'class'
		"""

		self.load_data()

		def sf_encode(smiles):
			try:
				return sf.encoder(smiles)
			except:
				return np.nan
		def is_multi_mol(selfies):
			if selfies is np.nan:
				return False
			else:
				return "." in selfies
		def is_too_long(selfies):
			if selfies is np.nan:
				return False
			else:
				return len(list(sf.split_selfies(selfies))) > self.max_seq_len
		
		self.df.loc[:,"selfies"] = self.df["smiles"].apply(sf_encode)
		multi_mol = self.df["selfies"].apply(is_multi_mol)
		seq_len = self.df["selfies"].apply(is_too_long)
		print(f"To be dropped: Num multi molecules: {multi_mol.sum()}, Num sequence exceeding length {seq_len.sum()}")
		# tokenize selfies strings
		if self.stoi is None:
			selfies = self.df["selfies"].dropna()
			self.alphabet = sf.get_alphabet_from_selfies(selfies.values)
			self.alphabet.add("[nop]")
			self.alphabet.add(".")
			self.stoi = {s: i for i,s in enumerate(sorted(self.alphabet))}
		else:
			self.alphabet = set(self.stoi.keys())
		num_tokens = len(self.stoi)
		pad_token = self.stoi["[nop]"]

		def selfies_to_encoding(selfies):
			if selfies is np.nan:
				return np.nan
			else:
				token_labels = []
				for idx,token in enumerate(sf.split_selfies(selfies)):
					if idx == self.max_seq_len:
						# crop it!
						break
					if token in self.stoi:
						token_labels.append(self.stoi[token])
					else:
						token_labels.append(num_tokens)
				for i in range(len(token_labels),self.max_seq_len):
					token_labels.append(pad_token)
				return token_labels
		self.df.loc[:,"selfies_tokens"] = self.df["selfies"].apply(selfies_to_encoding)

		self.load_grover_fps()

		# instantiate
		if self.grover_fp != "none":
			index_to_keep = np.array(~(self.df["selfies"].isna()) & ~(self.df["grover_fp"].isna()))
		else:
			index_to_keep = np.array(~(self.df["selfies"].isna()))

		print(self.df.shape, index_to_keep.sum())

		self.df = self.df.iloc[index_to_keep].reset_index(drop=True)
		self.smiles = self.df["smiles"].values
		self.fingerprints = [self.convert_smile_to_fp_bit_string(smile) for smile in self.df['smiles'].values]
		self.selfies_tokens = torch.tensor(self.df["selfies_tokens"],dtype=torch.long)

		self.carc_continuous = torch.tensor(self.df['beta_standardized']).to(torch.float)
		self.carc_mask_continuous = torch.tensor(self.df['beta_standardized'].notnull().values)

		self.carc_prob = torch.tensor(self.df['beta_normalized']).to(torch.float)
		self.carc_labels = torch.tensor(self.df["carc_class"]).to(torch.float)
		self.carc_mask = torch.tensor(self.df['carc_class'].notnull().values)

		self.carc_labels_multi = torch.tensor(self.df["carc_class_multi"]).to(torch.long)
		self.carc_mask_multi = torch.tensor(self.df['carc_class_multi'].notnull().values)

		self.mut_labels = torch.tensor(self.df["mut_class"]).to(torch.float)
		self.mut_mask = torch.tensor(self.df['mut_class'].notnull().values)

		if self.grover_fp != "none":
			self.grover_fps = torch.tensor(self.df["grover_fp"], dtype=torch.float)

	def __getitem__(self, i):

		item_d = {}
		item_d["seq"] = F.one_hot(self.selfies_tokens[i], num_classes=len(self.stoi)+1).float().T
		item_d["smiles"] = self.smiles[i]
		item_d['fingerprints'] = self.fingerprints[i]

		item_d["mut_label"] = self.mut_labels[i]
		item_d["mut_mask"] = self.mut_mask[i]

		item_d["carc_mask_continuous"] = self.carc_mask_continuous[i]
		item_d["carc_continuous"] = self.carc_continuous[i]

		item_d["carc_label"] = self.carc_labels[i]
		item_d["carc_prob"] = self.carc_prob[i]
		item_d["carc_mask"] = self.carc_mask[i]

		item_d["carc_label_multi"] = self.carc_labels_multi[i]
		item_d["carc_mask_multi"] = self.carc_mask_multi[i]

		if self.grover_fp != "none":
			item_d["grover_fp"] = self.grover_fps[i]

		return item_d

	def __len__(self):
		return self.selfies_tokens.shape[0]

	def get_collate_fn(self):
		def _collate(data_list):
			batch_d = {}
			seqs, smileses, fingerprints, grover_fps = [], [], [], []
			mut_labels, mut_mask, carc_mask_multi, carc_labels_multi = [], [], [], []
			carc_mask, carc_labels, carc_continuous, carc_prob, carc_mask_continuous = [], [], [], [], []

			for d in data_list:
				seqs.append(d["seq"])
				smileses.append(d["smiles"])
				fingerprints.append(d['fingerprints'])

				mut_labels.append(d["mut_label"])
				mut_mask.append(d['mut_mask'])

				carc_labels_multi.append(d["carc_label_multi"])
				carc_mask_multi.append(d['carc_mask_multi'])

				carc_labels.append(d["carc_label"])
				carc_prob.append(d['carc_prob'])
				carc_mask.append(d['carc_mask'])

				carc_continuous.append(d['carc_continuous'])
				carc_mask_continuous.append(d['carc_mask_continuous'])

				if self.grover_fp != "none":
					grover_fps.append(d["grover_fp"])

			batch_d["batch_size"] = len(seqs)
			batch_d["seq"] = torch.stack(seqs, dim=0)
			batch_d["fingerprints"] = torch.stack(fingerprints, dim=0)
			batch_d["smiles"] = smileses

			batch_d["mut_label"] = torch.stack(mut_labels, dim=0)
			batch_d["mut_mask"] = torch.stack(mut_mask, dim=0)

			batch_d["carc_label"] = torch.stack(carc_labels, dim=0)
			batch_d["carc_prob"] = torch.stack(carc_prob, dim=0)
			batch_d["carc_mask"] = torch.stack(carc_mask, dim=0)

			batch_d["carc_label_multi"] = torch.stack(carc_labels_multi, dim=0)
			batch_d["carc_mask_multi"] = torch.stack(carc_mask_multi, dim=0)

			batch_d["carc_continuous"] = torch.stack(carc_continuous, dim=0)
			batch_d["carc_mask_continuous"] = torch.stack(carc_mask_continuous, dim=0)

			if self.grover_fp != "none":
				batch_d["grover_fp"] = torch.stack(grover_fps, dim=0)
			return batch_d
		return _collate

	def get_data_feats(self):
		c_feats = len(self.stoi)+1
		l_feats = self.max_seq_len
		finger_feats = len(self.fingerprints[0])

		data_feats_d = dict(c_feats=c_feats,l_feats=l_feats, finger_feats=finger_feats)
		if self.grover_fp != "none":
			data_feats_d["grover_fp_feats"] = self.grover_fps.shape[1]
		else:
			data_feats_d["grover_fp_feats"] = 0
		return data_feats_d


class ExplainMolecules(torch.utils.data.Dataset):

	def __init__(self,
		smileses=None,
		grover_fp_df=None,
		use_graph=False,
		use_seq=False,
		use_fingerprint=False,
		use_grover=False,
		drop_ionic=True,
		min_carbon_count=0,
		graph_featurizer="canonical",
		max_seq_len=None,
		stoi=None,
		torsion_fingerprints=True,
		atom_pairs_fingerprints=True,
		fp_nbits=1024):
		"""
		smiles: list of str
		grover_fps: list of ndarray
		"""

		super(ExplainMolecules,self).__init__()
		self.smileses = smileses
		self.grover_fp_df = grover_fp_df
		self.use_graph = use_graph
		self.use_seq = use_seq
		self.use_fingerprint = use_fingerprint
		self.use_grover = use_grover
		self.drop_ionic = drop_ionic
		self.graph_featurizer = graph_featurizer
		self.max_seq_len = max_seq_len
		self.stoi = stoi
		self.min_carbon_count = min_carbon_count
		self.torsion_fingerprints = torsion_fingerprints
		self.atom_pairs_fingerprints = atom_pairs_fingerprints
		self.fp_nbits = fp_nbits
		
		self.process()

	@classmethod
	def smiles_standardize(cls, smiles):
		if pd.isnull(smiles):
			return np.nan
		# Generate the molecule from smile string
		mol = Chem.MolFromSmiles(smiles)
		# If the smile string is null then continue
		if pd.isnull(mol):
			return np.nan
		else:
			return Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False,allBondsExplicit=False)

	@classmethod
	def sf_encode(cls,smiles):
		try:
			return sf.encoder(smiles)
		except:
			return np.nan 

	@classmethod
	def is_ionic(cls,smiles):
		return "." in smiles
	
	def is_too_long(self,selfies):
		return len(list(sf.split_selfies(selfies))) > self.max_seq_len

	def has_too_few_carbons(self, smile):
		mol = Chem.MolFromSmiles(smile)
		num_c = 0
		for atom in mol.GetAtoms():
			if atom.GetSymbol().upper() == 'C':
				num_c += 1
		return num_c <= self.min_carbon_count

	def convert_smile_to_fp_bit_string(self, smile):
		"""
		RDFKIT Morgan and MACCSS are default fingerprints. Torsion and atom pairs are optional
		"""
		# RDFKIT
		x = Chem.MolFromSmiles(smile)
		fp1 = Chem.RDKFingerprint(x, fpSize=self.fp_nbits)
		# MACCSS substructure
		fp2 = MACCSkeys.GenMACCSKeys(x)
		# Morgan
		fp_hashes = []
		fp3 = AllChem.GetHashedMorganFingerprint(x, 2, nBits=self.fp_nbits)
		fp3_array = np.zeros((0,), dtype=np.int8)
		DataStructs.ConvertToNumpyArray(fp3, fp3_array)
		fp_hashes.append(fp3_array)

		# Hashed atom pairs
		if self.atom_pairs_fingerprints:
			fp4 = AllChem.GetHashedAtomPairFingerprint(x, nBits=self.fp_nbits)
			fp4_array = np.zeros((0,), dtype=np.int8)
			DataStructs.ConvertToNumpyArray(fp4, fp4_array)
			fp_hashes.append(fp4_array)

		# Torsion fingerprints?
		if self.torsion_fingerprints:
			fp5 = AllChem.GetHashedTopologicalTorsionFingerprint(x, nBits=self.fp_nbits)
			fp5_array = np.zeros((0,), dtype=np.int8)
			DataStructs.ConvertToNumpyArray(fp5, fp5_array)
			fp_hashes.append(fp5_array)

		fp = fp1.ToBitString() + fp2.ToBitString()
		fp = np.array(list(fp)).astype(np.int8)
		fp = np.concatenate([fp] + fp_hashes)
		fp = torch.tensor(fp).to(torch.float32)
		return fp

	def process(self):

		# dataframe is used for filtering convenience
		smiles_df = pd.DataFrame({"smiles": self.smileses})
		# standardize
		smiles_df.loc[:,"smiles"] = smiles_df["smiles"].apply(lambda x: self.smiles_standardize(x)).dropna()
		# compound filtering
		if self.drop_ionic:
			ionic_mask = smiles_df["smiles"].apply(lambda x: self.is_ionic(x))
			smiles_df = smiles_df[~ionic_mask]

		# drop carbon defficient things
		has_too_few_carbons_mask = smiles_df["smiles"].apply(lambda x: self.has_too_few_carbons(x))
		smiles_df = smiles_df[~has_too_few_carbons_mask]
		has_too_few_carbons_mask_g = self.grover_fp_df["smiles"].apply(lambda x: self.has_too_few_carbons(x))
		self.grover_fp_df = self.grover_fp_df[~has_too_few_carbons_mask_g]

		if self.use_grover:
			grover_df_2 = self.grover_fp_df.copy()
			grover_df_2.loc[:, "smiles"] = grover_df_2["smiles"].apply(
				lambda x: self.smiles_standardize(x)
			).dropna()
			if sum(~smiles_df['smiles'].isin(grover_df_2['smiles'])) != 0:
				warnings.warn(
					f"Unable to generate grover fingerprints for all molecules dropping a portion of them "
					f"n={sum(~smiles_df['smiles'].isin(grover_df_2['smiles']))}"
				)
				smiles_df = smiles_df[smiles_df['smiles'].isin(grover_df_2['smiles'])]

		if self.use_graph:
			if self.graph_featurizer == "canonical":
				node_featurizer = CanonicalAtomFeaturizer()
				edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
			elif self.graph_featurizer == "pretrain":
				node_featurizer = PretrainAtomFeaturizer(atomic_number_types=list(range(120)))
				edge_featurizer = PretrainBondFeaturizer()
			else:
				raise ValueError
			with suppress_output(stdout=True, stderr=False):
				bigraphs = []
				smileses = smiles_df["smiles"].tolist()
				for smiles in smileses:
					bigraphs.append(
						dgllife.utils.smiles_to_bigraph(
						smiles,
						add_self_loop=True,
						node_featurizer=node_featurizer,
						edge_featurizer=edge_featurizer,
						canonical_atom_order=False,
					))
				index_to_keep = np.array([pd.notnull(bigraph) for bigraph in bigraphs])
				smiles_df = smiles_df[index_to_keep]
			self.graphs = bigraphs
		
		if self.use_seq:
			raise NotImplementedError

		if self.use_fingerprint:
			self.fingerprints = [self.convert_smile_to_fp_bit_string(smiles) for smiles in smiles_df["smiles"].tolist()]

		if self.use_grover:
			num_smiles = smiles_df.shape[0]
			assert not smiles_df["smiles"].isna().any()
			grover_fp_df = self.grover_fp_df.drop_duplicates(subset="smiles")
			grover_fp_df.loc[:, "smiles"] = grover_fp_df["smiles"].apply(lambda x: self.smiles_standardize(x)).dropna()
			grover_fp_df = smiles_df.merge(grover_fp_df, how="left", on=["smiles"])

			assert grover_fp_df['fps'].isnull().sum() == 0
			assert len(grover_fp_df) == num_smiles, (len(grover_fp_df),num_smiles)
			self.grover_fps = [torch.as_tensor(fp) for fp in grover_fp_df["fps"].tolist()]

		# no more filtering, update smileses
		self.smileses = smiles_df["smiles"].tolist()

	def __getitem__(self, i):
		item_d = {}
		item_d["smiles"] = self.smileses[i]
		if self.use_graph:
			item_d["graph"] = self.graphs[i]
		if self.use_seq:
			item_d["seq"] = self.seqs[i]
		if self.use_fingerprint:
			item_d["fingerprints"] = self.fingerprints[i]
		if self.use_grover:
			item_d["grover_fp"] = self.grover_fps[i]
		return item_d

	def __len__(self):
		return len(self.smileses)

	def get_collate_fn(self):
		keys = ["smiles"]
		if self.use_graph:
			keys.append("graph")
		if self.use_seq:
			keys.append("seq")
		if self.use_fingerprint:
			keys.append("fingerprints")
		if self.use_grover:
			keys.append("grover_fp")
		def _collate(data_list):
			batch_d = {k: [] for k in keys}
			for d in data_list:
				for k in keys:
					batch_d[k].append(d[k])
			batch_d["batch_size"] = len(batch_d["smiles"])
			for k in batch_d.keys():
				if k == "graph":
					batch_d[k] = dgl.batch(batch_d["graph"])
				elif k in ["seq","fingerprints","grover_fp"]:
					batch_d[k] = torch.stack(batch_d[k],dim=0)
			return batch_d
		return _collate

	def get_data_feats(self):

		if self.use_graph:
			if self.graph_featurizer == "canonical":
				graph = self.graphs[0]
				n_feats = graph.ndata["h"].shape[1]
				e_feats = graph.edata["e"].shape[1]
			else:
				n_feats = [120, 3]
				e_feats = [6, 3]
		else:
			n_feats = 0
			e_feats = 0
		if self.use_seq:
			c_feats = len(self.stoi)+1
			l_feats = self.max_seq_len
		else:
			c_feats = 0
			l_feats = 0
		if self.use_fingerprint:
			finger_feats = len(self.fingerprints[0])
		else:
			finger_feats = 0
		if self.use_grover:
			grover_fp_feats = len(self.grover_fps[0])
		else:
			grover_fp_feats = 0
		data_feats_d = dict(
			n_feats=n_feats,
			e_feats=e_feats,
			c_feats=c_feats,
			l_feats=l_feats,
			finger_feats=finger_feats,
			grover_fp_feats=grover_fp_feats
		)
		return data_feats_d


def split_data(dataset, train_fraction=0.7, val_fraction=0.15, test_fraction=.15):
	assert isinstance(dataset, CancerMolecules), type(dataset)
	with np_temp_seed(13371337):
		split = ScaffoldSplitter()
		train, val, test = split.train_val_test_split(
			dataset, scaffold_func='smiles', frac_train=train_fraction, frac_val=val_fraction, frac_test=test_fraction,
		)
	return train, val, test


def kfold_split_data(dataset, k=5):
	assert isinstance(dataset, CancerMolecules), type(dataset)
	with np_temp_seed(13371337):
		split = ScaffoldSplitter()
		list_of_data_tuples = split.k_fold_split(dataset, scaffold_func='smiles', k=k)
	return list_of_data_tuples


def split_subset_data_loader(dataset, train_fraction=0.5, val_fraction=0.5, test_fraction=0):
	assert isinstance(dataset, dgl.data.utils.Subset), type(dataset)
	with np_temp_seed(13371337):
		chem_mol = [Chem.MolFromSmiles(s, sanitize=True) for s in dataset.dataset.smiles[dataset.indices]]
		train, val, test = ScaffoldSplitter.train_val_test_split(
			dataset=dataset, mols=chem_mol, log_every_n=None, scaffold_func='smiles',
			frac_train=train_fraction, frac_val=val_fraction, frac_test=test_fraction,
		)
		# collapse 1 level. The above function creates a subset of a subset which can be misleading
		train = dgl.data.utils.Subset(train.dataset.dataset, np.array(train.dataset.indices)[train.indices])
		val = dgl.data.utils.Subset(val.dataset.dataset, np.array(val.dataset.indices)[val.indices])
		test = dgl.data.utils.Subset(test.dataset.dataset, np.array(test.dataset.indices)[test.indices])

	return train, val, test


def make_timestamp():
	timestamp = "_".join(re.split(':|-| ', str(datetime.now()).split('.')[0]))
	return timestamp


def test_ds(model_type):

	from arguments import get_args
	args = get_args()
	print(model_type)
	args["model_type"] = model_type
	if model_type == "grover_pt":
		args["grover_fp"] = "large"
		args["use_carc_prob"] = True
		args["carc_perecentile_to_drop"] = 99

	train_data, _ = get_datasets(args)
	item1 = train_data[0]
	item2 = train_data[-1]
	print(item1)
	print(item2)
	collate_fn = train_data.get_collate_fn()
	b_items = collate_fn([item1,item2])
	print(b_items)
	import ipdb;ipdb.set_trace()


if __name__ == '__main__':
	# data = GraphCancerMolecules(carc_datasets=('carc_ccris',))
	# data2 = GraphCancerMolecules(mut_datasets=('mut_li',))
	# data3 = GraphCancerMolecules(mut_datasets=('all_data',), carc_datasets=('all_data',))
	# test_ds("gat")
	test_ds("mlp_fingerprint")
	test_ds("cnn")
	test_ds("grover_pt")
	test_ds("gin_pt1")
