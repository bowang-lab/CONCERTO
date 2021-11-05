import wandb
import argparse
import exmol
import torch as th
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import glob
import json
import selfies as sf
import tqdm
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import seaborn as sns
sns.set()

from train import load_data, get_loss_criteria, run_an_eval_epoch, to_device
from models import get_model
from data_loader import get_explain_dataset
from plot_utils import fig_to_data
from grover_feats import convert_smiles_to_fp


def load_dicts(new_run_dp,template_d,args):

	with open(os.path.join(new_run_dp,"config.yaml"),"r") as config_file:
		config_d = yaml.load(config_file, Loader=yaml.FullLoader)
	data_d_keys = template_d["data"].keys()
	model_d_keys = template_d["model"].keys()
	run_d_keys = template_d["run"].keys()
	data_d, model_d, run_d = {}, {}, {}
	for k,v in config_d.items():
		if k in data_d_keys:
			data_d[k] = v["value"]
		elif k in model_d_keys:
			model_d[k] = v["value"]
		elif k in run_d_keys:
			run_d[k] = v["value"]
	# data_d["primary_dset"] = args.primary_dset
	# data_d["secondary_dset"] = args.secondary_dset
	run_d["do_test"] = False
	run_d["do_matching"] = False
	run_d["batch_size"] = 512
	return data_d, model_d, run_d


def get_samples(target_smiles,preset,num_samples):

	# set up stoned
	stoned_kw = {
		"num_samples": num_samples
	}
	if preset == "medium":
		stoned_kw["max_mutations"] = 2
		stoned_kw["alphabet"] = exmol.get_basic_alphabet()
	elif preset == "narrow":
		stoned_kw["max_mutations"] = 1
		stoned_kw["alphabet"] = exmol.get_basic_alphabet()
	elif preset == "wide":
		stoned_kw["max_mutations"] = 5
		stoned_kw["alphabet"] = sf.get_semantic_robust_alphabet()
	pbar = tqdm.tqdm(total=num_samples)
	samples, _ = exmol.run_stoned(target_smiles,_pbar=pbar,**stoned_kw)
	return samples

def get_exmol_exps(origin_smiles,smi_yhat,fxn_values,smiles,selfies,scores):

	pbar = tqdm.tqdm()

	# pack them into data structure with filtering out identical
	# and nan
	exps = [
		exmol.Example(
			origin_smiles,
			sf.encoder(origin_smiles),
			1.0,
			smi_yhat,
			index=0,
			is_origin=True,
		)
	] + [
		exmol.Example(sm, se, s, np.squeeze(y), index=0)
		for i, (sm, se, s, y) in enumerate(zip(smiles, selfies, scores, fxn_values))
		if s < 1.0 and np.isfinite(np.squeeze(y))
	]
	for i, e in enumerate(exps):
		e.index = i

	pbar.reset(len(exps))
	pbar.set_description('🔭Projecting...🔭')

	# compute distance matrix
	full_dmat = exmol.exmol._fp_dist_matrix(
		[e.smiles for e in exps],
		"ECFP6",
		_pbar=pbar
	)

	pbar.set_description('🥰Finishing up🥰')

	# compute PCA
	pca = PCA(n_components=2)
	proj_dmat = pca.fit_transform(full_dmat)
	for e in exps:
		e.position = proj_dmat[e.index, :]

	# do clustering everwhere (maybe do counter/same separately?)
	# clustering = AgglomerativeClustering(
	#    n_clusters=max_k, affinity='precomputed', linkage='complete').fit(full_dmat)
	# Just do it on projected so it looks prettier.
	clustering = DBSCAN(eps=0.15, min_samples=5).fit(proj_dmat)

	for i, e in enumerate(exps):
		e.cluster = clustering.labels_[i]

	pbar.set_description('🤘Done🤘')
	pbar.close()
	return exps

def plot_space(exps,delta,nmols,fig_kwargs):

	cfs = exmol.rcf_explain(exps,delta,nmols)
	exmol.plot_space(exps,cfs) #,figure_kwargs={"figsize": (24, 16)})
	fig = plt.gcf()
	# fig.savefig(f"{flags.explain_name}.png",bbox_inches='tight')
	data = fig_to_data(fig,**fig_kwargs)
	plt.savefig('./exmol_viz.pdf')
	plt.clf()
	return data


def plot_scatter(exps, true_y, delta, num_viz, nmols, offset, target_smiles, flags, carc_df, fig_kwargs):
	cfs = exmol.rcf_explain(exps, delta, nmols)
	x_cfs = [1 - e.similarity for e in cfs]
	y_cfs = [return_percentile(e.yhat, carc_df['carc_continuous'].values) for e in cfs]

	cmap = "viridis"
	sims = np.array([exp.similarity for exp in exps])
	dists = 1.-sims
	yhats = np.array([exp.yhat for exp in exps])
	pred_yhat = exps[0].yhat
	lower_yhat = pred_yhat-delta
	upper_yhat = pred_yhat+delta

	true_percentile = return_percentile(true_y, carc_df['carc_continuous'].values)
	pred_percentile = return_percentile(pred_yhat, carc_df['carc_continuous'].values)
	upper_percentile = return_percentile(upper_yhat, carc_df['carc_continuous'].values)
	lower_percentile = return_percentile(lower_yhat, carc_df['carc_continuous'].values)

	# make index selection somewhat stochastic so that we
	# don't select from the same cluster
	idx = np.argsort(dists)[1:5 * num_viz + 1]
	np.random.seed(1337)
	idx = np.random.choice(idx, num_viz)
	sns.set()
	sns.set_context('talk')

	fig = plt.figure(figsize=(12, 12))
	gs = fig.add_gridspec(3, 3)

	ax = fig.add_subplot(gs[:2, :])

	scatter_dists = np.concatenate([dists[idx], x_cfs])
	scatter_yhats = np.concatenate([yhats[idx], y_cfs])
	scatter_percentiles = np.array(
		[return_percentile(x, carc_df['carc_continuous'].values) for x in scatter_yhats])

	ax.axhline(y=true_percentile, color="blue", linestyle="solid", linewidth=3)
	ax.axhline(y=pred_percentile, color="red", linestyle="solid", linewidth=1)
	ax.axhline(y=lower_percentile, color="red", linestyle="dashed", linewidth=0.5)
	ax.axhline(y=upper_percentile, color="red", linestyle="dashed", linewidth=0.5)
	ax.scatter(
		scatter_dists,
		scatter_percentiles,
		cmap=cmap,
		edgecolors="black",
		color='gray',
	)

	# plt.title(target_smiles)
	ax.set_xlabel("Tanimoto score distance")
	ax.set_ylabel("Carcinogenecity Percentile")
	# fig.suptitle("prediction vs compound similarity")

	# x axis is similarity with target
	# y axis is model score
	mol_size = (800, 800)
	mol_fontsize = 8
	imgs = exmol.plot_utils._mol_images(cfs, mol_size, mol_fontsize)
	titles = []
	ax.set_title(target_smiles)

	sns.set_style("white")
	for j,e in enumerate(cfs):
		if not e.is_origin:
			titles.append(f"Distance = {1-e.similarity:.2f}\nPercentile={100*(y_cfs[j]):.0f}")
		else:
			titles.append(f"Base\nPercentile={100*(true_percentile):.0f}")

	if len(cfs) >= 1:
		ax.scatter(x_cfs[0], y_cfs[0], color='black', edgecolors="black", s=150)
		ax1 = fig.add_subplot(gs[2, 0])
		ax1.imshow(imgs[0])
		ax1.set_xticklabels('')
		ax1.set_yticklabels('')
		ax1.set_title(titles[0])

	if len(cfs) >= 2:
		ax.scatter(x_cfs[1], y_cfs[1], color='blue', edgecolors="black", s=150)
		ax2 = fig.add_subplot(gs[2, 1])
		ax2.imshow(imgs[1])
		ax2.set_xticklabels('')
		ax2.set_yticklabels('')
		ax2.set_title(titles[1], fontweight=500)
		for spine in ax2.spines.values():
			spine.set_edgecolor('blue')

	if len(cfs) >= 3:
		ax.scatter(x_cfs[2], y_cfs[2], color='red', edgecolors="black", s=150)
		ax3 = fig.add_subplot(gs[2, 2])
		ax3.imshow(imgs[2])
		ax3.set_xticklabels('')
		ax3.set_yticklabels('')
		ax3.set_title(titles[2], fontweight=500)
		for spine in ax3.spines.values():
			spine.set_edgecolor('red')

	if len(cfs) > 3:
		raise ValueError

	# exmol.plot_utils._image_scatter(x_cfs, y_cfs, imgs, titles, colors, ax, offset=offset)
	plt.tight_layout()
	box = ax.get_position()
	box.y0 = box.y0 + .04
	ax.set_position(box)

	data = fig_to_data(fig,**fig_kwargs)
	# ax.set_aspect('auto')
	save_dir = f'../plots/explain/explain_{flags.lower_bound_percent}_' \
			   f'{flags.upper_bound_percent}_{flags.delta}_{flags.run_id}/'
	os.makedirs(name=save_dir, exist_ok=True)
	plt.savefig(f'{save_dir}/scatter_{target_smiles}.png',
				# bbox_inches='tight'
				)
	plt.clf()
	return data


def return_percentile(value, distribution):
	index_of_closest = np.argmin(np.abs(np.sort(distribution) - value))
	return index_of_closest / len(distribution)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# parser.add_argument("--template_fp", type=str, default="config/template.yml")
	parser.add_argument("run_id", type=str)
	parser.add_argument("--explain_name", type=str, default="explain")
	parser.add_argument("--wandb_user", type=str, default="antichronology")
	parser.add_argument("--project_name", type=str, default="CONCERTO_v2-sweep")
	parser.add_argument("--device_id", type=int, default=0, choices=[-1,0,1,2,3])
	parser.add_argument("--delta", type=float,default=1.)
	parser.add_argument("--nmols", type=int, default=2)
	parser.add_argument("--preset", type=str, default="medium")
	parser.add_argument("--num_samples",type=int,default=3000)
	parser.add_argument("--grover_base_dp", type=str,
						default="/h/phil/Documents/01_projects/03_learning_molecular_representation/grover")
	parser.add_argument("--lower_bound_percent",type=int, default=65)
	parser.add_argument("--upper_bound_percent",type=int, default=85)
	parser.add_argument("--num_mols_to_explain",type=int, default=1)
	parser.add_argument("--wandb_run_dir", type=str, default="/scratch/gobi2/phil/")

	parser.add_argument("--num_viz",type=int,default=100)
	parser.add_argument("--wandb_mode",type=str,default="offline",choices=["online","offline"])
	parser.add_argument("--molecule_row_id",type=int,default=0)

	flags = parser.parse_args()

	assert (flags.lower_bound_percent >= 0) and (flags.lower_bound_percent <= 100)
	assert (flags.upper_bound_percent >= 0) and (flags.upper_bound_percent <= 100)

	if flags.device_id == -1:
		dev = th.device("cpu")
	else:
		dev = th.device(f"cuda:{flags.device_id}")

	wandb_base = os.path.join(flags.wandb_user, flags.project_name)
	wandb_config = {
		"run_id": flags.run_id
	}
	wandb.init(
		project=flags.project_name,
		name=f"{flags.explain_name}_{flags.lower_bound_percent}_"
			 f"{flags.upper_bound_percent}_{flags.delta}_{flags.run_id}",
		config=wandb_config,
		mode=flags.wandb_mode,
		dir=flags.wandb_run_dir,
	)
	# set up run directories
	old_run_dp = os.path.join(wandb_base,flags.run_id)
	new_run_dp = os.path.join(wandb.run.dir,flags.run_id)
	print(old_run_dp,new_run_dp)
	os.makedirs(new_run_dp,exist_ok=True)
	# restore files
	checkpoint_fn = "checkpoint_0.pkl" #glob.glob("checkpoint*.pkl")[0]
	config_fn = "config_0.json" #glob.glob("config*.json")[0]
	wandb.restore(checkpoint_fn,run_path=old_run_dp,root=new_run_dp,replace=True)
	wandb.restore(config_fn,run_path=old_run_dp,root=new_run_dp,replace=True)
	
	# wandb.restore(dataset_fn,run_path=old_run_dp,root=new_run_dp,replace=False)
	# set up args, data, model
	with open(os.path.join(new_run_dp,config_fn),'r') as config_file:
		args = json.load(config_file)
	args["device"] = dev
	train_loader, val_loader, test_loader, held_out_test_data_loader, data_feats = load_data(args)
	model = get_model(args,data_feats)
	
	model.to(dev)
	checkpoint = th.load(os.path.join(new_run_dp,checkpoint_fn),map_location=dev)
	model.load_state_dict(checkpoint)
	carc_loss_fn, mut_loss_fn, _ = get_loss_criteria(args)
	eval_df = run_an_eval_epoch(args,model,val_loader,mut_loss_fn,carc_loss_fn)[11]

	carc_df = eval_df[eval_df['carc_mask']]
	assert len(carc_df) > 0
	# TBD assumes continuous
	carc_logits = th.tensor(carc_df['carc_logits'].tolist()).flatten()
	carc_continuous = th.tensor(carc_df['carc_continuous'].tolist()).flatten()
	carc_loss_fn = th.nn.MSELoss(reduction="none")
	carc_loss = carc_loss_fn(carc_logits,carc_continuous)
	carc_df.loc[:,"loss"] = carc_loss.numpy()
	# get the carcinogenic molecule that was most accurately predicted
	lower_bound = np.percentile(carc_df['carc_continuous'].values, flags.lower_bound_percent)
	upper_bound = np.percentile(carc_df['carc_continuous'].values, flags.upper_bound_percent)
	print(f"Find molecules between {flags.upper_bound_percent}th and"
		  f" {flags.lower_bound_percent}th"
		  f" percentiles: {lower_bound}, {upper_bound}")

	# about 75% of carcinogenic molecules are shorter than 45
	carc_df_limit = carc_df[
		(carc_df["carc_continuous"] > lower_bound) &
		(carc_df["carc_continuous"] < upper_bound) &
		(carc_df["smiles"].str.len() < 50)
		]
	assert len(carc_df) > 0

	carc_df_limit.sort_values('loss', inplace=True, ascending=True)
	for index,row in carc_df_limit.head(flags.num_mols_to_explain).iterrows():

		target_smiles = row["smiles"]
		target_loss = row["loss"]
		target_carc = row["carc_continuous"]
		# sample around target
		smileses = get_samples(target_smiles,flags.preset,flags.num_samples)
		smiles_df = pd.DataFrame({"smiles": smileses})
		assert smiles_df["smiles"].nunique() == smiles_df.shape[0]
		# get grover fingerprints
		base_df, large_df = convert_smiles_to_fp(smiles_df,flags.grover_base_dp,"explain")

		# create dataset
		smileses = smiles_df["smiles"].tolist()
		try:
			ds = get_explain_dataset(args,smileses,base_df,large_df)
		except:
			continue
		# some compounds may have been dropped
		smileses = ds.smileses
		selfieses = [sf.encoder(smiles) for smiles in smileses]
		scores = exmol.stoned.get_fp_scores(smileses,target_smiles,fp_type="ECFP6")
		dl = th.utils.data.DataLoader(
			ds,
			num_workers=0,
			shuffle=False,
			drop_last=False,
			batch_size=128,
			collate_fn=ds.get_collate_fn()
		)

		carcs = []
		with th.no_grad():
			for b_idx, b_data in tqdm.tqdm(enumerate(dl),total=len(dl)):
				b_data = to_device(b_data,dev)
				b_pred = model(b_data)[:,1]
				carcs.append(b_pred.flatten().cpu().numpy())
		carcs = np.concatenate(carcs,axis=0)

		exps = get_exmol_exps(target_smiles,target_carc,carcs,smileses,selfieses,scores)
		plt.clf()

		fig_kwargs = {
			# "bbox_inches": "tight",
			"dpi": 500
		}
		offset=1
		scatter_data = plot_scatter(
			exps,
			float(target_carc),
			flags.delta,
			flags.num_viz,
			flags.nmols,
			offset,
			target_smiles,
			flags,
			carc_df,
			fig_kwargs,
		)

		exmol_df = pd.DataFrame({
			'dists': np.array([1 - exp.similarity for exp in exps]),
			'carc_pred': np.array([exp.yhat for exp in exps]),
			'smiles': np.array([exp.smiles for exp in exps]),
			# 'selfies': np.array([exp.selfies for exp in exps]),
		})

		log_d = {
			# f"space": wandb.Image(space_data),
			f"scatter_{target_smiles}": wandb.Image(scatter_data),
			f"table_{target_smiles}": wandb.Table(data=exmol_df, columns=exmol_df.columns)
		}

		wandb.log(log_d,commit=True)
		# except IndexError:
		# 	print('Error!')
		# 	continue
