import wandb
import argparse
import torch as th
import numpy as np
import os
import glob
import json
import selfies as sf
import tqdm
import pandas as pd
import seaborn as sns
sns.set()

from train import load_data, get_loss_criteria, run_an_eval_epoch, to_device
from models import get_model
from data_loader import get_explain_dataset, GraphCancerMolecules
from grover_feats import convert_smiles_to_fp


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--smiles_df_directory", type=str)
	parser.add_argument("--predictions_df_directory", type=str)
	parser.add_argument("--run_id", type=str)
	parser.add_argument("--wandb_user", type=str, default="antichronology")
	parser.add_argument("--project_name", type=str, default="CONCERTO_v2-sweep")
	parser.add_argument("--device_id", type=int, default=0, choices=[-1,0,1,2,3])
	parser.add_argument("--analysis_name")
	parser.add_argument("--wandb_mode",type=str,default="offline",choices=["online","offline"])
	parser.add_argument("--grover_base_dp", type=str,
						default="/h/phil/Documents/01_projects/03_learning_molecular_representation/grover")

	flags = parser.parse_args()

	smiles_df = pd.read_csv(flags.smiles_df_directory)
	smiles_df['smiles'] = GraphCancerMolecules.smiles_standardize(smiles_df['smiles'])

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
		name=flags.analysis_name,
		config=wandb_config,
		mode=flags.wandb_mode
	)
	# set up run directories
	old_run_dp = os.path.join(wandb_base, flags.run_id)
	new_run_dp = os.path.join(wandb.run.dir, flags.run_id)
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
	model = get_model(args, data_feats)
	
	model.to(dev)
	checkpoint = th.load(os.path.join(new_run_dp,checkpoint_fn),map_location=dev)
	model.load_state_dict(checkpoint)
	carc_loss_fn, mut_loss_fn, _ = get_loss_criteria(args)

	base_df, large_df = convert_smiles_to_fp(smiles_df,flags.grover_base_dp,"explain")

	# create dataset
	smileses = smiles_df["smiles"].tolist()
	ds = get_explain_dataset(args,smileses,base_df,large_df)
	# some compounds may have been dropped
	selfieses = [sf.encoder(smiles) for smiles in smileses]
	dl = th.utils.data.DataLoader(
		ds,
		num_workers=0,
		shuffle=False,
		drop_last=False,
		batch_size=128,
		collate_fn=ds.get_collate_fn()
	)

	carcs = []
	smiles = []
	# import ipdb; ipdb.set_trace()
	with th.no_grad():
		for b_idx, b_data in tqdm.tqdm(enumerate(dl),total=len(dl)):
			b_data = to_device(b_data, dev)
			b_pred = model(b_data)[:,1]
			carcs.append(b_pred.flatten().cpu().numpy())
			smiles.append(b_data['smiles'])

	smiles = np.concatenate(smiles, axis=0)
	carcs = np.concatenate(carcs, axis=0)

	pred_df = pd.DataFrame({'smiles': smiles, 'carcs': carcs})
	df = pd.merge(pred_df, smiles_df, on='smiles')
	df.to_csv(flags.predictions_df_directory, index=False)

	table = wandb.Table(data=df, columns=df.columns)
	wandb.log({"my_table": table})
