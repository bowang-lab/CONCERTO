import pandas as pd
from data_loader import get_datasets
from arguments import get_args
import subprocess
import numpy as np
import argparse
import os
import shutil


def convert_smiles_to_fp(smiles_df,base_dp,temp_dn):

    # create temporary directory
    temp_dp = os.path.join(base_dp,temp_dn)
    os.makedirs(temp_dp,exist_ok=False)

    smiles_df["smiles"].to_csv(os.path.join(temp_dp,"smiles.csv"),index=False)

    feature_command = ['python', f'{base_dp}/scripts/save_features.py',
                    f'--data_path={temp_dp}/smiles.csv',
                    f'--save_path={temp_dp}/features.npz',
                    '--features_generator=fgtasklabel', '--restart']
    print(" ".join(feature_command))
    subprocess.run(feature_command)

    base_fp_command = ['python', f'{base_dp}/main.py',
                        'fingerprint', f'--data_path={temp_dp}/smiles.csv',
                        f'--features_path={temp_dp}/features.npz',
                        f'--checkpoint_path={base_dp}/model/grover_base.pt', '--fingerprint_source=both',
                        f'--output={temp_dp}/grover_base_fp.pkl']
    print(" ".join(base_fp_command))
    subprocess.run(base_fp_command)

    large_fp_command = ['python', f'{base_dp}/main.py',
                        'fingerprint', f'--data_path={temp_dp}/smiles.csv',
                        f'--features_path={temp_dp}/features.npz',
                        f'--checkpoint_path={base_dp}/model/grover_large.pt', '--fingerprint_source=both',
                        f'--output={temp_dp}/grover_large_fp.pkl']
    print(" ".join(large_fp_command))
    subprocess.run(large_fp_command)

    # load datasets
    base_df = pd.read_pickle(f"{temp_dp}/grover_base_fp.pkl")
    large_df = pd.read_pickle(f"{temp_dp}/grover_large_fp.pkl")
    
    # delete temporary directory
    shutil.rmtree(temp_dp)

    return base_df, large_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dp",type=str,default="/home/adamo/grover")
    parser.add_argument("--temp_dn",type=str,default="concerto")
    parser.add_argument("--output_dp",type=str,default="/home/adamo/CONCERTO/data")
    args = parser.parse_args()

    dset_args = get_args()
    dset_args['use_wandb']=False
    dset_args['training_carc_datasets']=('all_data',)
    dset_args['training_mut_datasets']=('all_data',)
    train, external_test = get_datasets(dset_args)

    smiles_df = train.df[['smiles']]
    base_df, large_df = convert_smiles_to_fp(smiles_df,args.base_dp,args.temp_dn)

    base_df = pd.merge(smiles_df,base_df,on="smiles",how="outer")
    large_df = pd.merge(smiles_df,large_df,on="smiles",how="outer")
    assert len(base_df) == len(smiles_df), (len(base_df),len(smiles_df))
    assert len(large_df) == len(smiles_df), (len(large_df),len(smiles_df))

    base_df.to_pickle(os.path.join(args.output_dp,"grover_base_fp.pkl"))
    large_df.to_pickle(os.path.join(args.output_dp,"grover_large_fp.pkl"))

