import pandas as pd
import argparse

from data_loader import CancerMolecules


def export_all_smiles(args):

	ds = CancerMolecules(purpose="all_data")
	ds.load_data()
	smiles = ds.df["smiles"].to_frame()
	smiles.to_csv(args.output_fp,header=False,index=False)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("output_fp", type=str)
	args = parser.parse_args()
	export_all_smiles(args)
