import argparse
import torch.nn as nn
import numpy as np
from data_loader import make_timestamp

def str2bool(v):
	return v.lower() in ('yes', 'true', 't', 'y', '1')

def get_args():
    timestamp = make_timestamp()
    parser = argparse.ArgumentParser(description='Specify parameters for the model')
    # basic model stuff
    parser.add_argument("--model_type", type=str, default="gat",
                        choices=["gat_p","gat","gin","gin_pt1","cnn", 'mlp_fingerprint',
                                 "grover_pt","wln", "gin_pt2", "gin_pt3", "gin_pt4"])
    parser.add_argument("--out_feats", type=int, default=2)
    # dataset stuff
    parser.add_argument("--max_seq_len", type=int, default=500)
    parser.add_argument("--fraction_of_data", type=float, default=1)
    parser.add_argument("--min_carbon_count", type=float, default=0)
    parser.add_argument("--carc_percentile_to_drop", type=float, default=0)
    parser.add_argument('--training_carc_datasets', type=tuple, default=('carc_cpdb',))
    parser.add_argument('--training_mut_datasets', type=tuple, default=('mut_hansen',))

    parser.add_argument("--held_out_test_carc_datasets", type=tuple, default=("carc_ccris", ))
    parser.add_argument("--held_out_test_mut_datasets", type=tuple, default=("mut_li", 'mut_ccris',))
    parser.add_argument("--grover_fp", type=str, default="none", choices=["none","base","large"])
    # GNN stuff
    parser.add_argument('--gnn_num_layers', type=int, default=2)
    parser.add_argument('--gnn_hidden_feats', type=int, default=32,
                        help="hidden dimension of GNN (meaning varies based on GNN type)")
    parser.add_argument('--gat_num_heads', type=int, default=4,
                        help="num_heads gives the number of attention heads in GAT layer")
    parser.add_argument('--gnn_dropout', type=float, default=0.1)
    parser.add_argument("--fix_pt_weights", type=str2bool, default="true")
    parser.add_argument("--gnn_pool_type", type=str, default="avg", choices=["avg","max","attn"])

    # FF stuff
    parser.add_argument('--separate_heads', type=str2bool, default='false',
                        help='separate mlp heads for carcinogenic and mutagenic outputs')
    parser.add_argument("--ff_hidden_feats", type=int, default=128,
                        help="Size for hidden representations in the output MLP predictor")
    parser.add_argument("--ff_dropout", type=float, default=0.2,
                        help="The probability for dropout in the output MLP predictor")
    parser.add_argument("--ff_num_layers", type=int, default=3)

    # CNN stuff
    parser.add_argument("--conv_kernel_size", type=int, default=5)
    parser.add_argument("--conv_stride", type=int, default=1)
    parser.add_argument("--conv_pool_type", type=str, default="max")
    parser.add_argument("--conv_pool_size", type=int, default=4)
    parser.add_argument("--conv_num_kernels", type=int, default=32)
    parser.add_argument("--conv_num_layers", type=int, default=3)

    # fingerprint stuff
    parser.add_argument("--torsion_fingerprints", type=str2bool, default='true')
    parser.add_argument("--atom_pairs_fingerprints", type=str2bool, default='true')
    parser.add_argument("--fp_nbits", type=int, default=1024)

    # training stuff
    parser.add_argument("--mut_pre_training", type=str2bool, default="false",
                        help='two training loops. One only mutagenic the next only carcinogenic')
    parser.add_argument('--num_mut_pre_training_loop', type=int, default=1,
                        help='number of times to repeat the training mutagenic then carcinogenic')
    parser.add_argument("--cross_validation", type=str2bool, default="false")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument("--network_weight_decay", type=float, default=5e-2, help="Network weights decay")
    parser.add_argument("--lr_decay_factor", type=float, default=0.5, help="lr weight decay on plateau")
    parser.add_argument("--gradient_clip_norm", type=float, default=5,
                        help="Max norm of the gradient at which it will be clipped")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument('--early_stopping_metric', type=tuple, default=('rmse', 'carc'),
                        choices=[
                            ("roc_auc_score", 'carc'),
                            ("roc_auc_score", 'mut'),
                            ("pearson_r2", 'carc'),
                            ("validation_loss", 'carc'),
                            ('rmse', 'carc')
                        ],
                        help='metric for stopping')
    parser.add_argument('--run', type=str, default=timestamp)
    parser.add_argument("--num_viz", type=int, default=0)
    parser.add_argument("--viz_layer_idx", type=int, default=-1)
    # loss stuff
    parser.add_argument('--use_carc_prob', type=str2bool, default='false',
                        help='convert carc labels into probabilities based off td50 valueus')
    parser.add_argument("--use_carc_loss", type=str2bool, default="true")
    parser.add_argument("--use_mut_loss", type=str2bool, default="true")
    parser.add_argument('--mut_loss_ratio', type=float, default=0.5, help='How much to scale mutagenicity loss by')
    parser.add_argument('--train_carc_loss_fnc', type=str, default='MSE', choices=['MSE', 'CE', "BCE"],
                        help='whether to use discrete loss for cpdb over multiple classes')
    # wandb stuff
    parser.add_argument('--group_name', type=str, default='')
    parser.add_argument('--project_note', type=str, default="",
                        help='Add to project name to create unique wandb project')
    parser.add_argument("--use_wandb", type=str2bool, default="true")
    parser.add_argument("--wandb_run_dir", type=str, default="")

    args = parser.parse_args().__dict__

    return args
