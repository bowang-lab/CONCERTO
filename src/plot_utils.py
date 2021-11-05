import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem.Draw import rdMolDraw2D
import torch as th
import wandb
import io
from PIL import Image

from utils import np_temp_seed


def viz_mols(args,epoch,model,losses,ys,dataset):

    num_viz = args["num_viz"]
    if num_viz > 0:
        assert args["use_wandb"]
        topk, topk_idx = th.topk(-losses,num_viz,largest=True)
        bottomk, bottomk_idx = th.topk(-losses,num_viz,largest=False)
        topk_idx = topk_idx.numpy()
        bottomk_idx = bottomk_idx.numpy()
        with np_temp_seed(1492):
            rand_idx = np.random.choice(np.arange(len(dataset)),size=num_viz,replace=False)
        # import pdb; pdb.set_trace()
        ys = ys.numpy()
        pos_idx = np.flatnonzero(ys)
        neg_idx = np.flatnonzero(-2*ys+1)
        assert len(pos_idx) >= num_viz, (len(pos_idx),num_viz)
        assert len(neg_idx) >= num_viz, (len(neg_idx),num_viz)
        with np_temp_seed(666666):
            pos_rand_idx = np.random.choice(pos_idx,size=num_viz,replace=False)
            neg_rand_idx = np.random.choice(neg_idx,size=num_viz,replace=False)
        # assemble dictionary
        data_dict = {}
        data_dict["topk"] = [dataset[int(i)] for i in topk_idx]
        data_dict["bottomk"] = [dataset[int(i)] for i in bottomk_idx]
        data_dict["rand"] = [dataset[int(i)] for i in rand_idx]
        data_dict["pos_rand"] = [dataset[int(i)] for i in pos_rand_idx]
        data_dict["neg_rand"] = [dataset[int(i)] for i in neg_rand_idx]
        log_dict = {"epoch": epoch}
        if args["model_type"] in ["gat"]:
            for k,v in data_dict.items():
                for d_idx, d in enumerate(v):
                    graph, smiles = d[0], d[2]
                    graph = graph.to(args["device"])
                    attn_dict = model.get_attention(graph,layer_idx=args["viz_layer_idx"])
                    im = viz_attention(smiles,attn_dict)
                    log_dict[f"{k}_{d_idx}"] = wandb.Image(im)
        else:
            raise NotImplementedError
        wandb.log(log_dict)

def attn_to_rgb(attn):
    cmap = matplotlib.cm.get_cmap("Reds")
    return cmap(attn)[:3]

def viz_attention(smiles,attn_dict):
    """ note that we assume that the node order in attn_dict is the same as the order in smiles"""

    # import pdb; pdb.set_trace()
    mol = Chem.MolFromSmiles(smiles)
    d = rdMolDraw2D.MolDraw2DCairo(500, 500)
    atom_idxs, bond_idxs = [], []
    atom_colors, bond_colors = {}, {}
    # import pdb; pdb.set_trace()
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        assert atom_idx == atom.GetIdx(), (atom_idx,atom.GetIdx())
        atom_idxs.append(atom_idx)
        attn = th.mean(attn_dict[(atom_idx,atom_idx)]).item()
        atom_colors[atom_idx] = attn_to_rgb(attn)
    for bond_idx, bond in enumerate(mol.GetBonds()):
        src = bond.GetBeginAtomIdx()
        dest = bond.GetEndAtomIdx()
        if src < dest:
            attn = th.mean(attn_dict[(src,dest)]).item()
        elif src > dest:
            attn = th.mean(attn_dict[(dest,src)]).item()
        else:
            raise ValueError(f"bond {bond_idx}, src ({src}) == dest ({dest})")
        bond_idxs.append(bond_idx)
        bond_colors[bond_idx] = attn_to_rgb(attn)
    d.DrawMolecules(
        [mol],
        highlightAtoms=[atom_idxs],
        highlightAtomColors=[atom_colors],
        highlightBonds=[bond_idxs],
        highlightBondColors=[bond_colors],
        legends=[smiles]
    )
    d.FinishDrawing()
    png_buf = d.GetDrawingText()
    im = Image.open(io.BytesIO(png_buf))
    return im

def fig_to_data(fig,**kwargs):

	buf = io.BytesIO()
	fig.savefig(buf,**kwargs)
	buf.seek(0)
	image = Image.open(buf)
	return image
