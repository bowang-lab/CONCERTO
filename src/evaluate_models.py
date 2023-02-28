import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
import torch.nn as nn
import torch

from data_loader import GraphCancerMolecules, get_datasets
from data_loader import make_timestamp, split_data, kfold_split_data
from arguments import get_args
from train import run_a_train_epoch, run_an_eval_epoch, set_device_and_set_seed, load_data
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from torch.nn import ELU
from fcd_torch import FCD
import json
from scipy import stats

from rdkit import Chem
from torch.utils.data import DataLoader

from models import get_model
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

sns.set()
sns.set_context('talk')


# set up arguments for loading CarcinoPred-El training set
# load_data(args)
new_args = {}
new_args['model_type'] = 'mlp_fingerprint'
new_args['grover_fp'] = 'none'
new_args["max_seq_len"] = 500
new_args["atom_pairs_fingerprints"] = True
new_args["torsion_fingerprints"] = False
new_args['fp_nbits'] = 256


new_args['training_carc_datasets'] = ('carc_caps_pred_el',)
new_args['training_mut_datasets'] = ()
new_args['held_out_test_carc_datasets'] = ("carc_ccris", )
new_args['held_out_test_mut_datasets'] = ()
new_args['out_feats'] = 1
new_args['fraction_of_data'] = 1
new_args['min_carbon_count'] = 3
new_args['use_carc_prob'] = False
new_args['carc_percentile_to_drop'] = 1

data, held_out_test_data = get_datasets(new_args)
list_of_data_tuples = kfold_split_data(data, k=3)

#perform hyperparameter search on RF
X = np.array([x.numpy() for x in data.fingerprints])
y = np.array([x.numpy() for x in data.carc_labels])
print(X.shape, y.shape)

clf = RandomForestClassifier(max_depth=200, random_state=0, oob_score=True,n_estimators=100)

n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1600, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf_random = RandomizedSearchCV(
    estimator = clf, param_distributions = random_grid,
    n_iter = 50, cv = 3, verbose=2, random_state=0, n_jobs = -1
)# Fit the random search model
rf_random.fit(X, y)

print(rf_random.best_params_)

load_model=True
if load_model:
    clf = joblib.load("../models/in_house_rf.joblib")
else:
    clf=RandomForestClassifier(**rf_random.best_params_, oob_score=True, random_state=0)
    clf.fit(X, y)

data.df['in_house_rf_pred_oob'] = clf.oob_decision_function_[:, 1]
carc_pred_el_df = data.df

save_model=False
if save_model:
    joblib.dump(clf, "../models/in_house_rf_new.joblib")



# Load all of the data
directory = '../results/'
timestamp='2021_08_25_08_33_05_2'
with open(f'{directory}{timestamp}_config_0.json') as fin:
        config = json.load(fin)
args= config
args['training_carc_datasets']=('all_data',)
args['training_mut_datasets']=('all_data',)
args['model_type'] = 'mlp_fingerprint'
args["atom_pairs_fingerprints"] = True
args["torsion_fingerprints"] = False
args['fp_nbits'] = 256

train, _ = get_datasets(args)

X = np.array([x.numpy() for x in train.fingerprints])
y = np.array([x.numpy() for x in train.carc_labels])

train.df['in_house_rf_pred'] = clf.predict_proba(X)[:, 1]

carcino_pred = pd.read_csv('../data/carcino_pred_el_concerto_pred.csv')
df = train.df.merge(carcino_pred, on='smiles', how='left')

df = pd.merge(left=df, right=carc_pred_el_df[['smiles', 'in_house_rf_pred_oob']], on='smiles', how='left')
# assign out of bag values to predictions that it was trained on
df.loc[
    df['in_house_rf_pred_oob'].notnull(), 'in_house_rf_pred'
] = df[df['in_house_rf_pred_oob'].notnull()]['in_house_rf_pred_oob']

df.drop(columns=['in_house_rf_pred_oob'], inplace=True)
assert df['smiles'].duplicated().sum() == 0


# messy function for loading all the data
def load_data_eval(dataset='held_out_test_0'):
    cnn1_baseline = pd.read_csv(f'../results/cnn_baseline/2021_11_03_13_00_54_2/{dataset}.csv')
    cnn2_baseline = pd.read_csv(f'../results/cnn_baseline/2021_11_03_11_24_56_2/{dataset}.csv')
    cnn3_baseline = pd.read_csv(f'../results/cnn_baseline/2021_11_03_12_06_06_2/{dataset}.csv')

    cnn_baseline = pd.merge(
        cnn1_baseline, cnn2_baseline[['smiles', 'carc_pred']],
        on='smiles', suffixes=['_1', '_2'])
    cnn_baseline = pd.merge(
        cnn_baseline, cnn3_baseline[['smiles', 'carc_pred']],
        on='smiles')
    cnn_baseline = cnn_baseline.rename(columns={'carc_pred': 'carc_pred_3'})
    cnn_baseline['carc_pred_cnn_baseline'] = (
                                                     cnn_baseline['carc_pred_1'] +
                                                     cnn_baseline['carc_pred_2'] +
                                                     cnn_baseline['carc_pred_3']) / 3

    fp1_baseline = pd.read_csv(f'../results/fp_baseline/2021_11_04_10_50_53_2/{dataset}.csv')
    fp2_baseline = pd.read_csv(f'../results/fp_baseline/2021_11_04_11_58_06_2/{dataset}.csv')
    fp3_baseline = pd.read_csv(f'../results/fp_baseline/2021_11_04_12_08_11_2/{dataset}.csv')

    fp_baseline = pd.merge(
        fp1_baseline, fp2_baseline[['smiles', 'carc_pred']],
        on='smiles', suffixes=['_1', '_2'])
    fp_baseline = pd.merge(
        fp_baseline, fp3_baseline[['smiles', 'carc_pred']],
        on='smiles')
    fp_baseline = fp_baseline.rename(columns={'carc_pred': 'carc_pred_3'})
    fp_baseline['carc_pred_fp_baseline'] = (
                                                   fp_baseline['carc_pred_1'] +
                                                   fp_baseline['carc_pred_2'] +
                                                   fp_baseline['carc_pred_3']) / 3

    fp_baseline = pd.read_csv(f'../results/fp_baseline/2021_11_04_15_07_41_2/{dataset}.csv')
    fp_baseline['carc_pred_fp_baseline'] = fp_baseline['carc_pred']

    grover_baseline = pd.read_csv(f'../results/grover_baseline/2021_11_04_17_10_14_2/{dataset}.csv')
    grover_baseline['carc_pred_grover'] = grover_baseline['carc_pred']

    cnn1 = pd.read_csv(f'../results/cnn/2021_08_26_21_20_44_2/{dataset}.csv')
    cnn2 = pd.read_csv(f'../results/cnn/2021_08_25_21_49_37_2/{dataset}.csv')
    cnn3 = pd.read_csv(f'../results/cnn/2021_08_27_07_47_27_2/{dataset}.csv')
    cnn = pd.merge(cnn1, cnn2[['smiles', 'carc_pred']], on='smiles', suffixes=['_1', '_2'])
    cnn = pd.merge(cnn, cnn3[['smiles', 'carc_pred']], on='smiles')
    cnn = cnn.rename(columns={'carc_pred': 'carc_pred_3'})
    cnn['carc_pred_cnn'] = (cnn['carc_pred_1'] + cnn['carc_pred_2'] + cnn['carc_pred_3']) / 3

    fp1 = pd.read_csv(f'../results/mlp_fp/2021_09_08_05_32_35_2/{dataset}.csv')
    fp2 = pd.read_csv(f'../results/mlp_fp/2021_09_08_04_06_24_2/{dataset}.csv')
    fp3 = pd.read_csv(f'../results/mlp_fp/2021_09_08_01_07_51_2/{dataset}.csv')
    fp = pd.merge(fp1, fp2[['smiles', 'carc_pred']], on='smiles', suffixes=['_1', '_2'])
    fp = pd.merge(fp, fp3[['smiles', 'carc_pred']], on='smiles')
    fp = fp.rename(columns={'carc_pred': 'carc_pred_3'})
    fp['carc_pred_fp'] = (fp['carc_pred_1'] + fp['carc_pred_2'] + fp['carc_pred_3']) / 3

    gin1 = pd.read_csv(f'../results/gin/2021_08_25_08_33_05_2/{dataset}.csv')
    gin2 = pd.read_csv(f'../results/gin/2021_08_24_16_20_49_2/{dataset}.csv')
    gin3 = pd.read_csv(f'../results/gin/2021_08_25_09_28_13_2/{dataset}.csv')
    gin = pd.merge(gin1, gin2[['smiles', 'carc_pred']], on='smiles', suffixes=['_gin_1', '_gin_2'])
    gin = pd.merge(gin, gin3[['smiles', 'carc_pred']], on='smiles')
    gin = gin.rename(columns={'carc_pred': 'carc_pred_gin_3'})
    gin['carc_pred_gin'] = (gin['carc_pred_gin_1'] +
                            gin['carc_pred_gin_2'] + gin['carc_pred_gin_3']) / 3

    gat1 = pd.read_csv(f'../results/gat/2021_10_15_14_49_27_2/{dataset}.csv')
    gat2 = pd.read_csv(f'../results/gat/2021_10_13_13_11_27_2/{dataset}.csv')
    gat3 = pd.read_csv(f'../results/gat/2021_10_15_13_51_19_2/{dataset}.csv')
    gat = pd.merge(gat1, gat2[['smiles', 'carc_pred']], on='smiles', suffixes=['_1', '_2'])
    gat = pd.merge(gat, gat3[['smiles', 'carc_pred']], on='smiles')
    gat = gat.rename(columns={'carc_pred': 'carc_pred_3'})
    gat['carc_pred_gat'] = (gat['carc_pred_1'] + gat['carc_pred_2'] + cnn['carc_pred_3']) / 3

    return gin, cnn, fp, gat, cnn_baseline, grover_baseline, fp_baseline

# function for merging all the data
def combine_data(gin, cnn, fp, gat, cnn_baseline, grover, fp_baseline, include_grover=False, df=df):
    held_out_eval = pd.merge(
        cnn[['smiles', 'carc_pred_cnn',
             'carc_continuous', 'carc_label', 'mut_label']],
        df[['smiles', 'rf_pred', 'CDK', 'Pubchem', 'in_house_rf_pred']],
        on='smiles'
    )
    held_out_eval = held_out_eval.merge(
        gin[['smiles', 'carc_pred_gin']], on='smiles'
    )
    held_out_eval = held_out_eval.merge(
        fp[['smiles', 'carc_pred_fp']], on='smiles'
    )
    held_out_eval = held_out_eval.merge(
        gat[['smiles', 'carc_pred_gat']], on='smiles'
    )
    held_out_eval = held_out_eval.merge(
        cnn_baseline[['smiles', 'carc_pred_cnn_baseline']], on='smiles'
    )
    if include_grover:
        held_out_eval = held_out_eval.merge(
            grover[['smiles', 'carc_pred_grover']], on='smiles'
        )
    held_out_eval = held_out_eval.merge(
        fp_baseline[['smiles', 'carc_pred_fp_baseline']], on='smiles'
    )

    return held_out_eval

held_out_eval = combine_data(*load_data_eval('held_out_test_0'), include_grover=True, df=df)
viz_predictors = [
    'carc_pred_fp',
#     'carc_pred_cnn_baseline',
#     'carc_pred_gin',
#     'carc_pred_gat',
#     'CDK',
    'carc_pred_fp_baseline',
    'carc_pred_grover',
    'rf_pred',
#     'Pubchem',
    'in_house_rf_pred',

]

predictors = [
    'carc_pred_cnn',
    'carc_pred_fp',
    'carc_pred_cnn_baseline',
    'carc_pred_gin',
    'carc_pred_gat',
    'rf_pred',
    'carc_pred_fp_baseline',
    'carc_pred_grover',
    'Pubchem',
    'in_house_rf_pred',
]
held_out_carc = held_out_eval[
    (held_out_eval['carc_label'].notnull())
    & (held_out_eval['CDK'].notnull())
]

held_out_mut = held_out_eval[
    (held_out_eval['mut_label'].notnull())
    & (held_out_eval['CDK'].notnull())
]


print(f"Length of carcinogenic held out test set {len(held_out_carc)}")
print(f"Length of mutagenic held out test set {len(held_out_mut)}")

def bootstrap(func, array1, array2, itrs = 100):
    outputs = []
    for _ in range(itrs):
        new_index1 = np.random.choice(np.arange(len(array1)), len(array1))
        new_index1 = np.sort(new_index1)
        outputs.append(
            func(array1[new_index1], array2[new_index1])
        )
    return outputs

def computer_pr_roc_auc(prediction, labels, name):
    precision, recall, thresholds = precision_recall_curve(labels, prediction)
    auc_value = auc(recall, precision)
    assert len(recall) == len(precision)
    outputs_roc = bootstrap(auc, recall, precision)
    auc_value_mean = np.mean(outputs_roc)
    auc_value_std = np.std(outputs_roc)
    print(f'{name} PR AUC: ${auc_value_mean:.2f} \pm {auc_value_std:.2f}$')
    fpr, tpr, thresholds = roc_curve(labels, prediction)
    auc_value = auc(fpr, tpr)
    assert len(fpr) == len(tpr)
    outputs_pr = bootstrap(auc, fpr, tpr)
    auc_value_mean = np.mean(outputs_pr)
    auc_value_std = np.std(outputs_pr)
    print(f'{name} ROC AUC: ${auc_value_mean:.2f} \pm {auc_value_std:.2f}$')
    return outputs_roc, outputs_pr


def add_pr_plot(prediction, labels, name, linestyle='solid', add_auc=False):
    precision, recall, thresholds = precision_recall_curve(labels, prediction)
    auc_value = auc(recall, precision)
    assert len(recall) == len(precision)
    outputs = bootstrap(auc, recall, precision)
    auc_value_mean = np.mean(outputs)
    auc_value_std = np.std(outputs)

    if add_auc:
        label = f'{name} AUC: {auc_value_mean:.2f} ± {auc_value_std:.2f}'
    else:
        print(f'{name} AUC: ${auc_value_mean:.2f} \pm {auc_value_std:.2f}$')
        label = f'{name}'

    plt.step(
        recall, precision,
        linestyle=linestyle,
        where='post', label=label
    )


def add_roc_plot(prediction, labels, name, linestyle='solid', add_auc=False):
    fpr, tpr, thresholds = roc_curve(labels, prediction)
    auc_value = auc(fpr, tpr)
    assert len(fpr) == len(tpr)
    outputs = bootstrap(auc, fpr, tpr)
    auc_value_mean = np.mean(outputs)
    auc_value_std = np.std(outputs)

    if add_auc:
        label = f'{name} AUC: {auc_value_mean:.2f} ± {auc_value_std:.2f}'
    else:
        print(f'{name} AUC: ${auc_value_mean:.2f} \pm {auc_value_std:.2f}$')
        label = f'{name}'

    plt.step(
        fpr, tpr,
        linestyle=linestyle,
        where='post',
        label=label
    )


predictor_name_dict = {
    'carc_pred_cnn': 'CNN backbone',
    'carc_pred_gin': 'CONCERTO GIN',
    'carc_pred_cnn_baseline': 'CNN Baseline',
    'carc_pred_gat': 'GAT backbone',
    'carc_pred_fp': 'CONCERTO',
    'rf_pred': 'Average RF CarcinoPred-EL',
    'Pubchem': 'Pubchem RF CarcinoPred-EL',
    'carc_pred_grover': 'GNN baseline',
    'carc_pred_fp_baseline': 'Fingerprint Baseline',
    'in_house_rf_pred': 'RF baseline'

}
sns.set()
sns.set_context('talk', font_scale=1.3)
figure_size = 9
df_name = held_out_carc
label_name = 'carc_label'

plt.figure(figsize=(figure_size, figure_size))

for predictor in viz_predictors:
    if predictor in ["Pubchem", "rf_pred", "CDK"]:
        linestyle = "dashed"
    elif predictor in ["carc_pred_cnn_baseline", "carc_pred_fp_baseline", 'carc_pred_grover', 'in_house_rf_pred']:
        linestyle = "dotted"
    else:
        linestyle = 'solid'
    add_pr_plot(
        df_name[predictor].values, df_name[label_name].values,
        predictor_name_dict[predictor], linestyle=linestyle
    )
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.01])
plt.xlim([0.0, 1.01])
# plt.title(f'Precision Recall Plot Held Out Test n={len(df_name)}')
plt.title(f'Precision Recall CCRIS')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('../plots/PR_concerto_v2_CCRIS.pdf')

plt.figure(figsize=(figure_size, figure_size))
for predictor in viz_predictors:
    if predictor in ["Pubchem", "rf_pred", "CDK"]:
        linestyle = "dashed"
    elif predictor in ["carc_pred_cnn_baseline", 'carc_pred_fp_baseline', 'carc_pred_grover']:
        linestyle = "dotted"
    else:
        linestyle = 'solid'

    add_roc_plot(
        df_name[predictor].values, df_name[label_name].values,
        predictor_name_dict[predictor], linestyle=linestyle
    )
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.ylim([0.0, 1.01])
plt.xlim([0.0, 1.01])
# plt.title(f"ROC Plot Held Out Test n={len(df_name)}")
plt.title(f"ROC CCRIS")
plt.legend()
plt.tight_layout()
plt.savefig('../plots/ROC_concerto_v2_CCRIS.pdf')


predictor_dict = {}
for predictor in predictors:
    outputs_roc, outputs_pr = computer_pr_roc_auc(
        df_name[predictor].values,
        df_name[label_name].values,
        predictor_name_dict[predictor],
    )
    predictor_dict[predictor + "_roc"] = outputs_roc
    predictor_dict[predictor + '_pr'] = outputs_pr

test_0 = combine_data(*load_data_eval('test_0'),include_grover=False)
test_2 = combine_data(*load_data_eval('test_1'),include_grover=False)
test_1 = combine_data(*load_data_eval('test_2'),include_grover=False)

test = pd.concat([test_0, test_1, test_2])
test = test[~test['smiles'].duplicated()]

test_carc = test[
    (test['carc_continuous'].notnull())
#     (test['rf_pred'].notnull())
]
test_carc.reset_index(inplace=True)
print(len(test_carc))

g1 = load_data_eval('test_0')[-2]
g2 = load_data_eval('test_1')[-2]
g3 = load_data_eval('test_2')[-2]
grover_test = pd.concat([g1, g2, g3])
grover_test = grover_test[~grover_test['smiles'].duplicated()]
grover_test = grover_test[
    (grover_test['carc_continuous'].notnull())
]
grover_test.reset_index(inplace=True)
print(len(grover_test))

def make_scatterplot(
    x, y, df=test_carc, title="", xlabel="", ylabel="",
    save_dir="", save_name="", save_file=False, add_correlation=False
):
    plt.clf()
    r, p = stats.pearsonr(df[x], df[y])
    print(f'r={r:.3f}, p={p:.3f}')
    j = sns.jointplot(
        x=x, y=y, data = df, kind='scatter', height=8, xlim=[-2,3.5], alpha=0.3
    )
    if add_correlation:
        j.ax_joint.legend([f'r={r:.3f}, p={p:.3f}'],prop={'size': 20})

    j.ax_joint.set_xlabel(xlabel)
    j.ax_joint.set_ylabel(ylabel)

    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93) # Reduce plot to make room
    if save_file:
        plt.savefig(f'{save_dir}/{save_name}')


make_scatterplot(
    x = 'carc_continuous',
    y = 'carc_pred_fp',
    df= test_carc,
    title="CONCERTO",
    xlabel='Measured Carcinogenicity',
    ylabel='Predicted Carcinogenicity',
    save_file=True,
    save_dir='./../plots/',
    save_name='fingerprint_test_concerto_v2.pdf'
)
make_scatterplot(
    x = 'carc_continuous',
    y = 'carc_pred_cnn',
    df= test_carc,
    title="CNN chemical embedder",
    xlabel='Measured Carcinogenicity',
    ylabel='Predicted Carcinogenicity',
    save_file=True,
    save_dir='./../plots/',
    save_name='CNN_test_concerto_v2.pdf'
)
make_scatterplot(
    x = 'carc_continuous',
    y = 'carc_pred_gin',
    df= test_carc,
    title="GIN chemical embedder",
    xlabel='Measured Carcinogenicity',
    ylabel='Predicted Carcinogenicity',
    save_file=True,
    save_dir='./../plots/',
    save_name='gin_test_concerto_v2.pdf'
)
make_scatterplot(
    x = 'carc_continuous',
    y = 'carc_pred_gat',
    df= test_carc,
    title="GAT chemical embedder",
    xlabel='Measured Carcinogenicity',
    ylabel='Predicted Carcinogenicity',
    save_file=True,
    save_dir='./../plots/',
    save_name='gat_test_concerto_v2.pdf'
)
make_scatterplot(
    x = 'carc_continuous',
    y = 'carc_pred_cnn_baseline',
    df= test_carc,
    title="CNN baseline",
    xlabel='Measured Carcinogenicity',
    ylabel='Predicted Carcinogenicity',
    save_file=True,
    save_dir='./../plots/',
    save_name='cnn_baseline_test_concerto_v2.pdf'
)
make_scatterplot(
    x = 'carc_continuous',
    y = 'carc_pred_fp_baseline',
    df= test_carc,
    title="Fingeprint MLP",
    xlabel='Measured Carcinogenicity',
    ylabel='Predicted Carcinogenicity',
    save_file=True,
    save_dir='./../plots/',
    save_name='fingerprint_baseline_test_concerto_v2.pdf'
)

make_scatterplot(
    x = 'carc_continuous',
    y = 'carc_pred_grover',
    df= grover_test,
    title="GROVER MLP",
    xlabel='Measured Carcinogenicity',
    ylabel='Predicted Carcinogenicity',
    save_file=True,
    save_dir='./../plots/',
    save_name='grover_baseline_test_concerto_v2.pdf'
)

make_scatterplot(
    x = 'carc_continuous',
    y = 'in_house_rf_pred',
    df= test_carc,
    title="In House Fingerprint RF",
    xlabel='Measured Carcinogenicity',
    ylabel='Predicted Carcinogenicity',
    save_file=True,
    save_dir='./../plots/',
    save_name='in_house_rf_baseline_test_concerto_v2.pdf'
)

x_names = ['carc_pred_cnn', 'carc_pred_gin', 'carc_pred_fp',
           'carc_pred_gat', 'carc_pred_cnn_baseline', 'carc_pred_fp_baseline',
           'rf_pred', 'CDK', 'in_house_rf_pred']

for x in x_names:
    print('\n')
    print(x)
    y = 'carc_continuous'
    vals = bootstrap(
        lambda x, y: np.mean((x - y)**2),
        test_carc[x].values,
        test_carc[y].values,
        itrs = 100
    )
    r,p = stats.pearsonr(
        test_carc[x].values,
        test_carc[y].values,
    )
    print(f"""${np.mean((test_carc[x].values - test_carc[y].values)**2):.2f}$""")
    print(np.mean(vals), np.std(vals))
    print(
        f'${np.mean(vals):.2f} '
        f'\pm {np.std(vals):.2f}$'
    )

vals = bootstrap(
    lambda x, y: np.mean((x - y)**2),
    grover_test['carc_pred_grover'].values,
    grover_test['carc_continuous'].values,
    itrs = 100
)
r,p = stats.pearsonr(
    grover_test['carc_pred_grover'].values,
    grover_test['carc_continuous'].values,
)
print('\n')
print('carc_pred_grover')
print(np.mean(vals), np.std(vals))
print(
    f'${np.mean(vals):.2f} '
    f'\pm {np.std(vals):.2f}$'
)

x_names = ['carc_pred_cnn', 'carc_pred_gin', 'carc_pred_fp',
           'carc_pred_gat', 'carc_pred_cnn_baseline', 'carc_pred_fp_baseline',
           'rf_pred', 'CDK', 'in_house_rf_pred']

for x in x_names:
    print('\n')
    print(x)
    y = 'carc_continuous'
    vals = bootstrap(
        stats.pearsonr,
        test_carc[x].values,
        test_carc[y].values,
        itrs = 1000
    )
    r,p = stats.pearsonr(
        test_carc[x].values,
        test_carc[y].values,
    )

    print(f'r={r:.3f}, p={p:.3f}')
    print(
        f'mean r={np.mean([x[0] for x in vals]):.3f}',
        f'std r={np.std([x[0] for x in vals]):.3f}'
    )
    print(f'${np.mean([x[0] for x in vals]):.2f} \pm {np.std([x[0] for x in vals]):.2f}$'
    )

vals = bootstrap(
    stats.pearsonr,
    grover_test['carc_pred_grover'].values,
    grover_test['carc_continuous'].values,
    itrs = 100
)
r,p = stats.pearsonr(
    grover_test['carc_pred_grover'].values,
    grover_test['carc_continuous'].values,
)
print('\n')
print('carc_pred_grover')
print(f'r={r:.3f}, p={p:.3f}')
print(np.mean(vals), np.std(vals))
print(
    f'${np.mean(vals):.2f} '
    f'\pm {np.std(vals):.2f}$'
)
