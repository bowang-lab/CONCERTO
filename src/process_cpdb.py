import pandas as pd
import numpy as np
from scipy.stats import hmean
import cirpy
import datetime

from matplotlib import pyplot as plt
import seaborn as sns

from data_loader import GraphCancerMolecules

sns.set()
sns.set_context('talk')


def read_in_cpdb():
    cpdb_lit = pd.read_csv('../data/cpdb.lit.tab.txt', sep='\t')
    cpdb_nci = pd.read_csv('../data/cpdb.ncintp.tab.txt', sep='\t')
    cpdb_df = pd.concat([cpdb_lit, cpdb_nci])

    print(f"number of unique chemcodes in rats cpdb {cpdb_df[cpdb_df['species']=='r']['chemcode'].nunique()}")
    print(f"number of unique chemcodes in cpdb {cpdb_df['chemcode'].nunique()}")
    cpdb_name = pd.read_csv('../data/cpdb_name.tsv', sep='\t')
    assert not cpdb_name['chemcode'].duplicated().sum()
    cpdb_df_len = len(cpdb_df)
    cpdb_df = cpdb_df.merge(cpdb_name, on='chemcode')
    assert cpdb_df_len == len(cpdb_df)
    cpdb_df['td50_log'] = np.log10(cpdb_df['td50'])
    return cpdb_df


def genereate_summary_cpdb_plots(cpdb_df):
    plt.figure(figsize=(20,5))
    vc = cpdb_df['chemcode'].value_counts()
    print(f"Number of experiments per compound stats: \n{pd.Series(vc.values).describe()}")
    print(f"number of compounds with more than 50 studies: {len(vc[vc>50])}")
    sns.distplot(vc[vc<=50],kde=False, bins=50)
    plt.xticks(np.arange(1,51, 1))
    plt.title('Number of experiments per compound')
    plt.xlim(0, 50)
    plt.savefig('../plots/cpdb_num_experiments_per_compound.pdf')

    plt.figure(figsize=(10,5))
    plt.title('Distribution of TD50 values per animal')
    ax = sns.violinplot(data=cpdb_df[cpdb_df['species'].isin(['m', 'r', 'h'])],x='species', y='td50_log')
    sns.stripplot(data=cpdb_df[cpdb_df['species'].isin(['m', 'r', 'h'])],x='species', y='td50_log',linewidth=1, color='white',alpha=.5)
    ax.set_xticklabels(['rat', 'mouse', 'hamster'])
    plt.savefig('../plots/TD50_distribution_per_species.pdf')


def agregate_cpdb(cpdb_df):
    # These cas IDS map to the same smiles strings. So first remap these cas IDS so don't have to do
    # SMILES matching on all the rows
    cas_map = {'14026-03-0': '36702-44-0', '104-46-1': '4180-23-8', '13073-35-3': '67-21-0', '1150-37-4': '1150-42-1',
     '121-14-2': '25321-14-6', '72-20-8': '60-57-1', '319-84-6': '58-89-9', '319-85-7': '58-89-9',
     '608-73-1': '58-89-9', '764-41-0': '110-57-6', '107-06-2':
         '7572-29-4', '73-22-3': '54-12-6', '100-63-0': '59-88-1', '488-41-5': '10318-26-0', '9006-42-2': '12122-67-7'}

    cpdb_df['cas'] = [cas_map[x] if x in cas_map.keys() else x for x in cpdb_df['cas'].values]
    cpdb_g = cpdb_df[~cpdb_df['cas'].duplicated()][['name', 'cas', 'chemcode', 'species']]
    cpdb_g_len = len(cpdb_g)

    temp = cpdb_df[['cas', 'td50_log', 'td50', ]].groupby('cas').min().reset_index()
    columns_t = temp.columns.tolist()
    columns_t.remove('cas')
    columns_t = {x: x + '_min' for x in columns_t}
    temp.rename(columns=columns_t, inplace=True)
    cpdb_g = pd.merge(cpdb_g, temp, on='cas')

    temp = cpdb_df[['cas', 'td50_log', 'td50', ]].groupby('cas').mean().reset_index()
    columns_t = temp.columns.tolist()
    columns_t.remove('cas')
    columns_t = {x: x + '_mean' for x in columns_t}
    temp.rename(columns=columns_t, inplace=True)
    cpdb_g = pd.merge(cpdb_g, temp, on='cas')

    temp = cpdb_df[['cas', 'td50_log', 'td50', ]].groupby('cas').median().reset_index()
    columns_t = temp.columns.tolist()
    columns_t.remove('cas')
    columns_t = {x: x + '_median' for x in columns_t}
    temp.rename(columns=columns_t, inplace=True)
    cpdb_g = pd.merge(cpdb_g, temp, on='cas')
    assert cpdb_g_len == len(cpdb_g)

    temp = cpdb_df[['cas', 'td50', ]].copy()
    temp = temp.groupby('cas').apply(hmean)
    temp = temp.reset_index()
    temp.rename(columns={0: 'td50_harmonic'}, inplace=True)
    temp['td50_harmonic'] = [x[0] for x in temp['td50_harmonic']]
    cpdb_g = cpdb_g.merge(temp, on='cas')

    temp = cpdb_df[['cas', 'td50', ]].copy()

    temp['td50'] = np.log(temp['td50'] + 1)
    temp = temp.groupby('cas').apply(hmean)
    temp = temp.reset_index()
    temp.rename(columns={0: 'td50_log_harmonic'}, inplace=True)
    temp['td50_log_harmonic'] = [x[0] for x in temp['td50_log_harmonic']]
    cpdb_g = cpdb_g.merge(temp, on='cas')

    assert cpdb_g_len == len(cpdb_g)

    exclude_cas = ['---', 'mixture']
    cpdb_g = cpdb_g[~cpdb_g['cas'].isin(exclude_cas)]
    return cpdb_g


def visualize_overlap_with_ddb(ddb, cpdbg):
    threshold_direction = 'less_than'
    metric_names = ['td50_log_min', 'td50_min', 'td50_log_mean', 'td50_mean', 'td50_log_median', 'td50_median',
                    'td50_harmonic', 'td50_log_harmonic']
    m_dfs = []
    for metric in metric_names:
        set_of_breaks = set(np.nanpercentile(cpdbg[metric].values, np.arange(0, 100, 1)))
        thresholds = sorted(list(set_of_breaks))
        data = []

        for threshold in thresholds:
            if threshold_direction == 'less_than':
                temp = cpdbg[
                    (cpdbg[metric] <= threshold)
                    #             & (cpdb_g[experiments_column_name] >= min_number_experiments)
                ]
            else:
                temp = cpdbg[
                    (cpdbg[metric] >= threshold)
                    #             & (cpdb_g[experiments_column_name] >= min_number_experiments)
                ]

            n_overlap = len(ddb[ddb['smiles'].isin(temp['smiles'])])
            data.append({
                metric: threshold,
                'ddb_overlap': n_overlap,
                'len_cpdb': len(temp),
                #         'num_required_experiments': min_number_experiments,
            })

        m_df = pd.DataFrame(data)
        m_df['fraction_overlap'] = m_df['ddb_overlap'] / m_df['ddb_overlap'].max()
        m_df['fraction_cpdb'] = m_df['len_cpdb'] / len(cpdbg)
        m_dfs.append(m_df)

    m_dfs = pd.concat(m_dfs)

    plot = plt.figure(figsize=(20, 10));
    metric_names = ['td50_log_min', 'td50_min', 'td50_log_mean', 'td50_mean', 'td50_log_median', 'td50_median',
                    'td50_harmonic', 'td50_log_harmonic']

    for metric_val in [metric_names[1], metric_names[2], metric_names[6], metric_names[7]]:
        temp = m_dfs[m_dfs[metric_val].notnull()]
        if 'log' not in metric_val:
            temp[metric_val] = np.log(temp[metric_val].values)
        plt.plot(temp['fraction_overlap'], temp['fraction_overlap'] / temp['fraction_cpdb'],
                 label=f'ratio of DDB overlap to CPDB percentage {metric_val}')

    plt.ylabel("")
    plt.title(f"Molecule overlap n={m_df['ddb_overlap'].max()} between DDB and CPDB\n as a function of td50")
    plt.legend()
    plt.ylabel('Percentage of DDB overlap')
    plt.xlabel('Percentage of CPDB thresholded by TD50 value')
    plt.savefig('../plots/ddb_overlap_vs_cpdb_ratio.pdf')


def read_ddb():
    ddb = pd.read_csv('../data/ddb_approved.csv')
    drug_df = pd.read_csv('../data/potentially_carcinogenic_drugs.csv')
    print(f"Number of drug data bank entries prior to processing {len(ddb)}")
    # Remove drugs that are either illicit or withdrawn
    ddb = ddb[(~ddb['Drug Groups'].str.contains('withdrawn')) & (~ddb['Drug Groups'].str.contains('illicit'))]
    # remove chemotherapeutics and nucleoside analogues
    ddb = ddb[~ddb['Name'].str.lower().isin(drug_df['drug_name'].str.lower())]
    ddb['smiles'] = GraphCancerMolecules.smiles_standardize(ddb['SMILES'])
    ddb = ddb[ddb['smiles'].notnull()]
    print(f"Number of drug data bank entries after processing {len(ddb)}")
    ddb = ddb[['Name', 'smiles', 'DrugBank ID', 'Drug Groups', 'CAS Number']]
    ddb['class'] = 0

    return ddb


def assign_smiles_values(cpdb_g):
    smiles_data = []

    for index, row in cpdb_g.iterrows():
        if index % 100 == 0:
            print(f"ticker for SMILES {str(datetime.datetime.now())}, {index}")

        try:
            smiles = cirpy.resolve(row['cas'], 'smiles', ['cas_number', 'name_by_cir', 'name_by_opsin'])
            smiles_data.append({'smiles': smiles, 'cas': row['cas']})
        except Exception as error:
            print(f"Error occurred during SMILES assignment {error}")
            continue

    len_old_df = len(cpdb_g)
    cpdbg = pd.merge(
        cpdb_g,
        pd.DataFrame(smiles_data),
        on='cas',
        how='outer',
    )
    assert len(cpdbg) == len_old_df
    cpdbg['smiles'] = GraphCancerMolecules.smiles_standardize(cpdbg['smiles'])
    cpdbg = cpdbg[~cpdbg['smiles'].isnull()]
    return cpdbg


def create_percentiles(cpdbg, thresholds=(100, 80, 60, 40, 20), column_name='carc_class'):
    for bucket, threshold in enumerate(thresholds):
        td50_val = np.nanpercentile(cpdbg['td50_log_harmonic'].values, threshold)
        # highest percent will include all the td50 values
        cpdbg.loc[cpdbg['td50_log_harmonic'] <= td50_val, column_name] = bucket
    return cpdbg


def main():
    cpdb_df = read_in_cpdb()
    ddb = read_ddb()
    genereate_summary_cpdb_plots(cpdb_df)
    cpdb_g = agregate_cpdb(cpdb_df)
    cpdbg = assign_smiles_values(cpdb_g)
    visualize_overlap_with_ddb(ddb, cpdbg)
    cpdbg = create_percentiles(cpdbg, column_name='carc_class_multi')
    cpdbg = create_percentiles(cpdbg, thresholds=(100, 50), column_name='carc_class')
    cpdbg.to_csv('../data/cpdb_aggergated.csv', index=False)


if __name__ == "__main__":
    main()
