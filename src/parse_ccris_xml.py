import xml.etree.ElementTree as ET
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

import cirpy
import datetime

import seaborn as sns
sns.set()
sns.set_context('talk')


# Read in XML file
tree_full = ET.parse("../data/ccris.xml.20110828")
full_root = tree_full.getroot()

carc_data = []
unique_fields = ['CASRegistryNumber', 'NameOfSubstance', 'use']
unique_cstu_fields = ["specc", "stsxc","routc","dosec","tstlc","rsltc"]

for doc in full_root:
    data = {}
    for field in unique_fields:
        events = doc.findall(field)
        assert len(events) == 1
        data[field] = events[0].text
    
    dtypes = doc.findall('dtyp')
    dtypes = [x.text for x in dtypes]
    assert not any(pd.Series(dtypes).duplicated())
    
    # can check if there is carcinogenicity
#     if "Carcinogenicity" in dtypes:
    for subdoc in doc:
        # Carcinogenicity study
        if subdoc.tag == 'cstu':
            # copy the element to add the study
            data_copy = data.copy()

            for field in unique_cstu_fields:
                events = subdoc.findall(field)
                assert len(events) <= 1, (events)
                if len(events) == 1:
                    data_copy[field] = events[0].text
                    
            carc_data.append(data_copy)
carc_df = pd.DataFrame(carc_data)
print(f"Number of unique carcinogenic compounds tested: "
      f"{len(pd.unique(carc_df['CASRegistryNumber']))}"
)

# Normalize all the results
carc_df['rsltc'] = carc_df['rsltc'].str.lower().str.strip()
carc_vc = carc_df['rsltc'].value_counts() 
carc_df.replace('$null', np.NaN, inplace=True)

print(f"All carcinogenic result fileds that have more than 5 hits:"
      f"\n{carc_vc[carc_vc > 5].to_string()}")

# Drop annotations that have less than 5 submissions with outcome
drop_result_annotations = carc_vc[carc_vc <= 5].keys()
carc_df = carc_df[~carc_df["rsltc"].isin(drop_result_annotations)]

# List of fields describing positives
positive_results = ['positive']
# Everything negative besides the positives
carc_df['carc_outcome'] = 'negative'
carc_df.loc[carc_df['rsltc'].isin(positive_results), 'carc_outcome'] = 'positive'
print(f"Outcomes of carcinogenic studies based on classification: \n"
      f" {carc_df['carc_outcome'].value_counts()}")

# Perform molecule level carcinogenicity classification from study level results
carc_data2 = []
for name, group in carc_df.groupby('CASRegistryNumber'):
    row_data = {}
    # Ratio of positive results to all
    pos_neg_ratio = (group['carc_outcome'] == 'positive').sum() / len(group['carc_outcome'])
    row_data['ratio_of_positive'] = pos_neg_ratio
    row_data['number_of_experiments'] = len(group)
    row_data['cas_number'] = name
    row_data['substance_name'] = group['NameOfSubstance'].iloc[0]
    row_data['use'] = ",".join([x for x in group['use'].unique() if pd.notnull(x)])    
    
    carc_data2.append(row_data)
    
carc = pd.DataFrame(carc_data2)

# ## Visualize Carcinogenic data
plt.figure()
g = sns.jointplot(
    carc['number_of_experiments'], 
    carc['ratio_of_positive'], 
    kind='scatter',
    alpha=0.05,
    height=10
)
g.ax_joint.set_xscale('log')
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=3)
plt.suptitle("Carcinogenic Experiments distribution")
g.fig.tight_layout()
g.fig.subplots_adjust(top=0.9) 
plt.savefig('../plots/carcinogenic_experiment_dist.pdf')

# Individual experiment level plots
plt.figure()
sns.histplot(carc['number_of_experiments'], linewidth= 3,alpha= 0.7, color= "g", binwidth=1)
plt.xlim(0,15)
plt.xlabel('Number of experiments')
plt.title("Number of Experiments per molecule Carcinogenicity")
plt.tight_layout()
plt.savefig('../plots/number_carcinogenic_experiment.pdf')

plt.figure()
sns.histplot(carc['ratio_of_positive'], linewidth= 3,alpha= 0.7, color= "g", binwidth=.1)
plt.xlabel('ratio of positives to all')
plt.title("Ratio of positive Carcinogenic results per molecule")
plt.tight_layout()
plt.savefig('../plots/ratio_carcinogenic_experiment.pdf')


# ## Process mutagenic data
mstu_data = []
unique_fields = ['CASRegistryNumber','NameOfSubstance','use']
unique_mstu_fields = [
    "endpm","tsstm","indcm","specm","stsxm","routm","matvm","methm","dosem","dosrm","rsltm",]

for doc in full_root:
    data = {}
    for field in unique_fields:
        events = doc.findall(field)
        assert len(events) == 1
        data[field] = events[0].text
    
    dtypes = doc.findall('dtyp')
    dtypes = [x.text for x in dtypes]
    # Make sure for every molecule there is only unique studies
    assert not any(pd.Series(dtypes).duplicated())
    
    # can check if there is carcinogenicity
    # if "Carcinogenicity" in dtypes:
    for subdoc in doc:
        # Carcinogenicity study
        if subdoc.tag == 'mstu':
            # copy the element to add the study
            data_copy = data.copy()

            for field in unique_mstu_fields:
                events = subdoc.findall(field)
                assert len(events) <= 1, (events)
                if len(events) == 1:
                    data_copy[field] = events[0].text
                    
            mstu_data.append(data_copy)
mstu_df = pd.DataFrame(mstu_data)
print(f"Number of unique mutagenic compounds tested: {len(pd.unique(mstu_df['CASRegistryNumber']))}")

# Clean results
mstu_df['rsltm'] = mstu_df['rsltm'].str.lower().str.strip()
mstu_df.replace('$null', np.NaN, inplace=True)
mstu_vc = mstu_df['rsltm'].value_counts() 

# Drop annotations that have less than 5 submissions with outcome
print(f"All mutagenicity result fileds that have more"
      f" than 5 hits :\n{mstu_vc[mstu_vc > 5].to_string()}")

drop_result_annotations = mstu_vc[mstu_vc <= 5].keys()
mstu_df = mstu_df[~mstu_df["rsltm"].isin(drop_result_annotations)]

# Manually go through the list and decide what fits the bill
positive_results = [
    'positive (structural changes)',
    'positive',
    'positive. reevaluation of results: positive',
    "positive (structural and ploidy changes)",
    "positive (ploidy changes)",
    "positive (p<0.01)",
    "positive (weak)",
    "positive (weak response)",
    "positive; weak response",
    "positive (ploidy)",
    "positive; precipitation occurred at all doses",
    "positive (p<0.001)",
    "positive, test conducted under both aerobic and anaerobic conditions.",
    "positive. precipitation at 3 highest doses.",
    "positive (two experiments)",
    "positive (decrease in mutagenicity when 0.99-15.90 mg/plate of\
chloroform extract of byrsonima basiloba added)",
    "positive (at highest dose only)",
    "positive (growth inhibition observed at 2 highest doses)",
    "positive (at higher doses)",
    "positive (p<0.05)",
    "positive; 97% pure",
]
mstu_df['mstu_outcome'] = 'negative'
mstu_df.loc[mstu_df['rsltm'].isin(positive_results),'mstu_outcome'] = 'positive'
mstu_df['mstu_outcome'].value_counts()

# For every compound calculate if there are most positive studies than negative
mstu_data2 = []
for name, group in mstu_df.groupby('CASRegistryNumber'):
    row_data = {}
    pos_neg_ratio = (group['mstu_outcome'] == 'positive').sum() / len(group['mstu_outcome'])
    row_data['ratio_of_positive'] = pos_neg_ratio
    row_data['number_of_experiments'] = len(group)
    row_data['cas_number'] = name
    row_data['substance_name'] = group['NameOfSubstance'].iloc[0]
    row_data['use'] = ",".join([x for x in group['use'].unique() if pd.notnull(x)])    
    
    mstu_data2.append(row_data)
    
mstu = pd.DataFrame(mstu_data2)

# ## Visualize Mutagenic data
plt.figure()
g = sns.jointplot(
    mstu['number_of_experiments'], 
    mstu['ratio_of_positive'], 
    kind='scatter',
    alpha=0.05,
    height=10,
)
g.ax_joint.set_xscale('log')
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=3)
plt.suptitle("Mutagenic Experiments distribution")
g.fig.tight_layout()
g.fig.subplots_adjust(top=0.9) 
plt.savefig('../plots/mutagenic_experiment_dist.pdf')

# Individual experiments
plt.figure()
sns.histplot(mstu["number_of_experiments"], linewidth= 3,alpha= 0.7, color= "g", binwidth=1)
plt.xlim(0,15)
plt.xlabel('Number of experiments')
plt.title("Number of Experiments per molecule Mutagenicity")
plt.tight_layout()
plt.savefig('../plots/number_mutagenic_experiment.pdf')

plt.figure()
sns.histplot(mstu['ratio_of_positive'], linewidth= 3,alpha= 0.7, color= "g", binwidth=.1)
plt.xlabel('ratio of positives to all')
plt.title("Ratio of positive mutagenic results per molecule")
plt.tight_layout()
plt.savefig('../plots/ratio_mutagenic_experiment.pdf')


# ## Combine mutagenic and carcinogenic data
print(f"Number of strong positives in mutagenicity data "
      f"(more than 50% positive experiments and at least 2 studies) "
      f"\n{len(mstu[(mstu['ratio_of_positive'] > 0.5) & (mstu['number_of_experiments'] > 1)])}")
print(f"Number of positives in mutagenicity data "
      f"(more than 50% positive experiments {len(mstu[(mstu['ratio_of_positive'] > 0.5)])})")
mstu['positive'] = 0
mstu.loc[mstu['ratio_of_positive'] > 0.5, 'positive'] = 1
mstu['positive_stringent'] = 0
mstu.loc[(mstu['ratio_of_positive'] > 0.5) & (mstu['number_of_experiments'] > 1), 'positive_stringent'] = 1

print(f"Number of strong negatives in mutagenicity data "
      f"(more than 50% negative experiments and at least 2 studies) "
      f"\n{len(mstu[(mstu['ratio_of_positive'] < 0.2) & (mstu['number_of_experiments'] > 3)])}")
print(f"Number of negatives in mutagenicity data "
      f"(less than 50% positive experiments {len(mstu[(mstu['ratio_of_positive'] < 0.2)])})")
mstu['negative'] = 0
mstu.loc[mstu['ratio_of_positive'] < 0.2, 'negative'] = 1
mstu['negative_stringent'] = 0
mstu.loc[(mstu['ratio_of_positive'] < 0.2) & (mstu['number_of_experiments'] > 3), 'negative_stringent'] = 1

print(f"Number of strong positives in carcinogenicity data "
      f"(more than 50% positive experiments and at least 2 studies) "
      f"\n{len(carc[(carc['ratio_of_positive'] > 0.5) & (carc['number_of_experiments'] > 1)])}")
print(f"Number of positives in carcinogenicity data "
      f"(more than 50% positive experiments {len(carc[(carc['ratio_of_positive'] > 0.5)])})")
carc['positive'] = 0
carc.loc[carc['ratio_of_positive'] > 0.5, 'positive'] = 1
carc['positive_stringent'] = 0
carc.loc[(carc['ratio_of_positive'] > 0.5) & (carc['number_of_experiments'] > 1), 'positive_stringent'] = 1

print(f"Number of strong negatives in carcinogenicity data "
      f"(more than 50% negative experiments and at least 2 studies) "
      f"\n{len(carc[(carc['ratio_of_positive'] < 0.2) & (carc['number_of_experiments'] > 3)])}")
print(f"Number of negatives in carcinogenicity data "
      f"(less than 50% positive experiments {len(carc[(carc['ratio_of_positive'] < 0.2)])})")
carc['negative'] = 0
carc.loc[carc['ratio_of_positive'] < 0.2, 'negative'] = 1
carc['negative_stringent'] = 0
carc.loc[(carc['ratio_of_positive'] < 0.2) & (carc['number_of_experiments'] > 3), 'negative_stringent'] = 1

# Merge carcinogenicity data and mutagenicity data
df = pd.merge(
    carc,
    mstu,
    on=['cas_number', 'substance_name'],
    how='outer',
    suffixes=['_carc', '_mut']
)
df['positive'] = 0
df['positive_stringent'] = 0
# Positive is if a compound is either carcinogenic or mutagenic
df.loc[(df['positive_mut'] == 1) | (df['positive_carc'] == 1), 'positive'] = 1
df.loc[(df['positive_stringent_mut'] == 1) | (df['positive_stringent_carc'] == 1), 'positive_stringent'] = 1

df['negative'] = 0
df['negative_stringent'] = 0
# Negative if it's negative for both or absent in one of them
df.loc[(
    (df['negative_mut'] == 1) & (df['negative_carc'] == 1) |
    (df['negative_mut'] == 1) & (df['negative_carc'].isnull()) |
    (df['negative_mut'].isnull()) & (df['negative_carc'] == 1)),
    'negative'] = 1

# Negative stringent if it's negative stringent for both or if it's absent in one of them
df.loc[(
    (df['negative_stringent_mut'] == 1) & (df['negative_stringent_carc'] == 1) |
    (df['negative_stringent_mut'] == 1) & (df['negative_stringent_carc'].isnull()) |
    (df['negative_stringent_mut'].isnull()) & (df['negative_stringent_carc'] == 1)),
    'negative_stringent'] = 1


df['use'] = df['use_mut'].fillna('')+ ',' + df['use_carc'].fillna("")

print(f"Number of positive compounds: \n{df['positive'].value_counts()}")
print(f"Number of positive stringent compounds: \n{df['positive_stringent'].value_counts()}")

print(f"Number of negative compounds: \n{df['negative'].value_counts()}")
print(f"Number of negative stringent compounds: \n{df['negative_stringent'].value_counts()}")
# Can't have overlap critieria exclusionary
assert (df['positive'] & df['negative']).sum() == 0


# Add smiles strings
data=[]
old_df_len = len(df)
print(old_df_len)

for index, row in df.iterrows():
    if index%100 == 0:
        print(str(datetime.datetime.now()), index)
    
    try:
        smiles = cirpy.resolve(row['cas_number'], 'smiles', ['cas_number', 'name_by_cir', 'name_by_opsin'])
        data.append({'smiles': smiles, 'cas_number':row['cas_number']})
    except Exception as error:
        print(error)
        continue
    
df = pd.merge(
    df,
    pd.DataFrame(data),
    on='cas_number',
    how='outer',
)
assert len(df) == old_df_len
df.to_csv('../data/ccri_smiles.csv', index=False)
print(df.head())
