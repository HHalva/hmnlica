import pylab as plt 
import seaborn as sns 
import numpy as np 
import os 
import pickle 
import pandas as pd
from pandas.io.json import json_normalize #package for flattening json in pandas df
import json

import pdb


# load data
data = pd.read_csv("results.csv")


# load TCL data
def load_log(path):
    d = []
    with open(path, 'r') as f:
        i = 0
        for line in f:
            i += 1
            try:
                d.append(json.loads(line))
            except:
                print(i)
    df = json_normalize(data=d).set_index('id')
    return df.rename(lambda x: x.split('.')[-1], axis='columns').sort_index()

tcl_data = load_log("tcl.json")
tcl_data = tcl_data[['d', 'mixing_layers', 'full_perf', 'elbo',
                     'data_seed', 'seed']]
tcl_best_seeds = tcl_data.groupby(["mixing_layers",
                                   "data_seed"]).max().reset_index()
pdb.set_trace()






# produce plot
n_obs_ = np.array([2, 3, 4, 5])
algos = ['HM-ICA']
marker_dict = {'HM-ICA': 'v', 'TCL': 'o', 'bVAE': 's', 'chance': '|'}
line_dict = {'HM-ICA': 'solid', 'TCL': '--', 'bVAE': ':', 'chance': '--'}

sns.set_style("whitegrid")
sns.set_palette('deep')
ns.set_context('paper', font_scale=1.25)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
for l in [1, 2, 4, 5]:
    for a in algos:
        ax1.plot(n_obs_, data[data.L == l]['corr'],
                 label=str(a) + ' (L=' + str(l) + ')',
                 marker=marker_dict[a], color=sns.color_palette()[l],
                 linestyle=line_dict[a], linewidth=2)
        ax2.plot(n_obs_, data[data.L == l]['acc'],
                 label = str(a) + ' (L=' + str(l) + ')',
                 marker=marker_dict[a], color = sns.color_palette()[l],
                 linestyle=line_dict[a], linewidth=2)

#add random chance to 2nd chart
ax1.plot(n_obs_, np.array([-9, -9, -9, -9]),
         label = 'Random chance=1/C',
         marker=marker_dict['chance'], color = sns.color_palette()[9],
         linestyle=line_dict['chance'], linewidth=2)

ax2.plot(n_obs_, np.array([0.2, 0.14, 0.11, 0.09]),
         label = 'Random chance=1/C',
         marker=marker_dict['chance'], color = sns.color_palette()[9],
         linestyle=line_dict['chance'], linewidth=2)




ax1.set_xlabel('Number of independent components')
ax2.set_xlabel('Number of independent components (C=2N+1)')
ax1.set_ylabel('Mean Correlation Coefficient')
ax2.set_ylabel('Mean accuracy')

ax1.set_xticks([2, 3, 4, 5])
ax2.set_xticks([2, 3, 4, 5])

ax1.set_title('Independent component estimation')
ax2.set_title('Unsupervised clustering')
ax1.legend(loc='best', fontsize=7)
f.tight_layout()

ax1.set_ylim([0, 1])
pdb.set_trace()
plt.savefig('../ExpsResults.pdf', dpi=301)
