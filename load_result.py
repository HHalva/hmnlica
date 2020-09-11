import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pdb

fileid = "id99N3K7L2"+".npz"
loaded = np.load(fileid, allow_pickle=True)['a'].item()

# load results
N = loaded['N']
K = loaded['K']
mix_depth = loaded['L']
hidden_units = loaded['hidden_units']
learning_rate = loaded['lr']
decay_rate = loaded['dr']
decay_interval = loaded['di']
distrib_seed = loaded['seed']
s_est = loaded['s_est']
est_seq = loaded['est_seq']
results = loaded['results']
est_params = loaded['est_params']
sort_idx = loaded['sort_idx']

# unpack results
logl_hist, corr_hist, acc_hist = results
max_logl_idx = np.argmax(logl_hist)
print(corr_hist[max_logl_idx])
print(acc_hist[max_logl_idx])
pdb.set_trace()
