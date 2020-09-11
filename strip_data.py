import os
import pdb
import numpy as np
filename = 'batch_m2.txt'
with open(filename) as f:
    content = f.readlines()
content = [x.strip() for x in content]
content = [x.replace('\t', ' ') for x in content]


logl_list = []
for r in content:
    if 'size:' in r:
        logl_list.append(0)
    else:
        logl_list.append(float(r.split(' ')[5]))
idx = np.argsort(logl_list)

## SET ##
seed_string = 'Seeds: 41-41'
## ABOVE ##

j = 0
for i in range(len(idx)):
    name = content[idx[i]]
    if seed_string in name:
        j +=1
        print(j, content[idx[i]])




