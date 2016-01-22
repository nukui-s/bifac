import os
import time
import pandas as pd
import numpy as np
from bifac import BiFac
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import f1_score

#preparation
logdir = "log/logdaisharin"
os.system("rm -rf log/logdaisharin")
df = pd.read_csv("data/edge_list.csv", header=None)
edge_list = [(i-1, j-1) for i, j in zip(df[0], df[1])]
weights = df[2]
ans = pd.read_csv("data/term_category.csv", header=None)[1]
N = ans.shape[0]
cnst = []
ncnst = 100
while len(cnst) < ncnst:
    n1 = np.random.randint(0, N)
    n2 = np.random.randint(0, N)
    if n1 == n2: continue
    if ans.ix[n1] == ans.ix[n2]:
        cnst.append((n1,n2))

#construct BiFac model
model = BiFac(K=8, edge_list=edge_list, weights=weights, learning_rate=1.0,
              lambda_r=0, constraint=cnst, lambda_c=0.)

#detect communities
start = time.time()
loss, z, theta1, theta2 = model.optimize(logdir=logdir, stop_threshold=10e-8,
                                        max_steps=5000)
sumtime = time.time() - start

com1, com2 = model.get_hard_community()

fvalue = f1_score(com1, ans)
nmi = normalized_mutual_info_score(com1, ans)

print("time", sumtime)
print("fvalue", fvalue)
print("nmi", nmi)
