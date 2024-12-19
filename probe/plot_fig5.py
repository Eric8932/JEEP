import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy
import scipy.stats as st

result_path = ""#result_path of probe.py
res = np.load(result_path,allow_pickle=True).item()
problem_set = [7,8,132,376]

start = 10
end = 32

fig = plt.figure(figsize=(12,6), dpi=300)
process_key_list = ['original_answer','factual_related', 'nonfactual_related']
target_key_map = {"original_answer":"Original Answer",'factual_related':"Factual","nonfactual_related":"Non-Factual",}

for i,key in enumerate(process_key_list) :
    l1 = []
    for k in res:
        if k in problem_set:
            continue
        l_layer = []
        for l in range(start,end):
            target_ = res[k]['pre_w_ln'][0][l][process_key_list.index(key)*4+1*2+1]
            l_layer.append((target_))
        l1.append(l_layer)
    col = ['#FA5A5A','orange','#F3D343'][i]
    ls = "-"
    lw = [6,6,6][i]
    l1 = np.array(l1)
    plt.plot(np.arange(start,end),np.mean(l1,0),label = target_key_map[key],linewidth=3,color=col,linestyle=ls,marker = "o")
    bot,top = st.norm.interval(confidence=0.95, loc=np.mean(l1,0), scale=st.sem(l1,0))
    plt.fill_between(np.arange(start,end),bot,top,facecolor=col,alpha=0.2)
    


l1= []
for k in res:
    if k in problem_set:
        continue
    l_layer = []
    for l in range(start,end):
        target_ = res[k]['pre_w_ln'][0][l][22]
        l_layer.append((target_))
    l1.append(l_layer)
plt.plot(np.arange(start,end),np.mean(l1,0),label = "Stopwords",linewidth=3,color="grey",linestyle="-",marker="x")
bot,top = st.norm.interval(confidence=0.95, loc=np.mean(l1,0), scale=st.sem(l1,0))
plt.fill_between(np.arange(start,end),bot,top,facecolor="grey",alpha=0.2)


l1= []
for k in res:
    if k in problem_set:
        continue
    l_layer = []
    for l in range(start,end):
        target_ = res[k]['pre_w_ln'][0][l][24]
        l_layer.append((target_))
    l1.append(l_layer)
plt.plot(np.arange(start,end),np.mean(l1,0),label = "Unrelated",linewidth=3)


plt.xlabel('Layers',fontsize=20)
plt.ylabel('Probability',fontsize=20)
plt.xticks([10,20,30],['0-10','20','30'],fontsize=15)
plt.yticks(fontsize=15)

plt.grid(axis='y')  
plt.legend(fontsize=20)
ax = plt.gca()


plt.tight_layout()
plt.show()


