import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy
import scipy.stats as st

result_path = ""#result_path of probe.py
res = np.load(result_path,allow_pickle=True).item()

range_l = [[0,11],[10,20],[19,32]]

target_list = ["True","New"]
module_list = ["sl_l","l_l"]
module_list += ['sl_emb','l_emb']
process_key_list = ['original_answer','target_answer','factual_related', 'nonfactual_related']

fig, axs = plt.subplots(2, 1, figsize=(14, 10), dpi=300,sharey='row')
problem_set = [7,8,132,376]

i = 0

start,end = 0,32
for ax_i,ax in enumerate(axs):
    for key in ['original_answer','target_answer']:
        c = 1
        target_module = ["sl_l","l_l"][ax_i]
        l1,l2 = [],[]
        
        for k in res:
            if k in problem_set:
                continue
            l_layer = []
            for l in range(start,end):
                if l ==0:
                    target_ = res[k]['pre_w_ln'][0][l][process_key_list.index(key)*8+module_list.index(target_module)*4+0][0]
                else:
                    target_ = res[k]['pre_w_ln'][0][l][process_key_list.index(key)*4+module_list.index(target_module)*2+0][0]
                l_layer.append((target_))  
            l1.append(l_layer)
    
        col = (['#FF6347','#2EBCE8']*2)[i]
        ls = '-'
        ax.plot(np.arange(start,end),np.mean(l1,0),label = key,color = col,linestyle =ls,linewidth=4,marker='o')
        
        bot,top = st.norm.interval(confidence=0.95, loc=np.mean(l1,0), scale=st.sem(l1,0))
        ax.fill_between(np.arange(start,end),bot,top,facecolor=col,alpha=0.2)
        i+=1
        title = ['Subject Last','Prediction'][ax_i]
        x_min, x_max = ax.get_xlim()
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.text(-0.023, -0.145, title, transform=ax.transAxes, va='center', ha='center', fontsize=22,fontweight='bold')
        ax.set_ylabel('Rank \n\u2192',fontsize=24)
        ax.yaxis.set_label_coords(-0.06, 0.5)
        ax.set_xlim(left=x_min+0.5, right=x_max-0.5)

        ax.axvline(x=10, color='gray', linestyle='-',linewidth=0.5)
        ax.axvline(x=19, color='gray', linestyle='-',linewidth=0.5)
        ax.text(5, -0.15, 'Low Layers', ha='center', va='center', transform=ax.get_xaxis_transform(), fontsize=22)
        ax.text(15, -0.15, 'Middle Layers', ha='center', va='center', transform=ax.get_xaxis_transform(), fontsize=22)
        ax.text(25, -0.15, 'High Layers', ha='center', va='center', transform=ax.get_xaxis_transform(), fontsize=22)
        if ax_i == 0:
            y_ticks = ax.get_yticks()
            if 0 not in y_ticks:
                y_ticks = list(y_ticks) + [0]
                y_ticks.sort()
            ax.set_yticks(y_ticks)
        if i%2==0:
            ax.invert_yaxis()
            

twin_axes = []
for ax_i,ax in enumerate(axs):
    ax = ax.twinx()
    twin_axes.append(ax)
    for key in ['original_answer','target_answer']:
        c = 1
        target_module = ["sl_l","l_l"][ax_i]
        l1,l2 = [],[]
        
        for k in res:
            if k in problem_set:
                continue
            l_layer = []
            for l in range(start,end):
                if l ==0:
                    target_ = res[k]['pre_w_ln'][0][l][process_key_list.index(key)*8+module_list.index(target_module)*4+1]
                else:
                    target_ = res[k]['pre_w_ln'][0][l][process_key_list.index(key)*4+module_list.index(target_module)*2+1]
                l_layer.append((target_))  
            l1.append(l_layer)
        col = (['#FF7347','#2ECCE8']*4)[i]
        ls = '--'
        ax.plot(np.arange(start,end),np.mean(l1,0),label = key,color = col,linestyle =ls,linewidth=4,marker='x' )
        bot,top = st.norm.interval(confidence=0.95, loc=np.mean(l1,0), scale=st.sem(l1,0))
        ax.fill_between(np.arange(start,end),bot,top,facecolor=col,alpha=0.2)
        i+=1
        x_min, x_max = ax.get_xlim()
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.set_xlim(left=x_min + 0.05, right=x_max-0.05)
            
        ax.set_ylabel('\u2192\nProbability',fontsize=24)
        ax.yaxis.set_label_coords(1.02, 0.5)        
        ax.tick_params(labelleft=False)

y_lims = [ax.get_ylim() for ax in twin_axes]
min_y = min(y[0] for y in y_lims)
max_y = max(y[1] for y in y_lims)
for ax in twin_axes:
    ax.set_ylim(min_y, max_y)

color_l = ['#FF6347','#2EBCE8']+['#FF7347','#2ECCE8']
lines = [
    Line2D([0], [0], color=color_l[0], linestyle='-', linewidth=3,marker='o'),
    Line2D([0], [0], color=color_l[2], linestyle='--', linewidth=3,marker='o'),
    Line2D([0], [0], color=color_l[1], linestyle='-', linewidth=3,marker='x'),
    Line2D([0], [0], color=color_l[3], linestyle='--', linewidth=3,marker='x')
]
fig.legend(lines, ['Original Rank','Original Prob',"Target Rank","Target Prob"] , loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.985),frameon=False,fontsize=22)

plt.subplots_adjust(wspace=0.1)
plt.subplots_adjust(hspace=0.3)
plt.show()



