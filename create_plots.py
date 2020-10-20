#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 22:25:37 2020

@author: denniswei
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Method attributes
dfMethods = [('notears', 'NOTEARS', 'BPR_all_new', 'r'),
             ('notearsl1', 'NOTEARS', 'all_l2', 'r'),
#             ('hybrid', 'hybrid', 'all_l1', 'b'),
             ('abs', 'Abs', 'all_l1', 'b'),
             ('search', 'KKTS', 'BPR_lars', 'm'),
             ('CAM', 'CAM', 'CAM', 'c'),
             ('FGS', 'FGS', 'FGS', 'g'),
             ('MMHC', 'MMHC', 'MMHC', 'c'),
             ('PC', 'PC', 'PC', 'gray'),
             ('GES', 'GES', 'GES', 'k'),
             ]
dfMethods = pd.DataFrame(dfMethods, columns=['method','label','filename','color'])
dfMethods.set_index('method', inplace=True)

# Figure name
figName = 'RebutMix'
# Attributes of plot lines
if figName == 'nzm':
    dfLines = [('notearsl1', True, 0, 1e-5, 1e-10, ''),
               ('notearsl1', True, 3, 1e-5, 1e-10, ''),
               ('notearsl1', False, 0, 1e-5, 1e-10, ''),
               ('notearsl1', False, 3, 1e-5, 1e-10, ''),
               ('abs', True, 0, 1e-5, 1e-10, ''),
               ('abs', True, 3, 1e-5, 1e-10, ''),
               ('abs', False, 0, 1e-5, 1e-10, ''),
               ('abs', False, 3, 1e-5, 1e-10, ''),
#               ('search', True, 0, 1e-5, 1e-10, ''),
#               ('search', False, 0, 1e-5, 1e-10, ''),
               ]
elif figName == 'MinZRev':
    dfLines = [('notearsl1', True, 0, 1e-5, 1e-10, ''),
               ('notearsl1', True, 3, 1e-5, 1e-10, ''),
               ('notearsl1', True, 3, 1e-5, 1e-10, 'noMinZ'),
               ('notearsl1', True, 3, 1e-5, 1e-10, 'noRev'),
               ('abs', True, 0, 1e-5, 1e-10, ''),
               ('abs', True, 3, 1e-5, 1e-10, ''),
               ('abs', True, 3, 1e-5, 1e-10, 'noMinZ'),
               ('abs', True, 3, 1e-5, 1e-10, 'noRev'),
               ]
elif figName == 'Rebut':
    dfLines = [('notearsl1', True, 0, 1e-5, 1e-10, ''),
               ('notearsl1', True, 3, 1e-5, 1e-10, ''),
               ('abs', True, 0, 1e-5, 1e-10, ''),
               ('abs', True, 3, 1e-5, 1e-10, ''),
               ('FGS', True, 0, 1e-5, 1e-10, ''),
               ('FGS', True, 3, 1e-5, 1e-10, ''),
               ('MMHC', True, 0, 1e-5, 1e-10, ''),
               ('MMHC', True, 3, 1e-5, 1e-10, ''),
               ('PC', True, 0, 1e-5, 1e-10, ''),
               ('PC', True, 3, 1e-5, 1e-10, ''),
               ('search', True, 0, 1e-5, 1e-10, ''),
               ]
elif figName == 'Rebut_n_2d':
    dfLines = [('notearsl1', True, 0, 1e-5, 1e-10, ''),
               ('notearsl1', True, 3, 1e-5, 1e-10, ''),
               ('abs', True, 0, 1e-5, 1e-10, ''),
               ('abs', True, 3, 1e-5, 1e-10, ''),
               ('FGS', True, 0, 1e-5, 1e-10, ''),
               ('FGS', True, 3, 1e-5, 1e-10, ''),
               ('MMHC', True, 0, 1e-5, 1e-10, ''),
               ('MMHC', True, 3, 1e-5, 1e-10, ''),
#               ('PC', True, 0, 1e-5, 1e-10, ''),
#               ('PC', True, 3, 1e-5, 1e-10, ''),
               ('search', True, 0, 1e-5, 1e-10, ''),
               ]
elif figName == 'RebutMix':
    dfLines = [('notearsl1', True, 0, 1e-5, 1e-10, ''),
               ('notearsl1', True, 3, 1e-5, 1e-10, ''),
               ('abs', True, 0, 1e-5, 1e-10, ''),
               ('abs', True, 3, 1e-5, 1e-10, ''),
               ('FGS', True, 0, 1e-5, 1e-10, ''),
               ('FGS', True, 3, 1e-5, 1e-10, ''),
               ('search', True, 0, 1e-5, 1e-10, ''),
               ('GES', True, 0, 1e-5, 1e-10, ''),
               ('GES', True, 3, 1e-5, 1e-10, ''),
               ('MMHC', True, 0, 1e-5, 1e-10, ''),
               ('MMHC', True, 3, 1e-5, 1e-10, ''),
               ('PC', True, 0, 1e-5, 1e-10, ''),
               ('PC', True, 3, 1e-5, 1e-10, ''),
               ]
else:
    dfLines = [('notearsl1', True, 0, 1e-5, 1e-10, ''),
               ('notearsl1', True, 3, 1e-5, 1e-10, ''),
               ('notearsl1', True, 0, 1e-5, 1e-5, ''),
               ('notearsl1', True, 3, 1e-5, 1e-5, ''),
    #           ('hybrid', True, 0, 1e-5, 1e-10, ''),
    #           ('hybrid', True, 3, 1e-5, 1e-10, ''),
               ('abs', True, 0, 1e-5, 1e-10, ''),
               ('abs', True, 3, 1e-5, 1e-10, ''),
    #           ('CAM', True, 0, 1e-5, ''),
    #           ('CAM', True, 3, 1e-5, ''),
               ('FGS', True, 0, 1e-5, 1e-10, ''),
               ('FGS', True, 3, 1e-5, 1e-10, ''),
               ('search', True, 0, 1e-5, 1e-10, ''),
               ]
dfLines = pd.DataFrame(dfLines, columns=['method','zm','search','preh','hTol','minZRev'])
# Legend labels
dfLines['label'] = dfLines['method'].map(dfMethods['label'])
#dfLines.loc[dfLines['hTol']!=1e-10, 'label'] += '-' + dfLines.loc[dfLines['hTol']!=1e-10, 'hTol'].astype(str)
for i in dfLines.index:
    if dfLines.at[i, 'hTol'] != 1e-10:
        dfLines.at[i, 'label'] += '-' + np.format_float_scientific(dfLines.at[i, 'hTol'], trim='-', exp_digits=1)
dfLines.loc[dfLines['search']>0, 'label'] += '-KKTS'
dfLines.loc[~dfLines['zm'], 'label'] += '-nzm'
dfLines.loc[dfLines['minZRev']=='noMinZ', 'label'] += '-noReduce'
dfLines.loc[dfLines['minZRev']=='noRev', 'label'] += '-noReverse'

# Dimensions and metrics
ds = [10, 30, 50, 100]
if figName.startswith('Rebut'):
    metrics = ['SHD']
else:
    metrics = ['SHD', 'time']

# Results folder
dirResults = os.path.join('results', 'nips_2020')
dirFigures = 'figures'

n = 1000
#hTol = 1e-10

graphTypes = {'ER': 'erdos-renyi', 'SF': 'barabasi-albert'}
SEMtypes = {'gaussian': 'linear-gauss', 'gumbel': 'linear-gumbel', 'exp': 'linear-exp'}
if figName in ['Main', 'Rebut', 'Rebut_n_2d']:
    graphDegSEMs = [('ER', 2, 'gaussian'), ('ER', 4, 'gumbel'), ('SF', 4, 'exp')]
elif figName == 'RebutMix':
    graphDegSEMs = [('ER', 2, 'gaussian'), ('ER', 4, 'gumbel'), ('ER', 2, 'gaussian')]
elif figName == 'nzm':
    graphDegSEMs = [('ER', 4, 'gaussian'), ('ER', 4, 'gumbel')]
elif figName == 'MinZRev':
    graphDegSEMs = [('ER', 4, 'gumbel')]
elif figName == 'Gauss':
    graphDegSEMs = [('ER', 2, 'gaussian'), ('ER', 4, 'gaussian'), ('SF', 4, 'gaussian')]
elif figName == 'Gumbel':
    graphDegSEMs = [('ER', 2, 'gumbel'), ('ER', 4, 'gumbel'), ('SF', 4, 'gumbel')]
elif figName == 'Exp':
    graphDegSEMs = [('ER', 2, 'exp'), ('ER', 4, 'exp'), ('SF', 4, 'exp')]
colsSubPlot = len(graphDegSEMs)

# Initialize figure
#fig = plt.figure()
fig = plt.figure(figsize=(2.*colsSubPlot+(figName=='Main'), 1.7*len(metrics)))
fig.subplotpars.wspace = 0.3
colSubPlot = 1
# Iterate over graph types, degrees, and SEM types
for gt, deg, st in graphDegSEMs:
    # Initialize mean and standard error DataFrames
    dfMean = pd.DataFrame(index=ds, columns=pd.MultiIndex.from_product([metrics, dfLines['label']]))
    dfSE = pd.DataFrame(index=ds, columns=pd.MultiIndex.from_product([metrics, dfLines['label']]))
    
    # Iterate over dimensions
    for d in ds:
        # Iterate over plot lines
        for i in dfLines.index:
            # Construct filename
            if figName.endswith('_n_2d') or (figName == 'RebutMix' and colSubPlot == 3):
                nStr = str(2 * d)
            else:
                nStr = str(n)
            filename = [dfMethods.at[dfLines.at[i,'method'], 'filename'],
                        str(d),
                        'synthetic',
                        nStr,
                        graphTypes[gt],
                        SEMtypes[st],
                        str(deg),
                        'zeroInit' if dfLines.at[i,'method'] == 'abs' else 'notearInit',
                        'zeroM' if dfLines.at[i,'zm'] else 'nonZero',
                        'PrehTol_' + str(dfLines.at[i,'preh']),
                        'hTol_' + str(dfLines.at[i,'hTol']),
                        ]
            if dfLines.at[i,'search'] == 1:
                filename[0] += '_search'
            if dfLines.at[i,'search'] == 3:
                filename[0] += '_searchINP'
            if dfLines.at[i,'minZRev']:
                filename.append(dfLines.at[i,'minZRev'])
            if dfLines.at[i,'method'] == 'MMHC':
                if figName == 'Rebut':
                    if ((gt,deg,st) == ('ER',2,'gaussian') and d in [50,100]) or \
                    ((gt,deg,st) == ('SF',4,'exp') and d == 100):
                        alphaCIT = 0.01
                    else:
                        alphaCIT = 0.05
                elif figName == 'Rebut_n_2d':
                    if ((gt,deg,st) == ('ER',2,'gaussian') and d in [50,100]) or \
                    ((gt,deg,st) == ('SF',4,'exp') and d in [30,50,100]):
                        alphaCIT = 0.01
                    else:
                        alphaCIT = 0.05
                elif figName == 'RebutMix':
                    if (gt,deg,st) == ('ER',2,'gaussian') and d in [50,100]:
                        alphaCIT = 0.01
                    else:
                        alphaCIT = 0.05
                filename.append('alphaCIT_' + str(alphaCIT))
            filename = pd.Series(filename).str.cat(sep='_')
            
            # Read results DataFrame
            try:             
                dfRes = pd.read_pickle(os.path.join(dirResults, filename + '.pkl'))
            except FileNotFoundError:
                continue
            numTrialsSqrt = np.sqrt(dfRes.shape[0])
            
            # Iterate over metrics
            for m in metrics:
                # Compute mean and standard error
                dfMean.loc[d, (m,dfLines.at[i,'label'])] = dfRes[m].mean()
                dfSE.loc[d, (m,dfLines.at[i,'label'])] = dfRes[m].std() / numTrialsSqrt
    
    # Plot applicable methods
    title = gt + str(deg) + ', ' + st
    if figName == 'RebutMix':
        if colSubPlot == 3:
            title += ', n=2d'
        else:
            title += ', n=1000'
    for mm, m in enumerate(metrics):
        ax = fig.add_subplot(len(metrics), colsSubPlot+(figName=='Main'), mm*(colsSubPlot+(figName=='Main'))+colSubPlot)
#        if m == 'time':
        ax.set_yscale('log', nonposy='clip')
        ax.grid(which='major', axis='y')
        legH = [None] * len(dfLines)
        for i in dfLines.index:
            if dfLines.at[i,'hTol'] == 1e-5:
                if dfLines.at[i,'method'] == 'notearsl1':
                    if dfLines.at[i,'minZRev'] == 'noMinZ':
                        color = 'tan'
                    elif dfLines.at[i,'minZRev'] == 'noRev':
                        color = 'gold'
                    else:
                        color = 'orange'
            elif not dfLines.at[i,'zm']: 
                if dfLines.at[i,'method'] == 'notearsl1':
                    color = 'pink'
                elif dfLines.at[i,'method'] == 'abs': 
                    color = 'c'
            elif dfLines.at[i,'minZRev'] == 'noMinZ':
                if dfLines.at[i,'method'] == 'notearsl1':
                    color = 'darkred'
                elif dfLines.at[i,'method'] == 'abs':
                    color = 'darkblue'
            elif dfLines.at[i,'minZRev'] == 'noRev':
                if dfLines.at[i,'method'] == 'notearsl1':
                    color = 'salmon'
                elif dfLines.at[i,'method'] == 'abs':
                    color = 'lightblue'
#            elif (dfLines.at[i,'method'] == 'search') and not dfLines.at[i,'zm']:
#                color = 'violet'
            else:
                color = dfMethods.at[dfLines.at[i,'method'],'color']
            legH[i], _, _ = ax.errorbar(ds, dfMean[(m,dfLines.at[i,'label'])],# / ds,
                                        yerr=dfSE[(m,dfLines.at[i,'label'])],# / ds, 
                                        c=color, marker='', linestyle='--' if dfLines.at[i,'search'] > 0 else '-')
        ax.set_xticks(ds)
        ax.xaxis.set_tick_params(labelsize='xx-small')
        ax.yaxis.set_tick_params(labelsize='xx-small')
        if mm == len(metrics) - 1:
            ax.set_xlabel('d', fontsize='x-small')
        if colSubPlot == 1:
            yLabel = m
            if m == 'time':
                yLabel += ' [seconds]'
            ax.set_ylabel(yLabel, fontsize='x-small')
        if mm == 0:
            ax.set_title(title, fontsize='x-small')
        
        # In main figure, re-plot ER4-gumbel SHD plot on a linear scale in last column
        if (figName == 'Main') and ((gt, deg, st) == ('ER', 4, 'gumbel')) and (m == 'SHD'):
            ax = fig.add_subplot(len(metrics), colsSubPlot+(figName=='Main'), colsSubPlot+(figName=='Main'))
            ax.grid(which='major', axis='y')
            legH = [None] * len(dfLines.loc[dfLines['method']=='notearsl1'])
            for i in dfLines.loc[dfLines['method']=='notearsl1'].index:
                if dfLines.at[i,'hTol'] == 1e-5:
                    if dfLines.at[i,'method'] == 'notearsl1':
                        if dfLines.at[i,'minZRev'] == 'noMinZ':
                            color = 'tan'
                        elif dfLines.at[i,'minZRev'] == 'noRev':
                            color = 'gold'
                        else:
                            color = 'orange'
                elif not dfLines.at[i,'zm']: 
                    if dfLines.at[i,'method'] == 'notearsl1':
                        color = 'pink'
                    elif dfLines.at[i,'method'] == 'abs': 
                        color = 'c'
                elif dfLines.at[i,'minZRev'] == 'noMinZ':
                    if dfLines.at[i,'method'] == 'notearsl1':
                        color = 'darkred'
                elif dfLines.at[i,'minZRev'] == 'noRev':
                    if dfLines.at[i,'method'] == 'notearsl1':
                        color = 'salmon'
    #            elif (dfLines.at[i,'method'] == 'search') and not dfLines.at[i,'zm']:
    #                color = 'violet'
                else:
                    color = dfMethods.at[dfLines.at[i,'method'],'color']
                legH[i], _, _ = ax.errorbar(ds, dfMean[(m,dfLines.at[i,'label'])],# / ds,
                                            yerr=dfSE[(m,dfLines.at[i,'label'])],# / ds, 
                                            c=color, marker='', linestyle='--' if dfLines.at[i,'search'] > 0 else '-')
            ax.set_xticks(ds)
            ax.xaxis.set_tick_params(labelsize='xx-small')
            ax.yaxis.set_tick_params(labelsize='xx-small')
            if mm == len(metrics) - 1:
                ax.set_xlabel('d', fontsize='x-small')
            if colSubPlot == 1:
                yLabel = m
                if m == 'time':
                    yLabel += ' [seconds]'
                ax.set_ylabel(yLabel, fontsize='x-small')
            if mm == 0:
                ax.set_title(title, fontsize='x-small')

#    if colSubPlot == np.ceil((colsSubPlot + 1) / 2):
#        # Center legend close to middle
#        ax.legend(handles=legH, labels=dfLines['label'].tolist(), loc='upper center', 
#                  bbox_to_anchor=(0.5*(colsSubPlot % 2), -0.3), ncol=5, fontsize='x-small')
    if (colSubPlot == colsSubPlot):
        # Legend to right of last column
        if figName == 'Main':
            bbox = (1.05, 0.44)
        elif figName.startswith('Rebut'):
            bbox = (1.05, 0.50)
        else:
            bbox = (1.1, 1.1)
        ax.legend(handles=legH, labels=dfLines['label'].tolist(), loc='center left', 
                  bbox_to_anchor=bbox, ncol=1+(figName=='RebutMix'), fontsize='x-small', columnspacing=1.0)
        
    colSubPlot += 1

#fig.tight_layout()

# Save figure
plt.savefig(os.path.join(dirFigures, 'fig' + figName + '.pdf'), bbox_inches='tight')
