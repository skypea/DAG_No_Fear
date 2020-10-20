#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 23:16:05 2020

@author: denniswei
"""

import os
import numpy as np
import pandas as pd

# Method attributes
dfMethods = [('notears', 'NOTEARS', 'BPR_all_new'),
#             ('notearsl1', 'NOTEARS-L1', 'all_l2'),
             ('notearsl1', 'NOTEARS', 'all_l2'),
             ('hybrid', 'Hybrid', 'all_l1'),
             ('hybridNT', 'HybridNT', 'all_l1'),
             ('abs', 'Abs', 'all_l1'),
             ('search', 'Search', 'BPR_lars'),
             ('CAM', 'CAM', 'CAM'),
             ('FGS', 'FGS', 'FGS'),
             ]
dfMethods = pd.DataFrame(dfMethods, columns=['method','name','filename'])
dfMethods.set_index('method', inplace=True)

# Dimensions and metrics
ds = [10, 30, 50, 100]
metrics = ['SHD', 'nnz', 'time']

# Results folder
dirResults = os.path.join('results', 'nips_2020')

n = 1000
#hTol = 1e-10

# Graph type, degree, SEM type combinations
graphTypes = {'ER': 'erdos-renyi', 'SF': 'barabasi-albert'}
SEMtypes = {'gaussian': 'linear-gauss', 'gumbel': 'linear-gumbel', 'exp': 'linear-exp'}
tableType = 'final'
if tableType == 'nzm':
    graphDegSEMs = [('ER', 4, 'gaussian'), ('ER', 4, 'gumbel')]
elif tableType == 'MinZRev':
    graphDegSEMs = [('ER', 4, 'gumbel')]
elif tableType == 'final':
    graphDegSEMs = [
                    ('ER', 2, 'gaussian'),
                    ('ER', 4, 'gaussian'),
#                    ('SF', 4, 'gaussian'),
#                    ('ER', 2, 'gumbel'),
#                    ('ER', 4, 'gumbel'),
#                    ('SF', 4, 'gumbel'),
#                    ('ER', 2, 'exp'),
#                    ('ER', 4, 'exp'),
#                    ('SF', 4, 'exp'),
                    ]
else:
    graphDegSEMs = [('ER', 2, 'gumbel'), ('ER', 4, 'gumbel'), 
                    ('ER', 2, 'gaussian'), ('ER', 4, 'gaussian'),
                    ('SF', 4, 'exp')]

# Iterate over graph types, degrees, and SEM types
for gt, deg, st in graphDegSEMs:
    # Attributes of table rows
    if tableType == 'nzm':
        dfRows = [
                  ('notearsl1', False, 0, 1e-5, 1e-10, ''),
                  ('notearsl1', True, 0, 1e-5, 1e-10, ''),
                  ('notearsl1', False, 3, 1e-5, 1e-10, ''),
                  ('notearsl1', True, 3, 1e-5, 1e-10, ''),
                  ('abs', False, 0, 1e-5, 1e-10, ''),
                  ('abs', True, 0, 1e-5, 1e-10, ''),
                  ('abs', False, 3, 1e-5, 1e-10, ''),
                  ('abs', True, 3, 1e-5, 1e-10, ''),
                  ('FGS', False, 0, 1e-5, 1e-10, ''),
                  ('FGS', True, 0, 1e-5, 1e-10, ''),
                  ('FGS', False, 3, 1e-5, 1e-10, ''),
                  ('FGS', True, 3, 1e-5, 1e-10, ''),
#                  ('CAM', False, 0, 1e-5, 1e-10, ''),
#                  ('CAM', True, 0, 1e-5, 1e-10, ''),
#                  ('CAM', False, 3, 1e-5, 1e-10, ''),
#                  ('CAM', True, 3, 1e-5, 1e-10, ''),
                  ]
    elif tableType == 'MinZRev':
        dfRows = [
                  ('notearsl1', True, 0, 1e-5, 1e-10, ''),
                  ('notearsl1', True, 3, 1e-5, 1e-10, ''),
                  ('notearsl1', True, 3, 1e-5, 1e-10, 'noMinZ'),
                  ('notearsl1', True, 3, 1e-5, 1e-10, 'noRev'),
                  ('abs', True, 0, 1e-5, 1e-10, ''),
                  ('abs', True, 3, 1e-5, 1e-10, ''),
                  ('abs', True, 3, 1e-5, 1e-10, 'noMinZ'),
                  ('abs', True, 3, 1e-5, 1e-10, 'noRev'),
                  ]
    elif tableType == 'final':
        dfRows = [('notearsl1', True, 0, 1e-5, 1e-10, ''),
                  ('notearsl1', True, 3, 1e-5, 1e-10, ''),
                  ('notearsl1', True, 0, 1e-5, 1e-5, ''),
                  ('notearsl1', True, 3, 1e-5, 1e-5, ''),
                  ('abs', True, 0, 1e-5, 1e-10, ''),
                  ('abs', True, 3, 1e-5, 1e-10, ''),
                  ('FGS', True, 0, 1e-5, 1e-10, ''),
                  ('FGS', True, 3, 1e-5, 1e-10, ''),
                  ('search', True, 0, 1e-5, 1e-10, ''),
                  ('CAM', True, 0, 1e-5, 1e-10, ''),
                  ('CAM', True, 3, 1e-5, 1e-10, ''),
                  ]        
    elif (gt == 'ER') and (st == 'gumbel'):
        dfRows = [('notears', False, 0, 1e-5, 1e-10, ''),
                  ('notears', False, 1, 1e-5, 1e-10, ''),
                  ('notears', False, 3, 1e-5, 1e-10, ''),
                  ('notears', True, 0, 1e-5, 1e-10, ''),
                  ('notears', True, 1, 1e-5, 1e-10, ''),
                  ('notears', True, 3, 1e-5, 1e-10, ''),
                  ('notearsl1', False, 0, 1e-5, 1e-10, ''),
                  ('notearsl1', False, 3, 1e-5, 1e-10, ''),
                  ('notearsl1', True, 0, 1e-5, 1e-10, ''),
                  ('notearsl1', True, 3, 1e-5, 1e-10, ''),
                  ('hybrid', True, 0, 1e-4, 1e-10, ''),
                  ('hybrid', True, 1, 1e-4, 1e-10, ''),
                  ('hybrid', True, 0, 1e-5, 1e-10, ''),
                  ('hybrid', True, 1, 1e-5, 1e-10, ''),
                  ('hybrid', True, 3, 1e-5, 1e-10, ''),
                  ('hybridNT', True, 0, 1e-5, 1e-10, ''),
                  ('hybridNT', True, 3, 1e-5, 1e-10, ''),
                  ('abs', True, 0, 1e-5, 1e-10, ''),
                  ('abs', True, 1, 1e-5, 1e-10, ''),
                  ('abs', True, 3, 1e-5, 1e-10, ''),
                  ('search', True, 0, 1e-5, 1e-10, ''),
                  ('search', True, 1, 1e-5, 1e-10, ''),
                  ('search', True, 3, 1e-5, 1e-10, ''),
                  ('CAM', False, 0, 1e-5, 1e-10, ''),
                  ('CAM', False, 1, 1e-5, 1e-10, ''),
                  ('CAM', False, 3, 1e-5, 1e-10, ''),
                  ('CAM', True, 0, 1e-5, 1e-10, ''),
                  ('CAM', True, 1, 1e-5, 1e-10, ''),
                  ('CAM', True, 3, 1e-5, 1e-10, ''),
                  ('FGS', False, 0, 1e-5, 1e-10, ''),
                  ('FGS', False, 3, 1e-5, 1e-10, ''),
                  ('FGS', True, 0, 1e-5, 1e-10, ''),
                  ('FGS', True, 3, 1e-5, 1e-10, ''),
                  ]
    elif gt == 'SF':
        dfRows = [('notearsl1', True, 0, 1e-5, 1e-10, ''),
                  ('notearsl1', True, 3, 1e-5, 1e-10, ''),
                  ('hybrid', True, 0, 1e-5, 1e-10, ''),
                  ('hybrid', True, 3, 1e-5, 1e-10, ''),
                  ('hybridNT', True, 0, 1e-5, 1e-10, ''),
                  ('hybridNT', True, 3, 1e-5, 1e-10, ''),
                  ('abs', True, 0, 1e-5, 1e-10, ''),
                  ('abs', True, 3, 1e-5, 1e-10, ''),
                  ('search', True, 0, 1e-5, 1e-10, ''),
                  ('search', True, 3, 1e-5, 1e-10, ''),
                  ('FGS', True, 0, 1e-5, 1e-10, ''),
                  ('FGS', True, 3, 1e-5, 1e-10, ''),
                  ]
        
    dfRows = pd.DataFrame(dfRows, columns=['method','zm','search','preh','hTol','minZRev'])
    # Row labels for table
    dfRows['label'] = dfRows['method'].map(dfMethods['name'])
    dfRows.loc[dfRows['search']==1, 'label'] += ', search'
    dfRows.loc[dfRows['search']==3, 'label'] += '-KKTS'
#    dfRows.loc[dfRows['zm'], 'label'] += ', zm'
    dfRows.loc[~dfRows['zm'], 'label'] += '-nzm'
    dfRows.loc[dfRows['method']=='hybrid', 'label'] += ', preh ' + \
        dfRows.loc[dfRows['method']=='hybrid', 'preh'].astype(str)
    dfRows.loc[dfRows['minZRev']=='noMinZ', 'label'] += '-noReduce'
    dfRows.loc[dfRows['minZRev']=='noRev', 'label'] += '-noReverse'

    # Initialize table
    dfTbl = pd.DataFrame(index=dfRows.index, columns=pd.MultiIndex.from_product([ds, metrics]))
    # Iterate over table rows
    for i in dfRows.index:
        # Iterate over dimensions
        for d in ds:
            # Construct filename
            filename = [dfMethods.at[dfRows.at[i,'method'], 'filename'],
                        str(d),
                        'synthetic',
                        str(n),
                        graphTypes[gt],
                        SEMtypes[st],
                        str(deg),
                        'zeroInit' if dfRows.at[i,'method'] == 'abs' else 'notearInit',
                        'zeroM' if dfRows.at[i,'zm'] else 'nonZero',
                        'PrehTol_' + str(dfRows.at[i,'preh']),
                        'hTol_' + str(dfRows.at[i,'hTol']),
                        ]
            if dfRows.at[i,'search'] == 1:
                filename[0] += '_search'
            if dfRows.at[i,'search'] == 3:
                filename[0] += '_searchINP'
            if dfRows.at[i,'method'] == 'hybridNT':
                filename[7] += 'NT'
            if dfRows.at[i,'minZRev']:
                filename.append(dfRows.at[i,'minZRev'])
            filename = pd.Series(filename).str.cat(sep='_')
            
            # Read results DataFrame
            try:             
                dfRes = pd.read_pickle(os.path.join(dirResults, filename + '.pkl'))
            except (FileNotFoundError, ModuleNotFoundError):
                continue
            numTrialsSqrt = np.sqrt(dfRes.shape[0])
            
            # Iterate over metrics
            for m in metrics:
                # Format mean and standard error of metric
                numDec = 1 if m == 'time' else 2
                dfTbl.loc[i, (d,m)] = format(dfRes[m].mean(), '0.{}f'.format(numDec)) +\
                    '$\\pm$' + format(dfRes[m].std() / numTrialsSqrt, '0.{}f'.format(numDec))
                
    # Replace index with row labels
    dfTbl.index = dfRows['label']
    
    for dsPrint in [[10, 30], [50, 100]]:
        # Save subset of table
        dfPrint = dfTbl[dsPrint].copy()
        # Append newline symbol
        dfPrint[dfPrint.columns[-1]] += '\\\\'
        dfPrint[dfPrint.columns[-1]].fillna('\\\\', inplace=True)
        
        # Save as delimited text file
        filename = 'tab_' + gt + str(deg) + st + '_d' + str(dsPrint[0]) + str(dsPrint[-1])
        if (tableType == 'nzm') or (tableType == 'MinZRev'):
            filename += '_' + tableType
        dfPrint.to_csv(os.path.join(dirResults, filename + '.dlm'), sep='&', header=False, index_label=False)
    