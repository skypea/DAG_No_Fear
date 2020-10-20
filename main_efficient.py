
''''
Main function for traininng GCN DAG

'''


from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
from tqdm import tqdm
import os.path

import numpy as np
import networkx as nx

import utils
import BPR


def get_args():
    parser = argparse.ArgumentParser()

    # -----------data parameters ------
    # configurations
    parser.add_argument('--data_type', type=str, default= 'synthetic',
                        choices=['synthetic', 'discrete', 'real'],
                        help='choosing which experiment to do.')
    parser.add_argument('--data_sample_size', type=int, default=1000,
                        help='the number of samples of data')
    parser.add_argument('--data_variable_size', type=int, default=100,
                        help='the number of variables in synthetic generated data')
    parser.add_argument('--graph_type', type=str, default='erdos-renyi',
                        choices=['barabasi-albert','erdos-renyi'],
                        help='the type of DAG graph by generation method')
    parser.add_argument('--graph_degree', type=int, default=2,
                        help='the number of degree in generated DAG graph')
    parser.add_argument('--graph_sem_type', type=str, default='linear-gumbel',
                        choices=['linear-gauss','linear-exp','linear-gumbel'],
                        help='the structure equation model (SEM) parameter type')
    parser.add_argument('--x_dims', type=int, default=1, # data dimension
                        help='The number of input dimensions: default 1.')

    # -----------training hyperparameters
    parser.add_argument('--repeat', type=int, default= 100,
                        help='the number of times to run experiments to get mean/std')

    parser.add_argument('--methods', type=str, default='BPR_lars',
                        choices=[
                                 # 'BPR_all_new', 'BPR_all_new_search',           # ****DO THIS*** new notear
                                 'BPR_lars',                                    # ***DO THIS*** local search only
                                 'all_l1', #'all_l1_search',                     # ***DO THIS*** proposed (l2) -> l1
                                 'all_l2', #'all_l2_search',                     # new notear
                                 'CAM', 'GES', 'MMPC', 'MMHC', 'FGS', 'PC'      # baselines
                                 ] ,
                        help='which method to test') # BPR_all = notear
    # parser.add_argument('--use_solution_path', type=int, default = 0,
    #                     help = 'whether to use solution path to search for L1 coefficient')

    ## LARSE parameters
    parser.add_argument('--minimize_Z', type=int, default = 1,
                        help = 'whether to minimize constrained set Z in LARS')
    # parser.add_argument('--augment_Z', type=bool, default = True,
    #                     help = 'whether to aurgment the constained set Z in LARS')
    parser.add_argument('--revEdges', type=str, default='alt-full',
                        choices = ['both', 'loss', 'alternate', 'alt-early','lower', 'alt-full', ''],
                        help = 'check reverse in LARS, the suggest default is alt-full')
    # parser.add_argument('--checkLARS', type = bool, default = False,
    #                     help = 'lars assertion ')

    # parser.add_argument('--optimizer', type = str, default = 'weighted_L1', choices=['weighted_L1', 'sklearn_l1'],
    #                     help = 'the choice of optimizer used')
    parser.add_argument('--graph_threshold', type=  float, default = 0.3,  # 0.3 is good, 0.2 is error prune
                        help = 'threshold for learned adjacency matrix binarization')
    parser.add_argument('--tau_A', type = float, default= 0.1,
                        help='coefficient for L-1 norm of A.')
    parser.add_argument('--alpha_A',  type = float, default= 10., #corresponding to alpha
                        help='coefficient for DAG constraint h(A).')
    parser.add_argument('--rho_A',  type = float, default= 0.1, #corresponding to rho, needs to  be >> lambda
                        help='coefficient for absolute value h(A).')
    parser.add_argument('--rho_A_max', type=float, default=1e+16,  # corresponding to rho, needs to  be >> lambda
                        help='coefficient for absolute value h(A).')
    parser.add_argument('--power_A', type=int, default=1, choices=[1,2], # Power used to define adjacency matrix
                        help='power of A in h(X) to be regularized ')

    parser.add_argument('--h_tol', type=float, default = 1e-10,
                        help='the tolerance of error of h(A) to zero')
    parser.add_argument('--pre_h_tol', type= float, default=1e-5,
                        help='the tolerance of h(A) to zero for some preprocessing step')
    parser.add_argument('--pre_use_l2', type= int, default = 1,
                        help='1=use L2 before L1 netears, 0 = all zero W init')

    parser.add_argument('--W_tol', type=float, default=1e-10,
                        help='the tolerance of W/A')
    parser.add_argument('--pen_tol', type=float, default=0,
                        help='the tolerance of penTol')
    parser.add_argument('--noPen', type=bool, default=True,
                        help='use of penalty noPen')
    parser.add_argument('--zero_mean', type=int , default = 1,
                        help ='normalize data in the begining or not')
    # parser.add_argument('--alphaCIT', type=float, default=0.01,
    #                     help='target false positive rate for conditional independence tests')

    parser.add_argument('--seed', type=int, default = 42, help='Random seed.')
    parser.add_argument('--train_epochs', type=int, default= 1e4,
                        help='Number of epochs to train.')

    parser.add_argument('--generate_data', type=int, default=0,
                        help='generate new data or use old data')
    parser.add_argument('--file_name', type = str, default = 'test_')
    parser.add_argument('--search', type=int, default= 3,
                        help='whether to do search after the method')
    # parser.add_argument('--save-folder', type=str, default='logs',
    #                     help='Where to save the trained model, leave empty to not save anything.')
    # parser.add_argument('--load-folder', type=str, default='',
    #                     help='Where to load the trained model if finetunning. ' +
    #                          'Leave empty to train from scratch')

    # -----------parsing
    args = parser.parse_args()


    return args

def main(args):

    # Generate and import data
    n, d = args.data_sample_size, args.data_variable_size # samples, variable size
    graph_type, degree, sem_type = args.graph_type, args.graph_degree, args.graph_sem_type

    # book keeping for results
    num_trials = args.repeat
    result_time = np.zeros((num_trials, 1))
    result_tpr = np.zeros((num_trials, 1))
    result_fpr = np.zeros((num_trials, 1))
    result_shd = np.zeros((num_trials, 1))
    result_nnz = np.zeros((num_trials, 1))
    result_fdr = np.zeros((num_trials, 1))
    result_h = np.zeros((num_trials, 1))
    result_extra = np.zeros((num_trials, 1))
    result_missing = np.zeros((num_trials, 1))
    result_reverse = np.zeros((num_trials, 1))

    # local search copy TODO both copies and generate data
    if args.search:
        result_time_search = np.zeros((num_trials, 1))
        result_tpr_search = np.zeros((num_trials, 1))
        result_fpr_search = np.zeros((num_trials, 1))
        result_shd_search = np.zeros((num_trials, 1))
        result_nnz_search = np.zeros((num_trials, 1))
        result_fdr_search = np.zeros((num_trials, 1))
        result_h_search = np.zeros((num_trials, 1))
        result_extra_search = np.zeros((num_trials, 1))
        result_missing_search = np.zeros((num_trials, 1))
        result_reverse_search = np.zeros((num_trials, 1))

    pc = None
    if args.methods == 'FGS':
        from pycausal.pycausal import pycausal as pc

        pc = pc()
        pc.start_vm()

    for trial_index in tqdm(range(num_trials)):
        # Hack to reuse 1000-sample datasets for smaller sample sizes
        sample_size_str = str(args.data_sample_size)
        if args.data_sample_size < 1000:
            sample_size_str = '1000'
        file_name = 'data/' + sample_size_str + '_' + str(args.data_variable_size) + '_' \
                    + str(args.graph_type) + '_' + str(args.graph_degree) + '_' \
                    + str(args.graph_sem_type) + '_' + str(trial_index) + '.pkl'

        # whether to generate data or not
        if args.generate_data and not os.path.exists(file_name):
            G = utils.simulate_random_dag(d, degree, graph_type)
            # G = np.array( [[0,1,0],[0,0,0],[0,1,0]])
            G = nx.DiGraph(G)
            X = utils.simulate_sem(G, args.data_sample_size, sem_type)

            with open(file_name, "wb") as f:
                pickle.dump( (G, X), f)
        else:
            with open(file_name, "rb") as f:
                G, X = pickle.load(f)

        if X.ndim > 2:
            X = X[:, :, 0]
        if args.data_sample_size < 1000:
            X = X[:args.data_sample_size, :]


        # Class BPR
        methods = args.methods

        # for method in methods:
        method = methods
        t =  time.time()
        bpr = BPR.BPR(args, G, pc = pc)

        A, h, alpha, rho = bpr.fit(X, method)
        result_time[trial_index] =  time.time() - t

        if args.search:
            A_search, h_search, itSearch = bpr.search_given_A(X, A, initNoPen=(args.search==3))
            result_time_search[trial_index] = time.time() - t
        if args.search == 2:
            A[np.abs(A) < args.graph_threshold] = 0

        result_h[trial_index] = h[-1]

        # logs results
        logger.info('Testing Method Done: %s' % method)

        G_est = nx.DiGraph(A)
        logger.info('Solving equality constrained problem ... Done')
        # evaluate
        fdr, tpr, fpr, shd, nnz, extra, missing, reverse = utils.count_accuracy_new(G, G_est)
        logger.info('Accuracy: fdr %f, tpr %f, fpr %f, shd %d, nnz %d',
                 fdr, tpr, fpr, shd, nnz)
        result_shd[trial_index] = shd
        result_nnz[trial_index] = nnz
        result_tpr[trial_index] = tpr
        result_fpr[trial_index] = fpr
        result_fdr[trial_index] = fdr

        result_extra[trial_index] = extra
        result_missing[trial_index] = missing
        result_reverse[trial_index] = reverse

        if args.search:
            result_h_search[trial_index] = h_search[-1]

            # logger.info('Testing  Method Done: %s' % method)
            # logger.info(np.column_stack((h, alpha, rho)))

            G_est_search = nx.DiGraph(A_search)
            # logger.info('Solving equality constrained problem ... Done')
            # evaluate
            fdr_search, tpr_search, fpr_search, shd_search, nnz_search, extra_search, missing_search, reverse_search =\
                utils.count_accuracy_new(G, G_est_search)
            logger.info('Search Accuracy: fdr %f, tpr %f, fpr %f, shd %d, nnz %d',
                        fdr_search, tpr_search, fpr_search, shd_search, nnz_search)
            result_shd_search[trial_index] = shd_search
            result_nnz_search[trial_index] = nnz_search
            result_tpr_search[trial_index] = tpr_search
            result_fpr_search[trial_index] = fpr_search
            result_fdr_search[trial_index] = fdr_search

            result_extra_search[trial_index] = extra_search
            result_missing_search[trial_index] = missing_search
            result_reverse_search[trial_index] = reverse_search

    logger.info('Accuracy: fdr ' + str(np.mean(result_fdr).item()) + '$\pm$' + str(np.std(result_fdr).item()) +
                ', tpr ' + str(np.mean(result_tpr).item()) + '$\pm$' + str(np.std(result_tpr).item()) +
                ', fpr ' + str(np.mean(result_fpr).item()) + '$\pm$' + str(np.std(result_fpr).item()) +
                ', h ' + str(np.mean(result_h).item()) + '$\pm$' + str(np.std(result_h).item()) +
                ', shd ' + str(np.mean(result_shd).item()) + '$\pm$' + str(np.std(result_shd).item()) +
                ', nnz ' + str(np.mean(result_nnz).item()) + '$\pm$' + str(np.std(result_nnz).item()) +
                ', time ' + str(np.mean(result_time).item()) + '$\pm$' + str(np.std(result_time).item()))

    logger.info('Edges: extra ' + str(np.mean(result_extra).item()) + '$\pm$' + str(np.std(result_extra).item()) +
                ', missing ' + str(np.mean(result_missing).item()) + '$\pm$' + str(np.std(result_missing).item()) +
                ', revgerse ' + str(np.mean(result_reverse).item()) + '$\pm$' + str(np.std(result_reverse).item())
                )

    utils.print_to_file(args,
                        result_time,
                        result_shd,
                        result_nnz,
                        result_h,
                        result_extra,
                        result_missing,
                        result_reverse,
                        search_result=0
                        )

    if args.search:
        logger.info('Search Accuracy: fdr ' + str(np.mean(result_fdr_search).item()) + '$\pm$' + str(np.std(result_fdr_search).item()) +
                            ', tpr '+ str(np.mean(result_tpr_search).item()) + '$\pm$' + str(np.std(result_tpr_search).item()) +
                            ', fpr '+ str(np.mean(result_fpr_search).item())+ '$\pm$' + str(np.std(result_fpr_search).item()) +
                            ', h ' + str(np.mean(result_h_search).item()) + '$\pm$' + str(np.std(result_h_search).item()) +
                            ', shd '+ str(np.mean(result_shd_search).item())+ '$\pm$' + str(np.std(result_shd_search).item()) +
                            ', nnz '+ str(np.mean(result_nnz_search).item()) + '$\pm$' + str(np.std(result_nnz_search).item()) +
                            ', time ' + str(np.mean(result_time_search).item()) + '$\pm$' + str(np.std(result_time_search).item()) )

        logger.info('Search Edges: extra ' + str(np.mean(result_extra_search).item()) + '$\pm$' + str(np.std(result_extra_search).item()) +
                    ', missing ' + str(np.mean(result_missing_search).item()) + '$\pm$' + str(np.std(result_missing_search).item()) +
                    ', reverse ' + str(np.mean(result_reverse_search).item()) + '$\pm$' + str(np.std(result_reverse_search).item())
                    )
        utils.print_to_file(args,
                            result_time_search,
                            result_shd_search,
                            result_nnz_search,
                            result_h_search,
                            result_extra_search,
                            result_missing_search,
                            result_reverse_search,
                            search_result=args.search
                            )
    # logger.info('Time:  %f' % )

    if args.methods == 'FGS':
        pc.stop_vm()



if __name__ == "__main__":

    args = get_args()
    logger = utils.setup_logger(mode='deubg')
    logger.info(args)

    main(args)
    print(args)



