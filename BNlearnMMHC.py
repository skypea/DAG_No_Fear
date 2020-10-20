#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 22:00:46 2020

@author: denniswei
"""

import os
import uuid
import networkx as nx
from shutil import rmtree
from cdt.causality.graph.model import GraphModel
from pandas import read_csv
from cdt.utils.R import RPackages, launch_R_script
from cdt.utils.Settings import SETTINGS


class BNlearnMMHC(GraphModel):
    """BNlearn MMHC algorithm.

    Args:
        score (str):the label of the conditional independence test to be used in the
           algorithm. If none is specified, the default test statistic is the mutual information
           for categorical variables, the Jonckheere-Terpstra test for ordered factors and the
           linear correlation for continuous variables. See below for available tests.
        alpha (float): a numeric value, the target nominal type I error rate.
        beta (int): a positive integer, the number of permutations considered for each permutation
           test. It will be ignored with a warning if the conditional independence test specified by the
           score argument is not a permutation test.
        optim (bool): See bnlearn-package for details.
        verbose (bool): Sets the verbosity. Defaults to SETTINGS.verbose

    .. _bnlearntests:

    Available tests:
        • discrete case (categorical variables)
           – mutual information: an information-theoretic distance measure.
               It's proportional to the log-likelihood ratio (they differ by a 2n factor)
               and is related to the deviance of the tested models. The asymptotic χ2 test
               (mi and mi-adf,  with  adjusted  degrees  of  freedom), the Monte Carlo
               permutation test (mc-mi), the sequential Monte Carlo permutation
               test (smc-mi), and the semiparametric test (sp-mi) are implemented.
           – shrinkage estimator for the mutual information (mi-sh)
               An improved
               asymptotic χ2 test based on the James-Stein estimator for the mutual
               information.
           – Pearson’s X2 : the classical Pearson's X2 test for contingency tables.
               The asymptotic χ2 test (x2 and x2-adf, with adjusted degrees of freedom),
               the Monte Carlo permutation test (mc-x2), the sequential Monte Carlo
               permutation test (smc-x2) and semiparametric test (sp-x2) are implemented  .

        • discrete case (ordered factors)
           – Jonckheere-Terpstra : a trend test for ordinal variables.
              The
              asymptotic normal test (jt), the Monte Carlo permutation test (mc-jt)
              and the sequential Monte Carlo permutation test (smc-jt) are implemented.

        • continuous case (normal variables)
           – linear  correlation:  Pearson’s  linear  correlation.
               The exact
               Student’s  t  test  (cor),  the Monte Carlo permutation test (mc-cor)
               and the sequential Monte Carlo permutation test (smc-cor) are implemented.
           – Fisher’s Z: a transformation of the linear correlation with asymptotic normal distribution.
               Used by commercial software (such as TETRAD II)
               for the PC algorithm (an R implementation is present in the pcalg
               package on CRAN). The asymptotic normal test (zf), the Monte Carlo
               permutation test (mc-zf) and the sequential Monte Carlo permutation
               test (smc-zf) are implemented.
           – mutual information: an information-theoretic distance measure.
               Again
               it is proportional to the log-likelihood ratio (they differ by a 2n
               factor). The asymptotic χ2 test (mi-g), the Monte Carlo permutation
               test (mc-mi-g) and the sequential Monte Carlo permutation test
               (smc-mi-g) are implemented.

           – shrinkage estimator for the mutual information(mi-g-sh):
               an improved
               asymptotic χ2 test based on the James-Stein estimator for the mutual
               information.

        • hybrid case (mixed discrete and normal variables)
           – mutual information: an information-theoretic distance measure.
               Again
               it is proportional to the log-likelihood ratio (they differ by a 2n
               factor). Only the asymptotic χ2 test (mi-cg) is implemented.
    """

    def __init__(self, score='NULL', alpha=0.05, beta='NULL',
                 optim=False, verbose=None):
        """Init the model."""
        if not RPackages.bnlearn:
            raise ImportError("R Package bnlearn is not available.")
#        super(BNlearnMMHC, self).__init__()
        self.arguments = {'{FOLDER}': '/tmp/cdt_bnlearn/',
                          '{FILE}': 'data.csv',
                          '{SKELETON}': 'FALSE',
                          '{ALGORITHM}': 'mmhc',
                          '{WHITELIST}': 'whitelist.csv',
                          '{BLACKLIST}': 'blacklist.csv',
                          '{SCORE}': 'NULL',
                          '{OPTIM}': 'FALSE',
                          '{ALPHA}': '0.05',
                          '{BETA}': 'NULL',
                          '{VERBOSE}': 'FALSE',
                          '{OUTPUT}': 'result.csv'}
        self.score = score
        self.alpha = alpha
        self.beta = beta
        self.optim = optim
        self.verbose = SETTINGS.get_default(verbose=verbose)


    def create_graph_from_data(self, data):
        """Run the algorithm on data.

        Args:
            data (pandas.DataFrame): DataFrame containing the data

        Returns:
            networkx.DiGraph: Solution given by the algorithm.

        """
        # Building setup w/ arguments.
        self.arguments['{SCORE}'] = self.score
        self.arguments['{VERBOSE}'] = str(self.verbose).upper()
        self.arguments['{BETA}'] = str(self.beta)
        self.arguments['{OPTIM}'] = str(self.optim).upper()
        self.arguments['{ALPHA}'] = str(self.alpha)

        cols = list(data.columns)
        data.columns = [i for i in range(data.shape[1])]
        results = self._run_bnlearn(data, verbose=self.verbose)
        graph = nx.DiGraph()
        graph.add_nodes_from(['X' + str(i) for i in range(data.shape[1])])
        graph.add_edges_from(results)
        return nx.relabel_nodes(graph, {i: j for i, j in
                                        zip(['X' + str(i) for i
                                             in range(data.shape[1])], cols)})


    def _run_bnlearn(self, data, whitelist=None, blacklist=None, verbose=True):
        """Setting up and running bnlearn with all arguments."""
        # Run the algorithm
        id = str(uuid.uuid4())
        os.makedirs('/tmp/cdt_bnlearn' + id + '/')
        self.arguments['{FOLDER}'] = '/tmp/cdt_bnlearn' + id + '/'

        def retrieve_result():
            return read_csv('/tmp/cdt_bnlearn' + id + '/result.csv', delimiter=',').values

        try:
            data.to_csv('/tmp/cdt_bnlearn' + id + '/data.csv', index=False)
            if blacklist is not None:
                whitelist.to_csv('/tmp/cdt_bnlearn' + id + '/whitelist.csv', index=False, header=False)
                blacklist.to_csv('/tmp/cdt_bnlearn' + id + '/blacklist.csv', index=False, header=False)
                self.arguments['{SKELETON}'] = 'TRUE'
            else:
                self.arguments['{SKELETON}'] = 'FALSE'

            bnlearn_result = launch_R_script("bnlearnMMHC.R",
                                             self.arguments, output_function=retrieve_result, verbose=verbose)
        # Cleanup
        except Exception as e:
            rmtree('/tmp/cdt_bnlearn' + id + '')
            raise e
        except KeyboardInterrupt:
            rmtree('/tmp/cdt_bnlearn' + id + '/')
            raise KeyboardInterrupt
        rmtree('/tmp/cdt_bnlearn' + id)
        return bnlearn_result

