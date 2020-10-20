'''
main class for bayesian network penalized regression

Tian Gao, 3/6/2020
'''


import numpy as np
from scipy.linalg import expm

import pandas as pd
#import spams
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from lars_path_weighted import fit_zero_cons
# from lars_path_weighted import fit_zero_cons
# from fixed_point_lars import fit_fixed_point
# from aug_lagr_Aabs import fit_aug_lagr_Aabs, \
#     fit_aug_lagr_Aabs_search, fit_lagr_path, \
#     fit_aug_lagr_AL2, fit_aug_lagr_AL2proj_search, fit_aug_lagr_AL2projnew
from local_search_given_matrix import local_search_given_W
import networkx as nx

from BNlearnMMHC import BNlearnMMHC

class BPR:
    def __init__(self, args, G = None, pc = None):
        self.args = args
        self.rho = args.rho_A
        self.rho_max = args.rho_A_max # augmented Langagragian coefficent
        self.tau = args.tau_A  # L-1 coefficent of A
        self.h_tol = args.h_tol
        self.train_epochs = args.train_epochs
        # self.method = method # args.methods
        self.power_A = args.power_A
        self.threshold_A = args.graph_threshold
        self.loss_type = 'l2'
        # self.use_solution_path = args.use_solution_path
        # self.use_dag_proj = args.use_dag_proj

        self.minimizeZ = args.minimize_Z
        # self.augmentZ = args.augment_Z
        # self.checkLARS = args.checkLARS
        self.revEdges = args.revEdges
        self.W_tol = args.W_tol
        self.pen_tol = args.pen_tol
        self.noPen = args.noPen
        # self.alphaCIT = args.alphaCIT
        self.pre_h_tol = args.pre_h_tol
        self.pre_use_l2 = args.pre_use_l2
        self.search = args.search
        self.ground_truth_G = G
        self.pc = pc

        # if args.optimizer == 'weighted_L1' and args.search < 1:
        #     # see doc: http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams005.html#sec13
        #     # self.model = spams.lassoWeighted
        #     # self.model_param = {'L': -1, 'pos': False,
        #     #          'lambda1': 0.15, 'numThreads': 2, 'mode': spams.PENALTY}
        #     self.model_param =  { 'L' : 20000,
        #             'lambda1' : self.tau, 'numThreads' : 1, 'mode' : 2} # spams.PENALTY}
        #     # W = np.asfortranarray(np.random.random(size=(D.shape[1], X.shape[1])), dtype=myfloat)
        #     # alpha = spams.lassoWeighted(X, D, W, **param)
        #
        # # else:
        # #     raise Exception('Optimization Method is not implemented.')

    def fit(self, X, method = 'BPR_indep'):

        # normalize to zero mean
        # Subtract mean from X
        if self.args.zero_mean:
            X -= X.mean(axis=0)

        # if method == 'BPR_all_new':
        #     # --new version of notear
        #     return self.fit_all_new(X)
        #
        # elif method == 'BPR_all_new_search':
        #     return self.fit_all_new_search(X)

        # elif method == 'BPR_dep':
        #     # ADMM version of notear
        #     return self.fit_dep(X)
        #

        # elif method =='BPR_fixP':
        #     return self.fit_indep_fixP(X)

        if method =='all_l1':
            return self.fit_all_L1(X)

        elif method == 'all_l1_search':
            return self.fit_all_L1_search(X)

        elif method =='all_l2':
            return self.fit_all_L2(X)

        elif method == 'all_l2_search':
            return self.fit_all_L2_search(X)

        elif method == 'BPR_lars':
            # use LARS weighted path
            return self.fit_indep_lars(X)

        elif method =='CAM':
            return self.fit_cam(X)

        elif method =='GES':
            return self.fit_ges(X)

        elif method == 'MMPC':
            return self.fit_mmpc(X)

        elif method == 'MMHC':
            return self.fit_mmhc(X)

        elif method =='FGS':
            return self.fit_fgs(X)

        elif method =='PC':
            return self.fit_pc(X)

        else:
            print('method is not support')

    #
    # def fit_different_search(self, X, init_func, learn_func):
    #
    #     preset_htol = deepcopy(self.h_tol)
    #     self.h_tol = 1e-4
    #     A, h_iter, alpha, rho_iter = init_func(X)
    #
    #     self.h_tol = preset_htol
    #     A, h_iter, alpha, rho_iter = learn_func(X)
    #
    #     Wstar, h, itSearch = local_search_given_W(X,
    #                                               W=A,
    #                                               tau=self.tau,
    #                                               hTol=self.h_tol,
    #                                               Wtol=self.args.W_tol,
    #                                               penTol=self.args.pen_tol,
    #                                               revEdges=self.revEdges,
    #                                               noPen=False)
    #
    #     Wstar[np.abs(Wstar) < self.threshold_A] = 0
    #
    #     return Wstar, h_iter, alpha, rho_iter

    def fit_fgs(self, X):
        '''FGS version'''
        from fges_continuous_yyu import fit_FGS

        d = X.shape[1]
        trueG = nx.to_numpy_array(self.ground_truth_G)
        A = fit_FGS(X, trueG, d, self.pc)

        return A, [-1], [], []

    def fit_cam(self, X):
        import cdt
        model = cdt.causality.graph.CAM(score='nonlinear', cutoff=0.001,
                                        variablesel=True, selmethod='gamboost',
                                        pruning=True, prunmethod='gam',
                                        njobs=None, verbose=None)  # causal additive model: Guasisan process + additive noise
        # model = cdt.causality.graph.LiNGAM()  # Linear Non-Gaussian Acyclic model + addtive noise
        #

        data_frame = pd.DataFrame(X)
        output_graph_nc = model.predict(data_frame)
        A = nx.adjacency_matrix(output_graph_nc).todense()

        A = np.asarray(A).astype(np.float64)

        return A, [-1], [],[]

    def fit_mmpc(self, X):
        import cdt
        model = cdt.causality.graph.bnlearn.MMPC()
        model.alpha = self.alphaCIT

        data_frame = pd.DataFrame(X)
        output_graph_nc = model.predict(data_frame)
        A = nx.adjacency_matrix(output_graph_nc).todense()
        A = np.asarray(A).astype(np.float64)

        return A, [-1], [], []

    def fit_pc(self, X):
        import cdt
        model = cdt.causality.graph.PC()
        data_frame = pd.DataFrame(X)
        output_graph_nc = model.predict(data_frame)
        A = nx.adjacency_matrix(output_graph_nc).todense()
        A = np.asarray(A).astype(np.float64)

        return A, [-1], [], []

    def fit_mmhc(self, X):
        model = BNlearnMMHC(alpha=self.alphaCIT)

        data_frame = pd.DataFrame(X)
        output_graph_nc = model.create_graph_from_data(data_frame)
        A = nx.adjacency_matrix(output_graph_nc).todense()
        A = np.asarray(A).astype(np.float64)

        return A, [-1], [], []

    def fit_ges(self, X):
        import cdt
        model = cdt.causality.graph.GES()
        data_frame = pd.DataFrame(X)
        output_graph_nc = model.predict(data_frame)
        A = nx.adjacency_matrix(output_graph_nc).todense()
        A = np.asarray(A).astype(np.float64)

        return A, [-1], [], []

    def fit_all_L1(self, X):
        # Wstar, Z, tau, it, itMin, h\
        Wstar, h, iter, h_iter, alpha, rho= self.fit_aug_lagr_Aabs(X,
                                                    tau=self.tau,
                                                    hTol= self.h_tol,
                                                    hFactor=0.25,
                                                    rho= self.rho,
                                                    rhoFactor=10,
                                                    rhoMax=self.rho_max,
                                                    alwaysAccept=False,
                                                    T=100,
                                                    pre_hTol= self.pre_h_tol,
                                                    pre_use_l2 = self.pre_use_l2)

        if self.search != 2:
            Wstar[np.abs(Wstar) < self.threshold_A] = 0

        # return Wstar, [h], iter, h_iter, alpha, rho

        return Wstar, [h],  alpha, rho

    def fit_all_L1_search(self, X):
        # Run augmented Lagrangian algorithm
        A, h, iter, h_iter, alpha, rho = self.fit_aug_lagr_Aabs(X,
                                                                    tau=self.tau,
                                                                    hTol=self.h_tol,
                                                                    hFactor=0.25,
                                                                    rho=self.rho,
                                                                    rhoFactor=10,
                                                                    rhoMax=self.rho_max,
                                                                    alwaysAccept=False,
                                                                    T=100,
                                                                    pre_hTol=self.pre_h_tol,
                                                                    pre_use_l2=self.pre_use_l2)

        # W, h, t, h_iter, alpha_iter, rho_iter = \
        #     self.fit_aug_lagr_Aabs(X, tau=tau, hTol=hTol, hFactor=hFactor, rho=rho, rhoFactor=rhoFactor, rhoMax=rhoMax,
        #                            T=T,
        #                            pre_use_l2=pre_use_l2, pre_hTol=pre_hTol)
        Wstar, h, iter_search = local_search_given_W(X,
                                                     A,
                                                     Wtol=self.W_tol,
                                                     penTol=self.pen_tol,
                                                     tau=self.tau,
                                                     hTol=self.h_tol,
                                                     revEdges=self.revEdges,
                                                     noPen=self.noPen)

        # Wstar, h, itSearch = local_search_given_W(X, W, Wtol, penTol, tau, hTol, revEdges, noPen)

        # Wstar, h, iter, h_iter, alpha, rho, iter_search = self.fit_aug_lagr_Aabs_search(X,
        #                                                     tau=self.tau,
        #                                                     hTol=self.h_tol,
        #                                                     hFactor=0.25,
        #                                                     rho= self.rho,
        #                                                     rhoFactor=10,
        #                                                     rhoMax= self.rho_max,
        #                                                     Wtol= self.W_tol,
        #                                                     penTol=self.pen_tol,
        #                                                     pre_hTol= self.pre_h_tol,
        #                                                     pre_use_l2=self.pre_use_l2,
        #                                                     revEdges= self.revEdges,
        #                                                     noPen=self.noPen,
        #                                                     T=100)

        Wstar[np.abs(Wstar) < self.threshold_A] = 0

        # return Wstar, [h], iter, h_iter, alpha, rho, iter_search
        return Wstar, [h],  alpha, rho

    # def fit_all_new_search(self, X):
    #     A, h_iter, alpha, rho_iter = self.fit_all_new(X)
    #
    #     Wstar, h, itSearch = local_search_given_W(X,
    #                                               W=A,
    #                                               tau=self.tau,
    #                                               hTol=self.h_tol,
    #                                               Wtol=self.args.W_tol,
    #                                               penTol=self.args.pen_tol,
    #                                               revEdges=self.revEdges,
    #                                               noPen=False)
    #
    #     Wstar[np.abs(Wstar) < self.threshold_A] = 0
    #
    #     return Wstar, h_iter, alpha, rho_iter

    # def fit_all_search(self, X):
    #     A, h_iter, alpha, rho_iter = self.fit_all(X)
    #
    #     Wstar, h, itSearch = local_search_given_W(X,
    #                                               W=A,
    #                                               tau=self.tau,
    #                                               hTol=self.h_tol,
    #                                               Wtol=self.args.W_tol,
    #                                               penTol=self.args.pen_tol,
    #                                               revEdges=self.revEdges,
    #                                               noPen=False)
    #
    #     Wstar[np.abs(Wstar) < self.threshold_A] = 0
    #
    #     return Wstar, h_iter, alpha, rho_iter

    def fit_all_L2(self, X):
        # Wstar, Z, tau, it, itMin, h\
        Wstar, h, iter, h_iter, alpha, rho= self.fit_aug_lagr_AL2(X,
                                                                tau=self.tau,
                                                                hTol=self.h_tol,
                                                                rho=self.rho,
                                                                rhoMax=self.rho_max,
                                                                T=100,
                                                                threshold=self.threshold_A)

        if self.search != 2:
            Wstar[np.abs(Wstar) < self.threshold_A] = 0

        # return Wstar, [h], iter, h_iter, alpha, rho
        return Wstar, [h], alpha, rho

    def fit_all_L2_search(self, X):
        A, h, iter, h_iter, alpha, rho = self.fit_aug_lagr_AL2(X,
                                                               tau=self.tau,
                                                                hTol=self.h_tol,
                                                                rho=self.rho,
                                                                rhoMax=self.rho_max,
                                                                T=100,
                                                                threshold=self.threshold_A)

        A[np.abs(A) < self.threshold_A] = 0

        Wstar, h, iter_search = local_search_given_W(X,
                                                  A,
                                                  Wtol = self.W_tol,
                                                  penTol =  self.pen_tol,
                                                  tau = self.tau,
                                                  hTol = self.h_tol,
                                                  revEdges = self.revEdges,
                                                  noPen = self.noPen)

        Wstar[np.abs(Wstar) < self.threshold_A] = 0

        # return Wstar, [h], iter, h_iter, alpha, rho, iter_search
        return Wstar, [h], alpha, rho


    def search_given_A(self,X, A, initNoPen=False):
        Wstar, h, itSearch = local_search_given_W(X,
                                                  W=A,
                                                  tau=self.tau,
                                                  hTol=self.h_tol,
                                                  Wtol=self.args.W_tol,
                                                  penTol=self.args.pen_tol,
                                                  minimizeZ=self.minimizeZ,
                                                  revEdges=self.revEdges,
                                                  noPen=False,
                                                  initNoPen=initNoPen)

        Wstar[np.abs(Wstar) < self.threshold_A] = 0

        return Wstar, [h], itSearch

    def fit_indep_lars(self, X):
        """Learn Bayesian network from data X using zero-value constraints

        tau = uniform base penalty on all coefficients, if None then set by cross-validation
        minimizeZ = reduce set Z of zero-value constraints to a minimal one
        augmentZ = add "necessary" zeros to Z in each iteration
        """
        Wstar, Z, tau, it, itMin, h, activeStar, corrStar, A, pen, Gram = fit_zero_cons(X,
                                                     tau= self.tau,
                                                     minimizeZ=self.minimizeZ,
                                                     augmentZ=True, #self.augmentZ,
                                                     eps=self.h_tol,
                                                     iterMax=self.train_epochs,
                                                    revEdges=self.revEdges,
                                                    checkLARS=False) #self.checkLARS)

        Wstar[np.abs(Wstar) < self.threshold_A] = 0

        # return Wstar, [h], Z, tau, it, itMin

        return Wstar, [h], Z, tau

    def fit_aug_lagr_Aabs(self, X, tau=None, hTol=1e-10, hFactor=0.25, rho=0.1, rhoFactor=10,
                          rhoMax=1e20, alwaysAccept=False, T=100, pre_use_l2=True,
                          pre_hTol=1e-4):
        '''NOTEARS augmented Lagrangian algorithm with absolute value adjacency matrix'''

        def _loss(W):
            """Evaluate value and gradient of loss."""
            M = X @ W
            # loss_type = self.loss_type
            if loss_type == 'l2':
                R = X - M
                loss = 0.5 / X.shape[0] * (R ** 2).sum()
                G_loss = - 1.0 / X.shape[0] * X.T @ R
            elif loss_type == 'logistic':
                loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
                G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
            elif loss_type == 'poisson':
                S = np.exp(M)
                loss = 1.0 / X.shape[0] * (S - X * M).sum()
                G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
            else:
                raise ValueError('unknown loss type')
            return loss, G_loss

        def _wtoW(w):
            """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
            return (w[:d * d] - w[d * d:]).reshape([d, d])

        def _wtoA(w):
            """Convert doubled variables ([2 d^2] array) back to absolute value [d, d] matrix."""
            return (w[:d * d] + w[d * d:]).reshape([d, d])

        def _h(A):
            """Evaluate value and gradient of acyclicity constraint."""
            #     E = slin.expm(W * W)  # (Zheng et al. 2018)
            #     h = np.trace(E) - d
            M = np.eye(d) + A / d  # (Yu et al. 2019)
            G_h = np.linalg.matrix_power(M, d - 1).T
            h = (G_h * M).sum() - d
            # h = np.expm1(np.linalg.eigvals(W)).real.sum()  # TODO this is worse than above
            return h, G_h

        def _func(w):
            """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
            # d = w.shape[0]
            W = _wtoW(w)
            loss, G_loss = _loss(W)
            A = _wtoA(w)
            h, G_h = _h(A)
            obj = loss + 0.5 * rho * h * h + alpha * h + tau * w.sum()
            g_obj = np.concatenate((G_loss + (rho * h + alpha) * G_h + tau,
                                    -G_loss + (rho * h + alpha) * G_h + tau), axis=None)
            # g_obj = np.concatenate((G_loss + (rho * h + alpha) * G_h,
            #                         -G_loss + (rho * h + alpha) * G_h), axis=None)
            return obj, g_obj

        loss_type = 'l2'
        t = 0
        #    T = self.train_epochs
        h_iter = np.zeros(T + 1)
        alpha_iter = np.zeros(T + 1)
        rho_iter = np.zeros(T + 1)
        #    rho = self.rho
        alpha = 0
        # X = np.asfortranarray(X, dtype=np.float)

        n, d = X.shape
        bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]

        # ----inital with zero

        if not pre_use_l2:
            w_est = np.zeros(2 * d * d)
            h = 1
        # ----init with lars separate
        # Z = np.eye(d, dtype=bool)
        # Wstar, _ = init_Wstar(X, Z, tau)
        # w_est = np.concatenate((np.maximum(Wstar, 0), np.maximum(-Wstar, 0)), axis=None)
        # h = _h(np.abs(Wstar))[0]

        # ----init with L2 notear
        else:
            Wstar, _, _, _, _, _ = self.fit_aug_lagr_AL2(X, tau=tau,
                                                    hTol=pre_hTol, rho=rho, rhoMax=rhoMax, T=T,
                                                    threshold=0 if pre_use_l2 == 2 else 0.3)
            # Wstar = fit_aug_lagr_AL2projnew(X, tau=tau, hTol=1e-4, rho=rho, rhoMax=rhoMax, T=T, threshold=0.3)
            w_est = np.concatenate((np.maximum(Wstar, 0), np.maximum(-Wstar, 0)), axis=None)
            h = _h(np.abs(Wstar))[0]

        h_iter[0] = h
        # print(h)

        while t < T:
            w_new, h_new = None, None
            # while control for coefficient rho value
            while rho < rhoMax:

                # Penalty matrix is scaled version of gradient of constraint function
                sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
                w_new = sol.x
                A_new = _wtoA(w_new)
                h_new, _ = _h(A_new)
                h_iter[t + 1] = h_new
                # Evaluate constraint
                # h[t + 1] = self.eval_h(A_new)
                # h_new = h[t+1]
                # print((h_iter[t + 1], rho, np.sum(np.abs(A_new)), alpha))

                if h_new > hFactor * h:
                    rho *= rhoFactor
                    if alwaysAccept:
                        break
                else:
                    break

            # Update Lagrange multiplier
            #        print(np.allclose(w_new, w_est))
            #        print(h_new > hFactor * h)
            w_est, h = w_new, h_new

            alpha += rho * h

            alpha_iter[t + 1] = alpha
            rho_iter[t + 1] = rho

            if h_new <= hTol or rho >= rhoMax:  # NOTE: remove result worse results
                break

            t += 1

        W = _wtoW(w_est)

        #    W[np.abs(W) < self.threshold_A] = 0

        return W, h, t, h_iter[:t + 1], alpha_iter[:t + 1], rho_iter[:t + 1]

    # def fit_aug_lagr_Aabs_search(self, X, tau=None, hTol=1e-10, hFactor=0.25, rho=0.1, rhoFactor=10,
    #                              rhoMax=1e20, T=100, Wtol=1e-4, penTol=0, revEdges='alt-full',
    #                              noPen=False, pre_use_l2=True,
    #                              pre_hTol=1e-4):
    #     """NOTEARS augmented Lagrangian algorithm with absolute value adjacency matrix
    #     followed by local search over DAGs
    #     """
    #
    #     # Run augmented Lagrangian algorithm
    #     W, h, t, h_iter, alpha_iter, rho_iter = \
    #         self.fit_aug_lagr_Aabs(X, tau=tau, hTol=hTol, hFactor=hFactor, rho=rho, rhoFactor=rhoFactor, rhoMax=rhoMax, T=T,
    #                           pre_use_l2=pre_use_l2, pre_hTol=pre_hTol)
    #
    #     Wstar, h, itSearch = local_search_given_W(X, W, Wtol, penTol, tau, hTol, revEdges, noPen)
    #
    #     return Wstar, h, t, h_iter[:t + 1], alpha_iter[:t + 1], rho_iter[:t + 1], itSearch

    def fit_aug_lagr_AL2(self, X, tau=None, hTol=1e-10, rho=0.1, rhoMax=1e20, T=100, threshold=0.3):
        '''NOTEARS augmented Lagrangian algorithm with absolute value adjacency matrix'''

        # def _loss(W):
        #     """Evaluate value and gradient of loss."""
        #     M = X @ W
        #     if loss_type == 'l2':
        #         R = X - M
        #         loss = 0.5 / X.shape[0] * (R ** 2).sum()
        #         G_loss = - 1.0 / X.shape[0] * X.T @ R
        #     elif loss_type == 'logistic':
        #         loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
        #         G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        #     elif loss_type == 'poisson':
        #         S = np.exp(M)
        #         loss = 1.0 / X.shape[0] * (S - X * M).sum()
        #         G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        #     else:
        #         raise ValueError('unknown loss type')
        #     return loss, G_loss
        #
        def _h(W):
            """Evaluate value and gradient of acyclicity constraint."""
            #     E = slin.expm(W * W)  # (Zheng et al. 2018)
            #     h = np.trace(E) - d
            M = np.eye(d) + W * W / d  # (Yu et al. 2019)
            E = np.linalg.matrix_power(M, d - 1)
            h = (E.T * M).sum() - d
            G_h = E.T * W * 2
            return h, G_h

        #
        # def _wtoW( w):
        #     """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        #     return (w[:d * d] - w[d * d:]).reshape([d, d])
        #
        def _func(w):
            """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
            W = _wtoW(w)
            loss, G_loss = _loss(W)
            h, G_h = _h(W)
            obj = loss + 0.5 * rho * h * h + alpha * h + tau * w.sum()
            G_smooth = G_loss + (rho * h + alpha) * G_h
            g_obj = np.concatenate((G_smooth + tau, - G_smooth + tau), axis=None)
            return obj, g_obj

        # def _func(w):
        #     """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        #     # d = w.shape[0]
        #     W = _wtoW(w)
        #     loss, G_loss = _loss(W)
        #     # A = _wtoA(w)
        #     h, G_h = _h(W)
        #     obj = loss + 0.5 * rho * h * h + alpha * h + tau * w.sum()
        #     g_obj = np.concatenate((G_loss + (rho * h + alpha) * G_h + tau,
        #                             -G_loss - (rho * h + alpha) * G_h + tau), axis=None)
        #     return obj, g_obj

        def _loss(W):
            """Evaluate value and gradient of loss."""
            M = X @ W
            # loss_type = self.loss_type
            if loss_type == 'l2':
                R = X - M
                loss = 0.5 / X.shape[0] * (R ** 2).sum()
                G_loss = - 1.0 / X.shape[0] * X.T @ R
            elif loss_type == 'logistic':
                loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
                G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
            elif loss_type == 'poisson':
                S = np.exp(M)
                loss = 1.0 / X.shape[0] * (S - X * M).sum()
                G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
            else:
                raise ValueError('unknown loss type')
            return loss, G_loss

        def _wtoW(w):
            """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
            return (w[:d * d] - w[d * d:]).reshape([d, d])

        # def _wtoA(w):
        #     """Convert doubled variables ([2 d^2] array) back to absolute value [d, d] matrix."""
        #     # return (w[:d * d] + w[d * d:]).reshape([d, d])
        #     W = (w[:d * d] - w[d * d:]).reshape([d, d])
        #     return W * W

        # def _h(A):
        #     """Evaluate value and gradient of acyclicity constraint."""
        #     #     E = slin.expm(W * W)  # (Zheng et al. 2018)
        #     #     h = np.trace(E) - d
        #     M = np.eye(d) + A / d  # (Yu et al. 2019)
        #     G_h = np.linalg.matrix_power(M, d - 1).T
        #     h = (G_h * M).sum() - d
        #     # h = np.expm1(np.linalg.eigvals(W)).real.sum()  # TODO this is worse than above
        #     return h, G_h

        loss_type = 'l2'
        t = 0
        #    T = self.train_epochs
        h_iter = np.zeros(T + 1)
        alpha_iter = np.zeros(T + 1)
        rho_iter = np.zeros(T + 1)
        #    rho = self.rho
        alpha = 0
        # X = np.asfortranarray(X, dtype=np.float)

        n, d = X.shape
        bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]

        w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)

        while t < T:

            w_new, h_new = None, None
            while rho < rhoMax:
                sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
                w_new = sol.x
                h_new, _ = _h(_wtoW(w_new))
                h_iter[t + 1] = h_new
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break

                # print((h_iter[t + 1], rho, np.sum(np.abs(w_new))))

            w_est, h = w_new, h_new
            alpha += rho * h
            h_iter[t + 1] = h_new
            alpha_iter[t + 1] = alpha
            rho_iter[t + 1] = rho
            if h <= hTol or rho >= rhoMax:
                break

            t += 1

        W = _wtoW(w_est)

        # W[np.abs(W) < threshold] = 0

        return W, h, t, h_iter[:t + 1], alpha_iter[:t + 1], rho_iter[:t + 1]
