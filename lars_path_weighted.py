#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:48:03 2020

@author: denniswei
"""

import numpy as np
from scipy import linalg
from sklearn.linear_model import LinearRegression, LassoLars, LassoLarsCV
# from penLinRegr_sklearn import pen_regress
from eval_h import eval_h, eval_h_deri
from local_search_given_matrix import remove_edge, reverse_edge, restore_reverse


def lars_path_weighted(X, y, pen, tau=None, maxIter=1000, eps=1e-10):
    """Compute Lasso solution path using LARS algorithm
    Allow non-uniform penalties including zeros
    Go "backward" starting from smallest penalty

    pen = additional penalties on model coefficients, can be zero
    tau = uniform base penalty on all coefficients, if None then set by cross-validation
    """
    # Gram matrix
    n = X.shape[0]
    Gram = np.dot(X.T, X) / n

    # Initialize with solution for uniform penalty tau
    alpha = 0
    if tau is None:
        # Lasso with tau chosen by cross-validation
        reg = LassoLarsCV(max_iter=maxIter)
    elif tau:
        # Lasso with given tau
        reg = LassoLars(alpha=tau)
    else:
        # Assume tau = 0, linear regression
        reg = LinearRegression()
    reg.fit(X, y)
    # Coefficients
    w = reg.coef_
    if tau is None:
        # tau chosen by CV
        tau = reg.alpha_
    # Active set
    active = np.abs(w) > eps
    # Correlations with residual
    corr = np.dot(y, X) / n - np.dot(Gram, w)

    # Iterate
    it = 0
    while it < maxIter:
        # LARS direction
        d = linalg.solve(Gram[np.ix_(active, active)],
                         np.sign(w[active] + corr[active]) * pen[active],
                         assume_a='pos')
        # Increments to correlations
        a = np.dot(Gram[:, active], d)
        # Bounds on step size
        gamma = np.empty_like(w)
        gamma[active] = w[active] / d
        gamma[~active] = (tau + alpha * pen[~active] - np.sign(a[~active]) * corr[~active]) \
                         / (np.abs(a[~active]) - pen[~active])
        gamma[gamma < eps] = np.inf
        # Step size and limiting index
        j = gamma.argmin()
        gamma = gamma[j]
        # Update alpha, coefficients, correlations
        alpha += gamma
        w[active] -= gamma * d
        corr += gamma * a
        # Update active set
        active[j] = ~active[j]

        it += 1

    return w


def lars_path_matrix_single(Wstar, Z, activeStar, corrStar, chol, pen, tauMat, Gram, Wtol=1e-10, penTol=0,
                            checkLARS=False, X=None):
    """Follow weighted Lasso path using LARS algorithm until one matrix element set to zero
    """

    d = Wstar.shape[1]
    # Compute LARS directions column by column
    dW = np.zeros_like(Wstar)
    for j in range(d):
        if activeStar[:, j].any():
            dW[activeStar[:, j], j] = linalg.cho_solve(chol[j],
                                                       np.sign(
                                                           Wstar[activeStar[:, j], j] + corrStar[activeStar[:, j], j]) *
                                                       pen[activeStar[:, j], j])
    # Increments to correlations
    a = np.dot(Gram, dW)

    # Bounds on step size
    gamma = np.full_like(Wstar, np.inf)
    ind = activeStar & (np.abs(dW) > 0)  # avoid divide-by-zero warning but infinities are actually correct
    gamma[ind] = Wstar[ind] / dW[ind]
    gamma[gamma < 0] = np.inf
    ind = ~activeStar & ~Z & (np.abs(a) > pen)
    gamma[ind] = (tauMat[ind] - np.sign(a[ind]) * corrStar[ind]) \
                 / (np.abs(a[ind]) - pen[ind])
    if np.isinf(gamma).all():
        print('WARNING: gamma is all np.inf!')
        print('activeStar =')
        print(activeStar)
        print('Wstar =')
        print(Wstar)
        print('dW =')
        print(dW)
        print('pen =')
        print(pen)
        print('corrStar =')
        print(corrStar)
        print('a =')
        print(a)
    # Limiting index
    (i, j) = np.unravel_index(gamma.argmin(), gamma.shape)

    # The following should happen only rarely
    if ~(activeStar[i, j] & (pen[i, j] > penTol)):
        # Instantiate penalized regression quantities
        W = Wstar.copy()
        W[np.abs(W) < Wtol] = 0
        active = activeStar.copy()
        corr = corrStar.copy()
        alpha = 0

        # The following should happen only rarely
        while ~(activeStar[i, j] & (pen[i, j] > penTol)) & (pen[active] > penTol).any():
            # Update penalized regression quantities
            gammaMin = gamma[i, j]
            W[active] -= gammaMin * dW[active]
            W[np.abs(W) < Wtol] = 0
            active[i, j] = ~active[i, j]
            corr += gammaMin * a
            alpha += gammaMin
            gamma -= gammaMin

            if checkLARS:
                # Check LARS solution
                reg = LassoLars(normalize=False)
                Wcheck = np.zeros_like(W)
                for jj in range(d):
                    if (~Z[:, jj]).any():
                        Wcheck[~Z[:, jj], jj] = pen_regress(reg, X[:, ~Z[:, jj]], X[:, jj], 1,
                                                            tauMat[~Z[:, jj], jj] + alpha * pen[~Z[:, jj], jj])
                assert np.allclose(Wcheck, W)

            # Update LARS direction and correlation increments for column j
            dW[:, j] = 0
            dW[active[:, j], j] = linalg.solve(Gram[np.ix_(active[:, j], active[:, j])],
                                               np.sign(W[active[:, j], j] + corr[active[:, j], j]) * pen[
                                                   active[:, j], j],
                                               assume_a='pos')
            a[:, j] = np.dot(Gram[:, active[:, j]], dW[active[:, j], j])

            # Update bounds on step size from column j
            gamma[:, j] = np.inf
            ind = active[:, j] & (np.abs(dW[:, j]) > 0)
            gamma[ind, j] = W[ind, j] / dW[ind, j]
            gamma[gamma < 0] = np.inf
            if active[i, j]:
                # Special treatment for previously inactive (i,j)
                gamma[i, j] = np.inf
            ind = ~active[:, j] & ~Z[:, j] & (np.abs(a[:, j]) > pen[:, j])
            gamma[ind, j] = (tauMat[ind, j] + alpha * pen[ind, j] - np.sign(a[ind, j]) * corr[ind, j]) \
                            / (np.abs(a[ind, j]) - pen[ind, j])
            if ~active[i, j]:
                # Special treatment for previously active (i,j)
                if np.abs(a[i, j]) > pen[i, j]:
                    if np.sign(a[i, j] * corr[i, j]) == 1:
                        gamma[i, j] = 0
                    else:
                        gamma[i, j] = 2 * (tauMat[i, j] + alpha * pen[i, j]) / (np.abs(a[i, j]) - pen[i, j])
                else:
                    gamma[i, j] = np.inf
            if np.isinf(gamma).all():
                print('WARNING: gamma is all np.inf!')
                print('active =')
                print(active)
                print('W =')
                print(W)
                print('dW =')
                print(dW)
                print('pen =')
                print(pen)
                print('corr =')
                print(corr)
                print('a =')
                print(a)

            # Limiting index
            (i, j) = np.unravel_index(gamma.argmin(), gamma.shape)

    return i, j


def init_Wstar(X, Z, tau=None, eps=1e-10):
    """Initialize columns of W with Lasso solutions for uniform penalty tau
    Choose tau by cross-validation if none provided
    """
    # Initialize
    d = X.shape[1]
    W = np.zeros((d, d))

    if tau is None:
        # Lasso with tau chosen by cross-validation
        reg = LassoLarsCV(normalize=False, cv=5)
    elif tau:
        # Lasso with given tau
        reg = LassoLars(alpha=tau, normalize=False)
    else:
        # Assume tau = 0, linear regression
        reg = LinearRegression()

        # Iterate over columns
    tau = np.full(d, tau)
    for j in range(d):
        # Selection mask
        reg.fit(X[:, ~Z[:, j]], X[:, j])
        W[~Z[:, j], j] = reg.coef_
        if tau[j] is None:
            tau[j] = reg.alpha_

    return W, tau.astype(float)


def fit_zero_cons(X, tau=None, minimizeZ=True, augmentZ=True, revEdges='both', noPen=False, eps=1e-10, iterMax=1e4,
                  checkLARS=False):
    """Learn Bayesian network from data X using zero-value constraints

    tau = uniform base penalty on all coefficients, if None then set by cross-validation
    minimizeZ = reduce set Z of zero-value constraints to a minimal one
    augmentZ = add "necessary" zeros to Z in each iteration
    """

    (n, d) = X.shape
    # Subtract mean from X
    X -= X.mean(axis=0)
    # Gram matrix
    Gram = np.dot(X.T, X) / n

    # Initial set Z of zero-value constraints
    Z = np.eye(d, dtype=bool)
    # Initialize W with Lasso solution for uniform penalty tau
    Wstar, tau = init_Wstar(X, Z, tau, eps)
    #    if tau.any():
    #        reg = LassoLars(normalize=False)
    #    else:
    #        reg = LinearRegression()
    tauMat = np.tile(tau, [d, 1])
    # Adjacency matrix
    A = np.abs(Wstar)
    # Evaluate constraint
    h = eval_h(A)
    # Active set
    activeStar = A > eps
    # Penalty matrix
    pen = eval_h_deri(A)  # linalg.expm(A).T
    # Add "necessary" zeros to Z
    if augmentZ:
        Z |= (~activeStar & (pen > eps))

    # Iterate while infeasible
    it = 0
    j = None
    while (h > eps) & (it < iterMax):
        # Infeasible, perform first iteration of penalized regression
        if j is None:
            # Compute entire gradient matrix i.e. correlations with residuals
            corrStar = Gram - np.dot(Gram, Wstar)
            # Cholesky factorizations of all Gram submatrices
            chol = {}
            for j in range(d):
                chol[j] = linalg.cho_factor(Gram[np.ix_(activeStar[:, j], activeStar[:, j])])

        # Determine edge (i,j) to remove
        i, j = lars_path_matrix_single(Wstar, Z, activeStar, corrStar, chol, pen,
                                       tauMat, Gram, Wtol=eps, penTol=eps, checkLARS=checkLARS, X=X)

        # Add (i,j) to Z, re-optimize and update
        Wstar, Z, activeStar, corrStar, chol, A, h, pen = \
            remove_edge(Wstar, Z, i, j, activeStar, corrStar, chol, A, tau, Gram, augmentZ, checkLARS=checkLARS, X=X)

        if revEdges:  # TODO HERE check
            # Candidate edges for reversal based on marginal change to constraint and loss
            rev = activeStar & ~activeStar.T & (pen.T - A - pen < eps) & (np.abs(corrStar.T) > tau + eps)
            if rev.any():
                iRev, jRev = rev.nonzero()
                # Sort by marginal change if more than one candidate
                if len(iRev) > 1:
                    idx = np.argsort((pen.T - A - pen - np.abs(corrStar.T) + tau)[rev])
                    iRev, jRev = iRev[idx], jRev[idx]
                for idx in range(len(iRev)):
                    # Try reversing edge
                    Wstar, Z, activeStar, corrStar, chol, A, h, pen = \
                        reverse_edge(Wstar, Z, iRev[idx], jRev[idx], activeStar,
                                     corrStar, chol, A, h, pen, tau, Gram, revEdges, eps, checkLARS=checkLARS, X=X)

        it += 1

    # Make Z minimal by eliminating unnecessary zero-value constraints
    Wstar, Z, activeStar, corrStar, A, h, pen, itMin = \
        restore_reverse(Wstar, Z, activeStar, corrStar, A, h, pen, tau, Gram,
                        minimizeZ=minimizeZ, revEdges=revEdges, noPen=noPen, checkLARS=checkLARS, X=X)
    #    itMin = 0
    #    if minimizeZ:
    ##        Wstar, Z, itMin = minimize_Z(Wstar, Z, pen, corrStar, j, Gram, X, tau, eps)
    #        # Unnecessary zero-value constraints
    #        unnec = Z & (pen < eps)
    #        while unnec.any():
    #            # Choose unnecessary constraint with largest gradient
    #            (i, j) = np.unravel_index(np.abs(unnec * corrStar).argmax(), unnec.shape)
    #            # Restore edge (i,j)
    #            Wstar, Z, activeStar, corrStar, chol, A, pen = \
    #                restore_edge(Wstar, Z, i, j, activeStar, corrStar, chol, A, tau, Gram, eps, checkLARS=checkLARS, X=X)
    #
    #            if revEdges: # TODO HERE check
    #                # Candidate edges for reversal based on marginal change to constraint and loss
    #                rev = activeStar & ~activeStar.T & (pen.T - A - pen < eps) & (np.abs(corrStar.T) > tau + eps)
    #                if rev.any():
    #                    iRev, jRev = rev.nonzero()
    #                    # Sort by marginal change if more than one candidate
    #                    if len(iRev) > 1:
    #                        idx = np.argsort((pen.T - A - pen - np.abs(corrStar.T) + tau)[rev])
    #                        iRev, jRev = iRev[idx], jRev[idx]
    #                    for idx in range(len(iRev)):
    #                        # Try reversing edge
    #                        Wstar, Z, activeStar, corrStar, chol, A, h, pen = \
    #                            reverse_edge(Wstar, Z, iRev[idx], jRev[idx], activeStar,
    #                                         corrStar, chol, A, h, pen, tau, Gram, revEdges, eps, checkLARS=checkLARS, X=X)
    #
    #            # Unnecessary zero-value constraints
    #            unnec = Z & (pen < eps)
    #
    #            itMin += 1

    return Wstar, Z, tau, it, itMin, h, activeStar, corrStar, A, pen, Gram


def minimize_Z(Wstar, Z, pen, corr, j, Gram, X, tau, eps=1e-10):
    """Given a feasible Wstar, make Z minimal by eliminating unnecessary zero-value constraints
    """

    # Unnecessary zero-value constraints
    unnec = Z & (pen < eps)
    it = 0
    while unnec.any():
        if j is None:
            # Compute entire gradient matrix
            corr = Gram - np.dot(Gram, Wstar)
        else:
            # Update gradient for column j
            corr[:, j] = Gram[:, j] - np.dot(Gram[:, ~Z[:, j]], Wstar[~Z[:, j], j])

        #        # Choose column with largest gradient norm over unnecessary components
        #        j = linalg.norm(unnec * corr, ord=2, axis=0).argmax()
        # Choose unnecessary constraint with largest gradient
        (i, j) = np.unravel_index(np.abs(unnec * corr).argmax(), unnec.shape)
        # Remove constraints
        #        Z[unnec[:,j], j] = False
        Z[i, j] = False
        #        # Convert row index i to index of non-Z elements
        #        j0 = ((~Z[:, j]).nonzero()[0] == i).nonzero()[0][0]
        # Re-optimize column
        reg = LassoLars(alpha=tau[j], normalize=False)
        reg.fit(X[:, ~Z[:, j]], X[:, j])
        Wstar[~Z[:, j], j] = reg.coef_

        # Adjacency matrix
        A = np.abs(Wstar)
        # Penalty matrix
        pen = eval_h_deri(A)  # linalg.expm(A).T
        # Unnecessary zero-value constrains
        unnec = Z & (pen < eps)

        it += 1

    return Wstar, Z, it

# def fit_zero_cons_no_relin(X, tau=None, eps=1e-10, iterMax=1e4):
#     (n, d) = X.shape
#     # Gram matrix
#     Gram = np.dot(X.T, X) / n
#
#     # Initial set Z of zero-value constraints
#     Z = np.eye(d, dtype=bool)
#     # Initialize W with Lasso solution for uniform penalty tau
#     Wstar, tau = init_Wstar(X, Z, tau, eps)
#     if tau.any():
#         reg = LassoLars(normalize=False)
#     else:
#         reg = LinearRegression()
#     tauMat = np.tile(tau, [d, 1])
#     # Adjacency matrix
#     A = np.abs(Wstar)
#     # Evaluate constraint
#     h = eval_h(A)
#     # Active set
#     activeStar = A > eps
#     # Penalty matrix
#     penStar = eval_h_deri(A)  # linalg.expm(A).T
#     # Add "necessary" zeros to Z
#     Z |= (~activeStar & (penStar > eps))
#
#     # First iteration
#     it = 0
#     if h > eps:
#         # Infeasible, initialize penalized regression quantities
#         alpha = 0
#         W = Wstar.copy()
#         active = activeStar.copy()
#         pen = penStar.copy()
#
#         # Compute entire gradient matrix i.e. correlations with residuals
#         corr = Gram - np.dot(Gram, Wstar)
#         # LARS directions
#         dW = np.zeros_like(W)
#         for j in range(d):
#             dW[active[:, j], j] = linalg.solve(Gram[np.ix_(active[:, j], active[:, j])],
#                                                np.sign(W[active[:, j], j] + corr[active[:, j], j]) * pen[
#                                                    active[:, j], j],
#                                                assume_a='pos')
#         # Increments to correlations
#         a = np.dot(Gram, dW)
#
#         # Bounds on step size
#         gamma = np.full_like(W, np.inf)
#         ind = active & (np.abs(dW) > eps)  # avoid divide-by-zero warning but infinities are actually correct
#         gamma[ind] = W[ind] / dW[ind]
#         ind = ~active & ~Z
#         gamma[ind] = (tauMat[ind] + alpha * pen[ind] - np.sign(a[ind]) * corr[ind]) \
#                      / (np.abs(a[ind]) - pen[ind])
#         gamma[gamma < eps] = np.inf
#         # Limiting index
#         (i, j) = np.unravel_index(gamma.argmin(), gamma.shape)
#
#         if activeStar[i, j]:
#             # Active coefficient in Wstar being set to zero, add to Z
#             Z[i, j] = True
#             # Update column j of Wstar
#             Wstar[:, j] = 0
#             if (~Z[:, j]).any():
#                 if type(reg) == LassoLars:
#                     reg.alpha = tau[j]
#                 reg.fit(X[:, ~Z[:, j]], X[:, j])
#                 Wstar[~Z[:, j], j] = reg.coef_
#             # Adjacency matrix
#             A[:, j] = np.abs(Wstar[:, j])
#             # Evaluate constraint
#             h = eval_h(A)
#             # Active set
#             activeStar[:, j] = A[:, j] > eps
#             # Penalty matrix
#             penStar = eval_h_deri(A)  # linalg.expm(A).T
#             # Add "necessary" zeros to Z
#             Z |= (~activeStar & (penStar > eps))
#
#         it += 1
#
#     # Iterate while infeasible
#     while (h > eps) & (it < iterMax):
#
#         # Update penalized regression quantities
#         gammaMin = gamma[i, j]
#         W[active] -= gammaMin * dW[active]
#         corr += gammaMin * a
#         active[i, j] = ~active[i, j]
#         alpha += gammaMin
#         gamma -= gammaMin
#
#         # Update LARS direction and correlation increments for column j
#         dW[:, j] = 0
#         dW[active[:, j], j] = linalg.solve(Gram[np.ix_(active[:, j], active[:, j])],
#                                            np.sign(W[active[:, j], j] + corr[active[:, j], j]) * pen[active[:, j], j],
#                                            assume_a='pos')
#         a[:, j] = np.dot(Gram[:, active[:, j]], dW[active[:, j], j])
#
#         # Update bounds on step size from column j
#         ind = active[:, j] & (np.abs(dW[:, j]) > eps)
#         gamma[ind, j] = W[ind, j] / dW[ind, j]
#         ind = ~active[:, j] & ~Z[:, j]
#         gamma[ind, j] = (tauMat[ind, j] + alpha * pen[ind, j] - np.sign(a[ind, j]) * corr[ind, j]) \
#                         / (np.abs(a[ind, j]) - pen[ind, j])
#         gamma[gamma < eps] = np.inf
#
#         # Limiting index
#         (i, j) = np.unravel_index(gamma.argmin(), gamma.shape)
#
#         if activeStar[i, j]:
#             # Active coefficient in Wstar being set to zero, add to Z
#             Z[i, j] = True
#             # Update column j of Wstar
#             Wstar[:, j] = 0
#             if (~Z[:, j]).any():
#                 if type(reg) == LassoLars:
#                     reg.alpha = tau[j]
#                 reg.fit(X[:, ~Z[:, j]], X[:, j])
#                 Wstar[~Z[:, j], j] = reg.coef_
#             # Adjacency matrix
#             A[:, j] = np.abs(Wstar[:, j])
#             # Evaluate constraint
#             h = eval_h(A)
#             # Active set
#             activeStar[:, j] = A[:, j] > eps
#             # Penalty matrix
#             penStar = eval_h_deri(A)  # linalg.expm(A).T
#             # Add "necessary" zeros to Z
#             Z |= (~activeStar & (penStar > eps))
#
#         it += 1
#
#     # Make Z minimal by eliminating unnecessary zero-value constraints
#     Wstar, Z, it = minimize_Z(Wstar, Z, penStar, corr, None, Gram, X, tau, eps)
#
#     return Wstar, Z, tau

def pen_regress(reg, X, y, p, pen):
    # ell_p-penalized regression of y on X
    # Factors for scaling weights and columns of X
    if p == 1:
        scale = pen
    elif p == 2:
        scale = np.sqrt(pen)

    # Scale columns of X to account for unequal penalties
    if not scale.all() == 0:
        Xscaled = X / scale  # TODO solving this diviing zero, experimental mores, check rho
    else:
        scale += 1e-6
        Xscaled = X/scale
        # print(np.isinf(Xscaled).any())
    # Fit model
    reg.fit(Xscaled, y)
    # Re-scale coefficients before returning
    return reg.coef_ / scale