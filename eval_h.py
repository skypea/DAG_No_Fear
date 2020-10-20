#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:33:36 2020

@author: denniswei
"""

import numpy as np

# from scipy.linalg import expm
# from sklearn.linear_model import LinearRegression, Lasso, Ridge

def eval_h(A):
    # Compute tr(exp(A)) - d
    # print(A)
    # h = np.expm1(np.linalg.eigvals(A)).real.sum() # TODO  worse than below

    # polynomial h
    d = A.shape[0]
    M = np.eye(d) + np.abs(A) / d  # (Yu et al. 2019)
    E = np.linalg.matrix_power(M, d - 1)
    h = (E.T * M).sum() - d

    # new h
    # h= np.expm1(np.linalg.eigvals(A)).real.sum()

    return h


def eval_h_deri( A):
    # Compute tr(exp(A)) - d
    # print(A)
    # h = np.expm1(np.linalg.eigvals(A)).real.sum() # TODO  worse than below

    # polynomial h
    d = A.shape[0]
    M = np.eye(d) + np.abs(A) / d  # (Yu et al. 2019)
    E = np.linalg.matrix_power(M, d - 1)
    E = E.T

    # new derivative
    # E = linalg.expm(A).T

    return E

def eval_grad_h(A):
    """Evaluate h(A) and gradient"""
    d = A.shape[0]
    M = np.eye(d) + A / d
    grad = np.linalg.matrix_power(M, d - 1).T
    h = (grad * M).sum() - d

    return h, grad

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