import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from sklearn.linear_model import LogisticRegression, LinearRegression
from math import log
from causallib.estimation import IPW


def check_input(A, Y, X):
    if not isinstance(A, pd.Series):
        if not np.max(A.shape) == A.size:
            raise Exception(f'A must be one dimensional, got shape {A.shape}')
        A = pd.Series(A.flatten())
    if not isinstance(Y, pd.Series):
        if not np.max(A.shape) == A.size:
            raise Exception(f'A must be one dimensional, got shape {A.shape}')
        Y = pd.Series(Y.flatten())
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not len(A.index) == len(Y.index) == len(X.index):
        raise Exception(f'A, Y, X must have same number of samples, '
                        f'got A: {len(A.index)} samples, Y: {len(Y.index)} samples, X: {len(X.index)} samples')
    return A, Y, X


def calc_ate_vanilla_ipw(A, Y, X, return_selection=False):
    """Estimate ATE using IPW with L1-penalized logistic regression.
    If return_selection=True, also return a binary vector indicating selected covariates."""
    logit = LogisticRegression(solver='liblinear', penalty='l1', C=1e2, max_iter=500)
    ipw = IPW(logit, use_stabilized=True).fit(X, A)
    weights = ipw.compute_weights(X, A)
    outcomes = ipw.estimate_population_outcome(X, A, Y, w=weights)
    effect = ipw.estimate_effect(outcomes[1], outcomes[0])

    if return_selection:
        selected_mask = (np.abs(logit.coef_.flatten()) > 1e-8).astype(int)
        return effect[0], selected_mask
    else:
        return effect[0]



def calc_group_diff(X, idx_trt, ipw, l_norm):
    """Utility function to calculate the difference in covariates between treatment and control groups"""
    return (np.abs(np.average(X[idx_trt], weights=ipw[idx_trt], axis=0) -
                   np.average(X[~idx_trt], weights=ipw[~idx_trt], axis=0)))**l_norm


def calc_wamd(A, X, ipw, x_coefs, l_norm=1):
    """Utility function to calculate the weighted absolute mean difference"""
    idx_trt = A == 1
    return calc_group_diff(X.values, idx_trt.values, ipw.values, l_norm).dot(np.abs(x_coefs))


def calc_outcome_adaptive_lasso_single_lambda(A, Y, X, Lambda, gamma_convergence_factor):
    """Calculate ATE with the outcome adaptive lasso"""
    n = A.shape[0]  # number of samples
    # extract gamma according to Lambda and gamma_convergence_factor
    gamma = 2 * (1 + gamma_convergence_factor - log(Lambda, n))
    
    # fit # Outcome model
    XA = X.merge(A.to_frame(), left_index=True, right_index=True)
    lr = LinearRegression(fit_intercept=True).fit(XA, Y)
    # extract the coefficients of the covariates
    x_coefs = lr.coef_.flatten()[1:]
    # calculate outcome adaptive penalization weights
    weights = (np.abs(x_coefs)) ** (-1 * gamma)
    # apply the penalization to the covariates themselves
    X_w = X / weights
    
    # fit logistic propensity score model from penalized covariates to the exposure
    logit = LogisticRegression(solver='liblinear', penalty='l1', C=1/Lambda)
    ipw = IPW(logit, use_stabilized=False).fit(X_w, A)
    # Selection indicator (nonzero coefficients)
    selected_mask = (np.abs(logit.coef_.flatten()) > 1e-8).astype(int)
    # compute inverse propensity weighting and calculate ATE
    weights = ipw.compute_weights(X_w, A)
    outcomes = ipw.estimate_population_outcome(X_w, A, Y, w=weights)
    effect = ipw.estimate_effect(outcomes[1], outcomes[0])
    return effect, x_coefs, weights, selected_mask


def calc_outcome_adaptive_lasso(
    A, Y, X, gamma_convergence_factor=2, log_lambdas=None, 
    plot=True, save_path=None
):
    """Calculate estimate of average treatment effect using the outcome adaptive LASSO (Shortreed and Ertefaie, 2017).
    Optionally save the AMD vs log(lambda) figure.
    """
    A, Y, X = check_input(A, Y, X)

    if log_lambdas is None:
        log_lambdas = [-10, -5, -2, -1, -0.75, -0.5, -0.25, 0.25, 0.49]

    n = A.shape[0]
    lambdas = n ** np.array(log_lambdas)
    amd_vec = np.zeros(len(lambdas))
    ate_vec = np.zeros(len(lambdas))
    selected_masks = []

    # Calculate ATE for each lambda
    for il, Lambda in enumerate(lambdas):
        ate_vec[il], x_coefs, ipw, selected_mask = calc_outcome_adaptive_lasso_single_lambda(
            A, Y, X, Lambda, gamma_convergence_factor
        )
        amd_vec[il] = calc_wamd(A, X, ipw, x_coefs)
        selected_masks.append(selected_mask)

    best_idx = np.argmin(amd_vec)
    best_ate = ate_vec[best_idx]
    best_selected_mask = selected_masks[best_idx]

    # Plot and save if requested
    if plot:
        plt.figure(figsize=(7, 4))
        plt.plot(log_lambdas, amd_vec, marker='o', color='steelblue')
        plt.axvline(log_lambdas[best_idx], color='r', linestyle='--', label='Min wAMD')
        plt.title("Weighted Absolute Mean Difference vs log(Lambda)")
        plt.xlabel("log(Lambda)")
        plt.ylabel("Weighted Absolute Mean Difference (wAMD)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Determine save path
        if save_path is None:
            save_path = os.path.join(os.getcwd(), "wamd_vs_loglambda.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Figure saved to: {save_path}")

    return best_ate, amd_vec, ate_vec, best_selected_mask