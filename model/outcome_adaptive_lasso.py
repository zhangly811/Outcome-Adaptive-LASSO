import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV, LassoCV
from math import log
from causallib.estimation import IPW
import pdb

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
    
def calc_amd_per_covariate(X, A, ipw, l_norm=1):
    """Utility function to calculate the difference in covariates between treatment and control groups"""
    idx_trt = A == 1
    return (np.abs(np.average(X[idx_trt], weights=ipw[idx_trt], axis=0) -
                   np.average(X[~idx_trt], weights=ipw[~idx_trt], axis=0)))**l_norm

def calc_wamd_per_covariate(X, A, ipw, betas_hat, l_norm=1):
    """Utility function to calculate the difference in covariates between treatment and control groups"""
    idx_trt = A == 1
    return (np.abs(np.average(X[idx_trt], weights=ipw[idx_trt], axis=0) -
                   np.average(X[~idx_trt], weights=ipw[~idx_trt], axis=0)))**l_norm * np.abs(betas_hat)
    
def calc_amd(X, A, ipw, l_norm=1):
    return np.sum(calc_amd_per_covariate(X, A, ipw, l_norm))

def calc_wamd(X, A, ipw, betas_hat, l_norm=1):
    return np.sum(calc_wamd_per_covariate(X, A, ipw, betas_hat, l_norm))

def fit_outcome_model(A, Y, X, model_type='ridge', save_dir=None):
    """Fit ridge regression outcome model and return estimated coefficients."""
    # Combine covariates and treatment
    XA = X.merge(A.to_frame(), left_index=True, right_index=True)
    
    # Fit ridge regression with cross-validation to find optimal alpha
    alphas = np.logspace(-4, 1, 10)
        # --- Fit model with cross-validation ---
    if model_type == 'ridge':
        model_cv = RidgeCV(alphas=alphas, store_cv_values=True)
        model_cv.fit(XA, Y)
        mean_cv_mse = np.mean(model_cv.cv_values_, axis=0)
        alpha_scores = pd.DataFrame({
            "alpha": alphas,
            "mean_cv_mse": mean_cv_mse
        })
        chosen_alpha = model_cv.alpha_
        ylabel = "Mean CV MSE"
        title = "RidgeCV: Alpha vs Mean Cross-Validation MSE"

    elif model_type == 'lasso':
        model_cv = LassoCV(alphas=alphas, max_iter=1000)
        model_cv.fit(XA, Y)
        alpha_scores = pd.DataFrame({
            "alpha": model_cv.alphas_,
            "mean_cv_mse": model_cv.mse_path_.mean(axis=1)
        })
        chosen_alpha = model_cv.alpha_
        ylabel = "Mean CV MSE"
        title = "LassoCV: Alpha vs Mean Cross-Validation MSE"

    else:
        raise ValueError("model_type must be 'ridge' or 'lasso'")

    # --- Save alpha scores ---
    score_path = os.path.join(save_dir, f"alpha_scores_{model_type}.csv")
    alpha_scores.to_csv(score_path, index=False)

    # --- Plot alpha vs CV scores ---
    plt.figure(figsize=(7, 5))
    sns.lineplot(x="alpha", y="mean_cv_mse", data=alpha_scores, marker="o")
    plt.xscale("log")
    plt.xlabel("Alpha (log scale)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)

    # mark chosen alpha
    plt.axvline(x=chosen_alpha, color="red", linestyle="--", alpha=0.7, label=f"Chosen α={chosen_alpha:.3g}")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(save_dir, f"{model_type}_alpha_cv_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    
    # Extract coefficients of the covariates (exclude treatment coefficient)
    betas_hat = model_cv.coef_.flatten()[1:]
    
    return betas_hat

def calc_outcome_adaptive_lasso_single_lambda(A, Y, X, betas_hat, Lambda, gamma_convergence_factor):
    """Calculate ATE with the outcome adaptive lasso"""
    n = A.shape[0]  # number of samples
    # extract gamma according to Lambda and gamma_convergence_factor
    gamma = 2 * (1 + gamma_convergence_factor - log(Lambda, n))
    
    # calculate outcome adaptive penalization weights
    weights = (np.abs(betas_hat)) ** (-1 * gamma)
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
    return effect, betas_hat, weights, selected_mask


def calc_outcome_adaptive_lasso(
    A, Y, X, rep, gamma_convergence_factor=2, log_lambdas=None, 
    plot=True, amd_save_path=None, wamd_save_path=None, base_dir=None):
    """Calculate estimate of average treatment effect using the outcome adaptive LASSO (Shortreed and Ertefaie, 2017).
    Optionally save the AMD vs log(lambda) figure.
    """
    # Determine save path
    if amd_save_path is None:
        amd_save_path = os.path.join(os.getcwd())
    if wamd_save_path is None:
        wamd_save_path = os.path.join(os.getcwd())
        
    A, Y, X = check_input(A, Y, X)

    if log_lambdas is None:
        log_lambdas = [-10, -5, -2, -1, -0.75, -0.5, -0.25, 0.25, 0.49]

    n = A.shape[0]
    lambdas = n ** np.array(log_lambdas)
    amd_vec = np.zeros(len(lambdas))
    wamd_vec = np.zeros(len(lambdas))
    ate_vec = np.zeros(len(lambdas))
    selected_masks = []

    # Calculate ATE for each lambda
    # Pre-fit outcome model to get betas_hat
    outcome_model_save_path = os.path.join(base_dir, "outcome_model", f"rep{rep}")
    os.makedirs(outcome_model_save_path, exist_ok=True)
    betas_hat = fit_outcome_model(A, Y, X, model_type='lasso', save_dir=outcome_model_save_path)
    
    for il, Lambda in enumerate(lambdas):
        ate_vec[il], betas_hat, ipw, selected_mask = calc_outcome_adaptive_lasso_single_lambda(
            A, Y, X, betas_hat, Lambda, gamma_convergence_factor
        )
        amd_after_per_covariate = calc_amd_per_covariate(X.values, A.values, ipw.values)
        wamd_after_per_covariate = calc_wamd_per_covariate(X.values, A.values, ipw.values, betas_hat)
        amd_after = np.sum(amd_after_per_covariate)
        wamd_after = np.sum(wamd_after_per_covariate)
        # save calculated_group_diff
        # [WRITE CODE TO SAVE amd_per_covariate AND wamd_per_covariate for each lambda]
        amd_vec[il] = amd_after
        wamd_vec[il] = wamd_after
        selected_masks.append(selected_mask)
        
        # save amd_after_per_covariate and wamd_after_per_covariate for each lambda
        amd_after_df = pd.DataFrame({
            "covariate_index": [col for col in X.columns],
            "amd_after": amd_after_per_covariate
        })
        amd_after_df.to_csv(os.path.join(amd_save_path, f"amd_after_per_covariate_rep{rep}_lambda{il}.csv"), index=False)
        wamd_after_df = pd.DataFrame({
            "covariate_index": [col for col in X.columns],
            "wamd_after": wamd_after_per_covariate
        })
        wamd_after_df.to_csv(os.path.join(wamd_save_path, f"wamd_after_per_covariate_rep{rep}_lambda{il}.csv"), index=False)

    best_idx = np.argmin(wamd_vec)
    best_ate = ate_vec[best_idx]
    best_selected_mask = selected_masks[best_idx]
    best_betas_hat = betas_hat

    # Plot and save if requested
    if plot:
        plt.figure(figsize=(7, 4))
        plt.plot(log_lambdas, wamd_vec, marker='o', color='steelblue')
        plt.axvline(log_lambdas[best_idx], color='r', linestyle='--', label='Min wAMD')
        plt.title("Weighted Absolute Mean Difference vs log(Lambda)")
        plt.xlabel("log(Lambda)")
        plt.ylabel("Weighted Absolute Mean Difference (wAMD)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # Determine save path
        plot_save_path = os.path.join(base_dir, "wamd_vs_loglambda")
        os.makedirs(plot_save_path, exist_ok=True)
        plt.savefig(os.path.join(plot_save_path, f"wamd_vs_loglambda_rep{rep}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(log_lambdas, amd_vec, marker='o', color='steelblue')
        plt.axvline(log_lambdas[best_idx], color='r', linestyle='--', label='Min wAMD')
        plt.title("Absolute Mean Difference vs log(Lambda)")
        plt.xlabel("log(Lambda)")
        plt.ylabel("Absolute Mean Difference (AMD)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # Determine save path
        plot_save_path = os.path.join(base_dir, "amd_vs_loglambda")
        os.makedirs(plot_save_path, exist_ok=True)
        plt.savefig(os.path.join(plot_save_path, f"amd_vs_loglambda_rep{rep}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    

    return best_ate, wamd_vec, ate_vec, best_selected_mask, best_betas_hat