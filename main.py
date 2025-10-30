import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import time
from datetime import datetime
import os
import json
from helper_func.plotting import plot_ate_comparison, plot_treatment_selection_proportion, plot_outcome_selection_proportion, plot_wamd_before_after_grid
from generate_data.synthetic_data_simulation import generate_synthetic_dataset
from model.outcome_adaptive_lasso import calc_outcome_adaptive_lasso, calc_ate_vanilla_ipw, calc_wamd_per_covariate, calc_amd_per_covariate, fit_outcome_model

warnings.filterwarnings(action='ignore') # ignore sklearn's ConvergenceWarning

# ---------------------------------------------------------------------
# 🕒 Create timestamped results folder
# ---------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = os.path.join("res", timestamp)
os.makedirs(base_dir, exist_ok=True)
data_dir = os.path.join(base_dir, "synthetic_data")
os.makedirs(data_dir, exist_ok=True)
# Save betas_hat for all lambdas in this replication
betas_hat_dir = os.path.join(base_dir, "betas_hat")
os.makedirs(betas_hat_dir, exist_ok=True)
# Save amd for all lambdas in this replication
amd_dir = os.path.join(base_dir, "amd")
os.makedirs(amd_dir, exist_ok=True)
# Save wamd for all lambdas in this replication
wamd_dir = os.path.join(base_dir, "wamd")
os.makedirs(wamd_dir, exist_ok=True)
print(f"📁 Results will be saved to: {base_dir}")

# ---------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------
res_dict = defaultdict(list)
selected_covariates_treatment = {"OAL": [], "IPWX": [], "Conf": [], "Targ": [], "PotConf": []}
selected_covariates_outcome = {"OAL": []}

# Simulation parameters
n, d, nrep, rho, eta = 1000, 8000, 5, 0, 0
scenario_num = 4

# Save parameters to a JSON file inside the results folder
config = {
    "n": n,
    "d": d,
    "nrep": nrep,
    "rho": rho,
    "eta": eta,
    "scenario_num": scenario_num,
    "timestamp": timestamp,
}

config_path = os.path.join(base_dir, "experiment_config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=4)
    
for rep in tqdm(range(nrep), desc="Simulation Progress", ncols=80):
    df = generate_synthetic_dataset(n=n, d=d, rho=rho, eta=eta, rep=rep, scenario_num=scenario_num, save_dir=data_dir)

    # Identify covariate subsets
    cols_Xc = [col for col in df if col.startswith("Xc")]
    cols_Xp = [col for col in df if col.startswith("Xp")]
    cols_Xi = [col for col in df if col.startswith("Xi")]
    cols_Xs = [col for col in df if col.startswith("Xs")]
    cols_all = cols_Xc + cols_Xp + cols_Xi + cols_Xs

    # Compute the index ranges of each subset in the full d-dimensional space
    idx_Xc = np.arange(len(cols_Xc))
    idx_Xp = np.arange(len(cols_Xc), len(cols_Xc) + len(cols_Xp))
    idx_Xi = np.arange(len(cols_Xc) + len(cols_Xp),
                       len(cols_Xc) + len(cols_Xp) + len(cols_Xi))

    # --- Calculate AMD and wAMD before propensity weighting ---
    A_values = df["A"].values
    X_values = df[cols_all].values
    ipw_unweighted = np.ones_like(A_values)
    amd_before_per_covariate = calc_amd_per_covariate(X_values, A_values, ipw_unweighted)
    amd_before = np.sum(amd_before_per_covariate)
    # save amd_before_per_covariate and amd_before as csv
    amd_before_df = pd.DataFrame({
        "covariate_index": [col for col in df if col.startswith("X")],
        "amd_before": amd_before_per_covariate
    })
    amd_before_df.to_csv(os.path.join(amd_dir, f"amd_before_per_covariate_rep{rep}.csv"), index=False)
    with open(os.path.join(amd_dir, f"amd_before_rep{rep}.txt"), "w") as f:
        f.write(str(amd_before))
    
    # --- Run Outcome Adaptive LASSO ---
    ate_oal, wamd_vec, ate_vec, selected_mask_oal, best_betas_hat, best_nus_hat = calc_outcome_adaptive_lasso(
        df["A"], df["Y"], df[cols_all], rep,
        plot=(rep < 5), 
        amd_save_path=amd_dir, 
        wamd_save_path=wamd_dir,
        base_dir=base_dir
    )
    
    wamd_before_per_covariate = calc_wamd_per_covariate(X_values, A_values, ipw_unweighted, best_betas_hat)
    wamd_before = np.sum(wamd_before_per_covariate)
    # save wamd_before_per_covariate and wamd_before as csv
    wamd_before_df = pd.DataFrame({
        "covariate_index": [col for col in df if col.startswith("X")],
        "wamd_before": wamd_before_per_covariate
    })
    wamd_before_df.to_csv(os.path.join(wamd_dir, f"wamd_before_per_covariate_rep{rep}.csv"), index=False)
    with open(os.path.join(wamd_dir, f"wamd_before_rep{rep}.txt"), "w") as f:
        f.write(str(wamd_before))
        
    # Map OAL selection (already covers all covariates)
    full_mask_oal = np.zeros(d)
    full_mask_oal[:len(selected_mask_oal)] = selected_mask_oal

    # Save betas_hat for all lambdas in this replication
    betas_hat_dir = os.path.join(base_dir, "betas_hat")
    os.makedirs(betas_hat_dir, exist_ok=True)
    best_betas_hat_df = pd.DataFrame({
    "coef_index": [col for col in df if col.startswith("X")],
    "coef_value": best_betas_hat
    })
    best_betas_hat_df.to_csv(os.path.join(betas_hat_dir, f"betas_hat_best_rep{rep}.csv"), index=False)
    # Save best_nus_hat
    nus_hat_dir = os.path.join(base_dir, "nus_hat")
    os.makedirs(nus_hat_dir, exist_ok=True)
    best_nus_hat_df = pd.DataFrame({
    "coef_index": [col for col in df if col.startswith("X")],
    "coef_value": best_nus_hat
    })
    best_nus_hat_df.to_csv(os.path.join(nus_hat_dir, f"nus_hat_best_rep{rep}.csv"), index=False)
    
    # selection mask for the outcome model
    outcome_selected_mask = (np.abs(best_betas_hat) > 1e-8).astype(int)
    selected_covariates_outcome["OAL"].append(outcome_selected_mask)

    # --- Run IPW-based methods with ALL covariates ---
    ate_ipwx, selected_mask_ipwx = calc_ate_vanilla_ipw(df["A"], df["Y"], df[cols_all], return_selection=True)
    full_mask_ipwx = np.zeros(d)
    full_mask_ipwx[:len(selected_mask_ipwx)] = selected_mask_ipwx
    
    # --- Run IPW-based methods with confounders ---
    ate_conf, mask_conf = calc_ate_vanilla_ipw(df["A"], df["Y"], df[cols_Xc], return_selection=True)
    full_mask_conf = np.zeros(d)
    full_mask_conf[idx_Xc] = mask_conf

    # --- Run IPW-based methods with confounders and predictors ---
    ate_targ, mask_targ = calc_ate_vanilla_ipw(
        df["A"], df["Y"], df[cols_Xc + cols_Xp], return_selection=True
    )
    full_mask_targ = np.zeros(d)
    full_mask_targ[np.concatenate([idx_Xc, idx_Xp])] = mask_targ

    # --- Run IPW-based methods with confounders, predictors, and instruments ---
    ate_pot_conf, mask_pot_conf = calc_ate_vanilla_ipw(
        df["A"], df["Y"], df[cols_Xc + cols_Xp + cols_Xi], return_selection=True
    )
    full_mask_pot_conf = np.zeros(d)
    full_mask_pot_conf[np.concatenate([idx_Xc, idx_Xp, idx_Xi])] = mask_pot_conf

    # Append to record
    selected_covariates_treatment["OAL"].append(full_mask_oal)
    selected_covariates_treatment["IPWX"].append(full_mask_ipwx)
    selected_covariates_treatment["Conf"].append(full_mask_conf)
    selected_covariates_treatment["Targ"].append(full_mask_targ)
    selected_covariates_treatment["PotConf"].append(full_mask_pot_conf)

    # Save ATEs
    res_dict["ate"].extend([ate_oal, ate_ipwx, ate_conf, ate_targ, ate_pot_conf])
    res_dict["method"].extend(["OAL", "IPWX", "Conf", "Targ", "PotConf"])
    res_dict["rep"].extend(5 * [rep])

# ---------------------------------------------------------------------
# Aggregate and save summary results
# ---------------------------------------------------------------------
df_res = pd.DataFrame(res_dict)
summary_path = os.path.join(base_dir, "simulation_summary.csv")
df_res.to_csv(summary_path, index=False)
print(f"✅ ATE results table saved: {summary_path}")

# ---------------------------------------------------------------------
# 📊 Generate and save plots
# ---------------------------------------------------------------------
plot_ate_comparison(df_res, base_dir)
plot_treatment_selection_proportion(selected_covariates_treatment, base_dir, d, cols_Xc, cols_Xp, cols_Xi)
plot_outcome_selection_proportion(
    selected_covariates_outcome,
    base_dir=base_dir,
    d=d,
    cols_Xc=cols_Xc,
    cols_Xp=cols_Xp,
    cols_Xi=cols_Xi
)
plot_wamd_before_after_grid(base_dir=wamd_dir, n_reps=5, n_lambdas=9, figsize=(18, 10))
print("🎉 Simulation completed successfully.")

