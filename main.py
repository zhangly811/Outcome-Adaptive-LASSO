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

from outcome_adaptive_lasso import calc_outcome_adaptive_lasso, generate_synthetic_dataset, calc_ate_vanilla_ipw

warnings.filterwarnings(action='ignore') # ignore sklearn's ConvergenceWarning

# ---------------------------------------------------------------------
# 🕒 Create timestamped results folder
# ---------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = os.path.join("res", timestamp)
os.makedirs(base_dir, exist_ok=True)
print(f"📁 Results will be saved to: {base_dir}")

# ---------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------
res_dict = defaultdict(list)
selected_records = {"OAL": [], "Conf": [], "Targ": [], "PotConf": []}

# Simulation parameters
n, d = 100, 800

for rep in tqdm(range(30), desc="Simulation Progress", ncols=80):
    df = generate_synthetic_dataset(n=n, d=d, rho=0, eta=0, scenario_num=4)

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

    # --- Run Outcome Adaptive LASSO ---
    save_path = os.path.join(base_dir, f"wamd_vs_loglambda_rep{rep}.png") if rep < 5 else None
    ate_oal, amd_vec, ate_vec, selected_mask_oal = calc_outcome_adaptive_lasso(
        df["A"], df["Y"], df[cols_all],
        plot=(rep < 5), save_path=save_path
    )

    # Map OAL selection (already covers all covariates)
    full_mask_oal = np.zeros(d)
    full_mask_oal[:len(selected_mask_oal)] = selected_mask_oal

    # --- Run IPW-based methods and map their selections into full d-space ---
    ate_conf, mask_conf = calc_ate_vanilla_ipw(df["A"], df["Y"], df[cols_Xc], return_selection=True)
    full_mask_conf = np.zeros(d)
    full_mask_conf[idx_Xc] = mask_conf

    ate_targ, mask_targ = calc_ate_vanilla_ipw(
        df["A"], df["Y"], df[cols_Xc + cols_Xp], return_selection=True
    )
    full_mask_targ = np.zeros(d)
    full_mask_targ[np.concatenate([idx_Xc, idx_Xp])] = mask_targ

    ate_pot_conf, mask_pot_conf = calc_ate_vanilla_ipw(
        df["A"], df["Y"], df[cols_Xc + cols_Xp + cols_Xi], return_selection=True
    )
    full_mask_pot_conf = np.zeros(d)
    full_mask_pot_conf[np.concatenate([idx_Xc, idx_Xp, idx_Xi])] = mask_pot_conf

    # Append to record
    selected_records["OAL"].append(full_mask_oal)
    selected_records["Conf"].append(full_mask_conf)
    selected_records["Targ"].append(full_mask_targ)
    selected_records["PotConf"].append(full_mask_pot_conf)

    # Save ATEs
    res_dict["ate"].extend([ate_oal, ate_conf, ate_targ, ate_pot_conf])
    res_dict["method"].extend(["OAL", "Conf", "Targ", "PotConf"])
    res_dict["rep"].extend(4 * [rep])

# ---------------------------------------------------------------------
# Aggregate and save summary results
# ---------------------------------------------------------------------
df_res = pd.DataFrame(res_dict)
summary_path = os.path.join(base_dir, "simulation_summary.csv")
df_res.to_csv(summary_path, index=False)
print(f"✅ Results table saved: {summary_path}")

# ---------------------------------------------------------------------
# Boxplot: ATE comparison
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
sns.boxplot(x="method", y="ate", data=df_res, ax=ax, palette=sns.color_palette("Set1"))
ax.grid(True, alpha=0.3)
ax.set_title("Outcome Adaptive LASSO vs IPW alternatives")
plt.tight_layout()
fig.savefig(os.path.join(base_dir, "compare_oal_ipw_output.png"), dpi=300)
plt.close()
print(f"✅ Boxplot saved: {os.path.join(base_dir, 'compare_oal_ipw_output.png')}")

# ---------------------------------------------------------------------
# Selection proportion plots
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))
styles = {
    "OAL": ("black", "-", 2),
    "Conf": ("red", "--", 1.5),
    "Targ": ("blue", ":", 1.5),
    "PotConf": ("orange", "-.", 1.5),
}

for method, (color, ls, lw) in styles.items():
    sel_matrix = np.vstack(selected_records[method])
    selection_proportion = sel_matrix.mean(axis=0)
    x = np.arange(1, d + 1)
    plt.plot(x, selection_proportion, color=color, linestyle=ls, linewidth=lw, label=method)

# Optional: add vertical lines marking the subset boundaries
plt.axvline(len(cols_Xc), color="gray", linestyle="--", alpha=0.4)
plt.axvline(len(cols_Xc) + len(cols_Xp), color="gray", linestyle="--", alpha=0.4)
plt.axvline(len(cols_Xc) + len(cols_Xp) + len(cols_Xi), color="gray", linestyle="--", alpha=0.4)

plt.xlabel("Covariate index")
plt.ylabel("Proportion of times covariate selected")
plt.ylim(0, 1.05)
plt.legend(frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()

sel_fig_path = os.path.join(base_dir, "selection_proportion_comparison.png")
plt.savefig(sel_fig_path, dpi=300)
plt.close()
print(f"✅ Selection proportion plot saved: {sel_fig_path}")

print("🎉 Simulation completed successfully.")