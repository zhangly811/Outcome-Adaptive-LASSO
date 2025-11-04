
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# 📊 1. Boxplot: ATE comparison
# ---------------------------------------------------------------------
def plot_ate_comparison(df_res, base_dir):
    """
    Plot and save the boxplot comparing ATE estimates across methods.

    Args:
        df_res (pd.DataFrame): DataFrame containing columns ['method', 'ate'].
        base_dir (str): Path to the results directory.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.boxplot(x="method", y="ate", data=df_res, ax=ax, palette=sns.color_palette("Set1"))
    ax.grid(True, alpha=0.3)
    ax.set_title("Outcome Adaptive LASSO vs IPW alternatives")
    plt.tight_layout()

    fig_path = os.path.join(base_dir, "compare_oal_ipw_output.png")
    fig.savefig(fig_path, dpi=300)
    plt.close()

    print(f"✅ Boxplot saved: {fig_path}")

# ---------------------------------------------------------------------
# 📈 2. Selection proportion plots of propensity score models
# ---------------------------------------------------------------------
def plot_treatment_selection_proportion(selected_records, base_dir, d, cols_Xc, cols_Xp, cols_Xi):
    """
    Plot and save the selection proportion for each method’s propensity model.

    Args:
        selected_records (dict): {method_name: [binary selection_mask arrays]}.
        base_dir (str): Path to save figure.
        d (int): Total number of covariates.
        cols_Xc, cols_Xp, cols_Xi (list[str]): Names of confounder, predictor, and exposure covariates.
    """
    plt.figure(figsize=(8, 5))
    styles = {
        "OAL": ("black", "-", 2),
        "IPWX": ("green", (0, (3, 1, 1, 1)), 1.5),  # custom dash-dot pattern for variety
        "Conf": ("red", "--", 1.5),
        "Targ": ("blue", ":", 1.5),
        "PotConf": ("orange", "-.", 1.5),
    }

    for method, (color, ls, lw) in styles.items():
        sel_matrix = np.vstack(selected_records[method])
        selection_proportion = sel_matrix.mean(axis=0)
        x = np.arange(1, d + 1)
        plt.plot(x, selection_proportion, color=color, linestyle=ls, linewidth=lw, label=method)

    # Vertical dashed lines mark covariate group boundaries
    plt.axvline(len(cols_Xc), color="gray", linestyle="--", alpha=0.4)
    plt.axvline(len(cols_Xc) + len(cols_Xp), color="gray", linestyle="--", alpha=0.4)
    plt.axvline(len(cols_Xc) + len(cols_Xp) + len(cols_Xi), color="gray", linestyle="--", alpha=0.4)

    plt.xlabel("Covariate index")
    plt.ylabel("Proportion of times covariate selected")
    plt.ylim(0, 1.05)
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    sel_fig_path = os.path.join(base_dir, "treatment_model_selection_proportion_comparison.png")
    plt.savefig(sel_fig_path, dpi=300)
    plt.close()
    
    # Save selection proportion data
    sel_data_path = os.path.join(base_dir, "treatment_model_selection_proportion_data.csv")
    sel_data = {"covariate_index": np.arange(1, d + 1)}
    for method in selected_records.keys():
        sel_matrix = np.vstack(selected_records[method])
        sel_data[method] = sel_matrix.mean(axis=0)
    pd.DataFrame(sel_data).to_csv(sel_data_path, index=False)   

    print(f"✅ Treatment model selection proportion plot saved: {sel_fig_path}")

# ---------------------------------------------------------------------
# 📈 3. Selection proportion plot of outcome model (OAL)
# ---------------------------------------------------------------------
def plot_outcome_selection_proportion(selected_records_outcome, base_dir, d, cols_Xc, cols_Xp, cols_Xi):
    """
    Plot the proportion of times each covariate is selected by the outcome model across nrep.
    
    Args:
        selected_records_outcome (dict): e.g., {"OAL": [mask_rep1, mask_rep2, ...]}
        base_dir (str): base directory to save output figure and CSV
        d (int): total number of covariates
        cols_Xc, cols_Xp, cols_Xi (list[str]): lists of covariate names for vertical boundaries
    """
    plt.figure(figsize=(8, 5))
    
    sel_matrix = np.vstack(selected_records_outcome["OAL"])
    selection_proportion = sel_matrix.mean(axis=0)
    x = np.arange(1, d + 1)

    plt.plot(
        x,
        selection_proportion,
        color="purple",
        linestyle="-",
        linewidth=2,
        label="Outcome Model (OAL)"
    )

    # mark boundaries
    plt.axvline(len(cols_Xc), color="gray", linestyle="--", alpha=0.4)
    plt.axvline(len(cols_Xc) + len(cols_Xp), color="gray", linestyle="--", alpha=0.4)
    plt.axvline(len(cols_Xc) + len(cols_Xp) + len(cols_Xi), color="gray", linestyle="--", alpha=0.4)

    plt.xlabel("Covariate index")
    plt.ylabel("Proportion of times covariate selected")
    plt.title("Selection Proportion of Outcome Model (OAL)")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()

    # Save plot and data
    fig_path = os.path.join(base_dir, "outcome_model_selection_proportion.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    csv_path = os.path.join(base_dir, "outcome_model_selection_proportion.csv")
    pd.DataFrame({
        "covariate_index": np.arange(1, d + 1),
        "selection_proportion": selection_proportion
    }).to_csv(csv_path, index=False)

    print(f"✅ Outcome model selection proportion plot saved: {fig_path}")
    print(f"✅ Outcome model selection proportion data saved: {csv_path}")
    
# ---------------------------------------------------------------------
# 📊 4. True and Estimated propensity score distribution by treatment
# ---------------------------------------------------------------------
def plot_true_ps_distribution(A, true_ps, rep, save_dir):
    """Plot true propensity score distribution stratified by treatment indicator."""
    # ensure numeric stability within (0,1)
    true_ps = np.clip(true_ps, 1e-6, 1 - 1e-6)
    
    # create dataframe for plotting
    df_plot = pd.DataFrame({"A": A, "true_ps": true_ps})
    
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=df_plot[df_plot.A == 1],
        x="true_ps", fill=True, alpha=0.5, color="orangered", label="A=1"
    )
    sns.kdeplot(
        data=df_plot[df_plot.A == 0],
        x="true_ps", fill=True, alpha=0.5, color="royalblue", label="A=0"
    )
    plt.xlabel("True Propensity Score")
    plt.ylabel("Density")
    plt.title("Distribution of True Propensity Scores")
    plt.legend()
    plt.xlim(0, 1)
    plt.tight_layout()

    plot_path = f"{save_dir}/true_propensity_plot_rep{rep}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

def plot_best_ps_distribution(A, best_propensity_scores_hat, rep, base_dir):
    """Plot estimated (best) propensity score distribution stratified by treatment indicator."""
    # ensure numeric stability within (0,1)
    best_propensity_scores_hat = np.clip(best_propensity_scores_hat, 1e-6, 1 - 1e-6)
    
    # create dataframe for plotting
    df_plot = pd.DataFrame({"A": A, "best_ps_hat": best_propensity_scores_hat})
    
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=df_plot[df_plot.A == 1],
        x="best_ps_hat", fill=True, alpha=0.5, color="orangered", label="A=1"
    )
    sns.kdeplot(
        data=df_plot[df_plot.A == 0],
        x="best_ps_hat", fill=True, alpha=0.5, color="royalblue", label="A=0"
    )
    plt.xlabel("Estimated Propensity Score")
    plt.ylabel("Density")
    plt.title("Distribution of Estimated Propensity Scores")
    plt.legend()
    plt.xlim(0, 1)
    plt.tight_layout()

    save_path = os.path.join(base_dir, "treatment_model")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"estimated_propensity_plot_rep{rep}.png"), dpi=300)
    plt.close()
    
    # save data
    df_plot.to_csv(os.path.join(save_path, f"estimated_propensity_data_rep{rep}.csv"), index=False)

# ---------------------------------------------------------------------
# 5. wAMD before vs. after grid plot
# ---------------------------------------------------------------------
def plot_wamd_before_after_grid(wamd_dir, n_reps=5, n_lambdas=9, figsize=(18, 10)):
    """
    Plot scatter plots comparing wAMD before vs. after for each covariate,
    across multiple reps (rows) and lambdas (columns).

    Args:
        base_dir (str): Directory containing wAMD .csv files.
        n_reps (int): Number of reps to plot (rows).
        n_lambdas (int): Number of lambda values per rep (columns).
        figsize (tuple): Size of the overall figure.
    """
    base_dir = os.path.dirname(wamd_dir)
    fig, axes = plt.subplots(n_reps, n_lambdas, figsize=figsize, sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    for rep in range(n_reps):
        # Load wAMD before file
        before_path = os.path.join(wamd_dir, f"wamd_before_per_covariate_rep{rep}.csv")
        if not os.path.exists(before_path):
            print(f"⚠️ Missing file: {before_path}, skipping rep {rep}")
            continue
        df_before = pd.read_csv(before_path)
        wamd_before = df_before["wamd_before"].values if "wamd_before" in df_before.columns else df_before.iloc[:, 1].values

        for lam in range(n_lambdas):
            ax = axes[rep, lam] if n_reps > 1 else axes[lam]
            after_path = os.path.join(wamd_dir, f"wamd_after_per_covariate_rep{rep}_lambda{lam}.csv")
            if not os.path.exists(after_path):
                ax.axis("off")
                continue

            df_after = pd.read_csv(after_path)
            wamd_after = df_after["wamd_after"].values if "wamd_after" in df_after.columns else df_after.iloc[:, 1].values

            # Ensure matching lengths
            n = min(len(wamd_before), len(wamd_after))
            wamd_before, wamd_after = wamd_before[:n], wamd_after[:n]

            # Scatter plot
            ax.scatter(wamd_before, wamd_after, alpha=0.6, s=15, color="steelblue", edgecolors="none")
            ax.plot([0, max(wamd_before.max(), wamd_after.max())],
                    [0, max(wamd_before.max(), wamd_after.max())],
                    "r--", lw=1)  # identity line

            if rep == 0:
                ax.set_title(f"λ{lam}", fontsize=10)
            if lam == 0:
                ax.set_ylabel(f"Rep {rep}", fontsize=9)

    # Common labels and save
    fig.text(0.5, 0.04, "wAMD before", ha="center", fontsize=12)
    fig.text(0.04, 0.5, "wAMD after", va="center", rotation="vertical", fontsize=12)
    plt.suptitle("wAMD Before vs. After (First 5 Reps × 9 Lambdas)", fontsize=14, y=1.02)

    fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    out_path = os.path.join(base_dir, "wamd_before_after_grid.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ wAMD before/after grid saved to: {out_path}")

# ---------------------------------------------------------------------
# 6. AMD and wAMD vs log(lambda) plots
# ---------------------------------------------------------------------
def plot_amd_vs_loglambda(log_lambdas, amd_vec, best_idx, rep, base_dir):
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
    plot_save_path = os.path.join(base_dir, "treatment_model")
    os.makedirs(plot_save_path, exist_ok=True)
    plt.savefig(os.path.join(plot_save_path, f"amd_vs_loglambda_rep{rep}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_wamd_vs_loglambda(log_lambdas, wamd_vec, best_idx, rep, base_dir):
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
    plot_save_path = os.path.join(base_dir, "treatment_model")
    os.makedirs(plot_save_path, exist_ok=True)
    plt.savefig(os.path.join(plot_save_path, f"wamd_vs_loglambda_rep{rep}.png"), dpi=300, bbox_inches='tight')
    plt.close()