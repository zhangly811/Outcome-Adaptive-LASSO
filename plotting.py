
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

    print(f"✅ Selection proportion plot saved: {sel_fig_path}")

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