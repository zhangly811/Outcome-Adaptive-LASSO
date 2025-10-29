
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

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