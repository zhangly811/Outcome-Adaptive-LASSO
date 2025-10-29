# post-analysis plotting

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_wamd_before_after_grid(base_dir, n_reps=5, n_lambdas=9, figsize=(18, 10)):
    """
    Plot scatter plots comparing wAMD before vs. after for each covariate,
    across multiple reps (rows) and lambdas (columns).

    Args:
        base_dir (str): Directory containing wAMD .csv files.
        n_reps (int): Number of reps to plot (rows).
        n_lambdas (int): Number of lambda values per rep (columns).
        figsize (tuple): Size of the overall figure.
    """
    fig, axes = plt.subplots(n_reps, n_lambdas, figsize=figsize, sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    for rep in range(n_reps):
        # Load wAMD before file
        before_path = os.path.join(base_dir, f"wamd_before_per_covariate_rep{rep}.csv")
        if not os.path.exists(before_path):
            print(f"⚠️ Missing file: {before_path}, skipping rep {rep}")
            continue
        df_before = pd.read_csv(before_path)
        wamd_before = df_before["wamd_before"].values if "wamd_before" in df_before.columns else df_before.iloc[:, 1].values

        for lam in range(n_lambdas):
            ax = axes[rep, lam] if n_reps > 1 else axes[lam]
            after_path = os.path.join(base_dir, f"wamd_after_per_covariate_rep{rep}_lambda{lam}.csv")
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

plot_wamd_before_after_grid(base_dir="res/20251029_154427/wamd", n_reps=5, n_lambdas=9, figsize=(18, 10))
