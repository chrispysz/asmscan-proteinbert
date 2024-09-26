import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from cycler import cycler

def load_data(path):
    return pd.read_csv(path, sep="\t")

def create_scatter_plot(ax, data, title, score_threshold):
    groups = data.groupby("dataset")
    
    # Create a color cycle
    color_cycle = plt.cm.rainbow(np.linspace(0, 1, len(groups)))
    ax.set_prop_cycle(cycler(color=color_cycle))
    
    for name, dset in groups:
        # Plot points above or equal to threshold
        above_threshold = dset[dset['prob'] >= score_threshold]
        ax.scatter(above_threshold["umap_x"], above_threshold["umap_y"], 
                   label=f"{name} (n={len(dset)})", s=10, alpha=1.0, marker='o')
        
        # Plot points below threshold with 'x' marker
        below_threshold = dset[dset['prob'] < score_threshold]
        ax.scatter(below_threshold["umap_x"], below_threshold["umap_y"], 
                   s=10, alpha=0.2, marker='x', color=ax._get_lines.get_next_color())
    
    if "LSTM" in title:
        ax.legend(title="Datasets", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(title)
    if "ProteinBERT" in title:
        ax.set_xlabel("UMAP X")
        ax.set_ylabel("UMAP Y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def create_retention_plot(ax, data, score_threshold, title):
    total_counts = data.groupby("dataset").size()
    filtered_counts = data[data['prob'] >= score_threshold].groupby("dataset").size()
    retention_percentages = (filtered_counts / total_counts * 100).round(2).fillna(0)
    
    bars = retention_percentages.plot(kind='bar', ax=ax, color=plt.cm.rainbow(np.linspace(0, 1, len(retention_percentages))))
    ax.set_title(title)
    ax.set_ylim(0, 100)

    if "ProteinBERT" in title:
        ax.set_xlabel("Datasets")
        ax.set_ylabel("Retention %")
    else:
        ax.set_xlabel("")
    
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for i, v in enumerate(retention_percentages):
        ax.text(i, v + 1, f'{v}%', ha='center', va='bottom', fontweight='bold')
    

def create_plot(path_pbert, path_lstm, score_threshold=0.5, figsize=(8, 6)):
    data_pbert = load_data(path_pbert)
    data_lstm = load_data(path_lstm)
   
    fig, axes = plt.subplots(2, 2, figsize=(figsize[0]*2, figsize[1]*2))
    fig.suptitle(f"ProteinBERT vs LSTM Comparison (Threshold: {score_threshold})", fontsize=16)
    axes = axes.flatten()  # Flatten the 2x2 array to make indexing easier
   
    # Scatter plot for PBERT with all data
    create_scatter_plot(axes[0], data_pbert, "ProteinBERT", score_threshold)
   
    # Scatter plot for LSTM with all data
    create_scatter_plot(axes[1], data_lstm, "LSTM", score_threshold)
   
    # Retention plot for PBERT
    create_retention_plot(axes[2], data_pbert, score_threshold, "ProteinBERT")
   
    # Retention plot for LSTM
    create_retention_plot(axes[3], data_lstm, score_threshold, "LSTM")
   
    plt.tight_layout()
    plt.savefig("comparison_plot.png", dpi=300)
    return fig, axes

# Example usage:
create_plot(path_lstm="dev/results/bilstm_umap_domains.csv",
            path_pbert="dev/results/NLReff_test.bass_ntm_domain_test.fass_ntm_domain_test.fass_ctm_domain_test.csga2203_nr40.csv",
            score_threshold=0.5)