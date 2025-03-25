import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(path):
    """
    Load tab-separated data from a CSV file.

    Parameters:
        path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.ParserError: If the file cannot be parsed.
    """
    try:
        data = pd.read_csv(path, sep="\t")
        return data
    except FileNotFoundError as e:
        print(f"Error: The file at path '{path}' was not found.")
        raise e
    except pd.errors.ParserError as e:
        print(f"Error: The file at path '{path}' could not be parsed. Please check the format.")
        raise e


def create_scatter_plot(ax, data, score_threshold, color_dict, order):
    """
    Create a scatter plot of UMAP embeddings with points colored by dataset.

    Parameters:
        ax (matplotlib.axes.Axes): The axes on which to plot.
        data (pandas.DataFrame): DataFrame containing 'umap_x', 'umap_y', 'prob', and 'dataset' columns.
        score_threshold (float): Threshold to separate high and low probability points.
        color_dict (dict): Dictionary mapping dataset names to colors.
        order (list): List of dataset names in the desired order.
    """
    for name in order:
        if name in data['dataset'].unique():
            dataset_group = data[data['dataset'] == name]
            color = color_dict[name]

            # Points above or equal to threshold
            above_threshold = dataset_group[dataset_group['prob'] >= score_threshold]
            sns.scatterplot(x="umap_x", y="umap_y", data=above_threshold,
                            ax=ax, label=None, color=color, s=28, alpha=1.0, marker='o', linewidth=0)

            # Points below threshold
            below_threshold = dataset_group[dataset_group['prob'] < score_threshold]
            sns.scatterplot(x="umap_x", y="umap_y", data=below_threshold,
                            ax=ax, label=None, color=color, s=28, alpha=0.4, marker='x', linewidth=1.0)

    ax.set_xlabel("UMAP X")
    ax.set_ylabel("UMAP Y")
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)

    ax.tick_params(axis='both', which='major', labelsize=18)


def create_retention_plot(ax, data, score_threshold, color_dict, y_max):
    """
    Create a bar plot showing the retention percentage for each dataset.

    Parameters:
        ax (matplotlib.axes.Axes): The axes on which to plot.
        data (pandas.DataFrame): DataFrame containing 'prob' and 'dataset' columns.
        score_threshold (float): Threshold to determine retention.
        color_dict (dict): Dictionary mapping dataset names to colors.
        y_max (int): Maximum value for the Y-axis across all retention plots.

    Returns:
        list: Ordered list of dataset names sorted by retention percentage.
    """
    # Calculate total and filtered counts
    total_counts = data.groupby("dataset").size()
    filtered_counts = data[data['prob'] >= score_threshold].groupby("dataset").size()
    retention_percentages = (filtered_counts / total_counts * 100).round(0).fillna(0).astype(int)

    retention_df = retention_percentages.reset_index()
    retention_df.columns = ['dataset', 'retention']
    retention_df = retention_df.sort_values(by='retention', ascending=False)

    sns.barplot(x='dataset', y='retention', data=retention_df,
                ax=ax, palette=[color_dict[name] for name in retention_df['dataset']])

    ax.set_ylim(0, y_max)
    ax.set_xlabel("Datasets")
    ax.set_ylabel("Retention (%)")
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)

    ax.tick_params(axis='both', which='major', labelsize=18)

    # Add percentage labels on top of each bar
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + y_max * 0.01,
                f'{int(height)}%', ha='center', va='bottom', fontweight='bold', fontsize=18)

    return retention_df['dataset'].tolist()


def create_plot(path_pbert, path_lstm, score_threshold=0.5, figsize=(8, 6)):
    """
    Create a comparison plot between ProteinBERT and LSTM models.

    Parameters:
        path_pbert (str): Path to the ProteinBERT data CSV file.
        path_lstm (str): Path to the LSTM data CSV file.
        score_threshold (float): Threshold to separate high and low probability points.
        figsize (tuple): Figure size in inches (width, height).

    Returns:
        tuple: The figure and axes objects.
    """
    # Load data
    data_pbert = load_data(path_pbert)
    data_lstm = load_data(path_lstm)

    required_columns = {'umap_x', 'umap_y', 'prob', 'dataset'}
    if not required_columns.issubset(data_pbert.columns):
        raise ValueError("ProteinBERT data must contain 'umap_x', 'umap_y', 'prob', and 'dataset' columns.")
    if not required_columns.issubset(data_lstm.columns):
        raise ValueError("LSTM data must contain 'umap_x', 'umap_y', 'prob', and 'dataset' columns.")

    fig, axes = plt.subplots(2, 2, figsize=(figsize[0] * 2, figsize[1] * 2))
    axes = axes.flatten()

    all_datasets = sorted(set(data_pbert['dataset'].unique()) | set(data_lstm['dataset'].unique()))
    num_datasets = len(all_datasets)
    if num_datasets == 5:
        palette = sns.color_palette(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        palette.append('#7f7f7f')
    else:
        palette = sns.color_palette('hls', n_colors=num_datasets)
    color_dict = dict(zip(all_datasets, palette))

    y_max = 105

    pbert_order = create_retention_plot(axes[2], data_pbert, score_threshold,
                                        color_dict, y_max)

    lstm_order = create_retention_plot(axes[3], data_lstm, score_threshold,
                                       color_dict, y_max)

    # Scatter plot for ProteinBERT
    create_scatter_plot(axes[0], data_pbert, score_threshold,
                        color_dict, pbert_order)

    # Scatter plot for LSTM
    create_scatter_plot(axes[1], data_lstm, score_threshold,
                        color_dict, pbert_order)

    # Create counts for datasets
    counts = data_pbert['dataset'].value_counts()

    # Create a unified legend with counts
    labels = [f"{dataset} (n={counts[dataset]})" for dataset in pbert_order]
    handles = [plt.Line2D([0], [0], marker='o', color=color_dict[dataset],
                          linestyle='', markersize=5) for dataset in pbert_order]
    fig.legend(handles, labels, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, title="Datasets", title_fontsize=18,
               fontsize=18, markerscale=2.5)

    subplot_labels = ['A', 'B', 'C', 'D']
    for label, ax in zip(subplot_labels, axes):
        ax.text(0.00, 1.1, label, transform=ax.transAxes, fontsize=30,
                fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, hspace=0.22, wspace=0.22)

    plt.savefig("cf_umap_families.pdf", bbox_inches='tight')
    plt.show()
    return fig, axes


create_plot(path_lstm="./results/results/bilstm_umap_families.csv",
            path_pbert="./results/results/pbert_umap_families.csv",
            score_threshold=0.5)
