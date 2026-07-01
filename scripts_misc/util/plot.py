import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_patient_entry_density(dataframes_to_plot, colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], show=False):
    """
    Plots the density of patient entries across multiple dataframes.
    Each dataframe should contain a "Participant Id" column with IDs in the format "RESP-XXXX".
    Parameters:
    - dataframes_to_plot: List of tuples (label, dataframe)
    - colors: List of colors for each dataframe (default = 4 dfs)
    - show: Whether to display the plot immediately (default = False)
    Returns:
    - Matplotlib figure object
    """
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax2 = ax.twinx()

    for (label, df), color in zip(dataframes_to_plot, colors):
        # Extract patient numbers from Participant Id
        patient_numbers = df["Participant Id"].str.extract(r'(\d+)$').astype(int)[0].unique()
        # # Plot KDE
        # sns.kdeplot(
        #     patient_numbers,
        #     label=label,
        #     fill=(label == "Pathology"),
        #     alpha=0.4,
        #     common_norm=True,
        #     bw_adjust=0.5,
        #     color=color
        # )

        if label == "Pathology":
            # For pathology, plot histogram on primary y-axis
            hist = ax.hist(
            patient_numbers,
            bins=20,
            label=label,
            alpha=0.4,
            color=color,
            edgecolor='black'
            )
            # Also add vertical markers
            ylim = ax.get_ylim()
            ymax = ylim[1] * 0.02  # 2% of plot height
            ax.vlines(patient_numbers, ymin=0, ymax=ymax, color='black', linewidth=1.2, alpha=0.7)
        else:
            # Calculate percentage of entries shared with pathology per bin
            bins = np.linspace(0, 1616, 21)
            pathology_numbers = dataframes_to_plot[0][1]["Participant Id"].str.extract(r'(\d+)$')[0].dropna().astype(int).unique()
            patient_numbers = df["Participant Id"].str.extract(r'(\d+)$')[0].dropna().astype(int).unique()
            counts, _ = np.histogram(patient_numbers, bins=bins)
            # Convert patient_numbers to pandas Series for .isin()
            patient_numbers_series = pd.Series(patient_numbers)
            shared_counts, _ = np.histogram(patient_numbers_series[patient_numbers_series.isin(pathology_numbers)], bins=bins)
            percent_shared = np.divide(shared_counts, counts, out=np.zeros_like(shared_counts, dtype=float), where=counts!=0) * 100
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Plot on secondary y-axis
            ax2.plot(bin_centers, percent_shared, label=f"{label} (% complete)", color=color, marker='o')

    # Style and labels
    ax.set_xlabel("RESP-number")
    ax.set_ylabel("Selected pathology entries per bin (#)")
    ax2.set_ylabel("Entry completeness (%)")
    ax.set_title("RESPect DB Entries: selected patients")
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.set_xlim(0, 1616)
    plt.tight_layout()

    # Show plot
    if show:
        plt.show()
    
    return plt.gcf()