import pandas as pd
import glob
import os
from collections import Counter

import matplotlib.pyplot as plt

def analyze_common_electrodes(directory_path):
    """
    Analyze common electrodes across TSV files and create a bar chart.
    
    Args:
        directory_path (str): Path to directory containing TSV files
    """
    
    # Find all TSV files in the directory
    tsv_files = glob.glob(os.path.join(directory_path, "*.tsv"))
    
    if not tsv_files:
        print("No TSV files found in the specified directory.")
        return
    
    # Collect all electrode names
    all_electrodes = []
    interp_electrodes = []
    
    for file_path in tsv_files:
        try:
            # Load TSV file
            df = pd.read_csv(file_path, sep='\t')
            
            # Check if 'good_channels' column exists
            if 'good_channels' in df.columns:
                # Get electrode names (remove NaN values)
                electrodes = df['good_channels'].dropna().tolist()
                electrodes = [e.split(',') for e in electrodes]
                electrodes = [item.strip() for sublist in electrodes for item in sublist]
                all_electrodes.extend(electrodes)
            else:
                print(f"Warning: 'good_channels' column not found in {file_path}")
            
            # Check if 'interpolated_channels' column exists
            if 'interpolated_channels' in df.columns:
                # Get electrode names (remove NaN values)
                electrodes = df['interpolated_channels'].dropna().tolist()
                electrodes = [e.split(',') for e in electrodes]
                electrodes = [item.strip() for sublist in electrodes for item in sublist]
                interp_electrodes.extend(electrodes)
            else:
                print(f"Warning: 'interpolated_channels' column not found in {file_path}")
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if not all_electrodes:
        print("No electrode data found.")
        return
    
    # Count electrode occurrences
    electrode_counts = Counter(all_electrodes)
    interp_electrode_counts = Counter(interp_electrodes)
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    electrodes = list(electrode_counts.keys())
    counts = list(electrode_counts.values())
    interp_electrodes = list(interp_electrode_counts.keys())
    interp_counts = list(interp_electrode_counts.values())

    # Get all unique electrodes from both datasets
    all_unique_electrodes = sorted(set(electrodes + interp_electrodes))
    
    # Create counts for each electrode (0 if not present)
    good_counts = [electrode_counts.get(e, 0) for e in all_unique_electrodes]
    interp_counts_aligned = [interp_electrode_counts.get(e, 0) for e in all_unique_electrodes]
    
    # Sort based on 10/20 list
    ten_twenty_channels = [
        "Fp1", "Fp2",
        "F7", "F3", "Fz", "F4", "F8",
        "T7", "C3", "Cz", "C4", "T8",
        "P7", "P3", "Pz", "P4", "P8",
        "O1", "O2"
    ]

    # Separate 10-20 channels from others
    ten_twenty_present = [ch for ch in ten_twenty_channels if ch in all_unique_electrodes]
    other_channels = sorted([ch for ch in all_unique_electrodes if ch not in ten_twenty_channels], 
                          key=lambda ch: electrode_counts.get(ch, 0), reverse=True)
    
    # Combine in desired order: 10-20 first, then others
    ordered_electrodes = ten_twenty_present + other_channels
    
    # Create counts for ordered electrodes
    good_counts_ordered = [electrode_counts.get(e, 0) for e in ordered_electrodes]
    interp_counts_ordered = [interp_electrode_counts.get(e, 0) for e in ordered_electrodes]
    
    # Create stacked bar chart
    plt.bar(ordered_electrodes, good_counts_ordered, color='skyblue', label='Good Channels')
    plt.bar(ordered_electrodes, interp_counts_ordered, bottom=good_counts_ordered, color='lightcoral', label='Interpolated Channels')
    
    # Add vertical line separator between 10-20 and other channels
    if ten_twenty_present and other_channels:
        separator_position = len(ten_twenty_present) - 0.5
        plt.axvline(x=separator_position, color='black', linestyle='--', linewidth=2, alpha=0.7)
        plt.text((len(ten_twenty_present) - 0.5) / 2, max(good_counts_ordered) + max(interp_counts_ordered) - 10,
                 '10-20 Channels', horizontalalignment='center', fontsize=10, fontweight='bold')
    
    plt.xlim(-0.5, len(ordered_electrodes) - 0.5)
    plt.ylim(0, max(good_counts_ordered) + max(interp_counts_ordered) + 5)
    plt.xlabel('Channels')
    plt.ylabel('Counts')
    plt.title('Frequency of EEG Channels Across .TRC files (SEIN + UMCU)')
    plt.xticks(rotation=90, ha='center')
    plt.legend()
    plt.tight_layout()
    plt.savefig('L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\data_availability\\eeg_channel_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"Total files processed: {len(tsv_files)}")
    print(f"Total electrode mentions: {len(all_electrodes)}")
    print(f"Unique electrodes: {len(electrode_counts)}")


if __name__ == "__main__":

    # Process folder
    directory_path = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\raw\\eeg\\EDFdata\\conversion_logs"
    analyze_common_electrodes(directory_path)