import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file1 = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\selection\\RESPectDB_CASTOR_export_2025_09_03\\RESPectDB_export_20250903.csv"
file2 = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\selection\\RESPectDB_CASTOR_export_2025_09_03\\RESPectDB_Pathology_export_20250903.csv"

# Read CSV files
df_all = pd.read_csv(file1, sep=';')
df_pa = pd.read_csv(file2, sep=';')

# Get sets of participant IDs
ids1 = df_all['Participant Id'].unique()
ids2 = df_pa['Participant Id'].unique()

ids_not_in_ids2 = set(ids1) - set(ids2)
ids_not_in_ids2 = sorted(list(ids_not_in_ids2))

# Plot histogram
plt.figure(figsize=(10, 6))

# Extract the numeric part (xxxx) from "RESPxxxx"
ids_numeric = [int(pid.replace('RESP', '')) for pid in ids_not_in_ids2]

# Histogram for ids_not_in_ids2
counts, bins = np.histogram(ids_numeric, bins=(max(ids_numeric)-min(ids_numeric)+1) // 50)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Get all ids from df_all for reference
all_ids_numeric = [int(pid.replace('RESP', '')) for pid in ids1]
total_counts, _ = np.histogram(all_ids_numeric, bins=bins)

# Plot the total bar (100%)
plt.bar(bin_centers, total_counts, width=(bins[1]-bins[0]), color='lightgray', edgecolor='black', label='All Participants')

# Plot the overlay bar for ids_not_in_ids2
plt.bar(bin_centers, counts, width=(bins[1]-bins[0]), color='blue', edgecolor='black', label='Not in Pathology Export')

plt.xlabel('Participant Id (RESPxxxx)')
plt.ylabel('Count')

# Set x-ticks every 100, formatted as RESPxxxx
xtick_min = (min(all_ids_numeric) // 100) * 100
xtick_max = ((max(all_ids_numeric) // 100) + 1) * 100
xticks = list(range(xtick_min, xtick_max + 1, 100))
xtick_labels = [f'RESP{str(x).zfill(4)}' for x in xticks]
plt.xticks(xticks, xtick_labels, rotation=45)

# Line
plt.axvline(x=1050, color='red', linestyle='--', linewidth=2)
plt.text(1050, plt.ylim()[1]*0.95, 'Manually checked from here', color='red', rotation=90, va='top', ha='right', fontsize=10)
plt.axvspan(1050, xtick_max, color='red', alpha=0.2)

plt.xlim(0, bin_centers[-1] + (bins[1]-bins[0])/2)

plt.title('Histogram of Participant Ids not in Pathology Export')
plt.legend()
plt.tight_layout()
plt.show()