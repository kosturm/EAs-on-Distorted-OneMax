import numpy as np
import matplotlib.pyplot as plt
import scienceplots 
import pandas as pd
import re
import os

linew = 3
plt.style.use(['ieee'])
plt.rcParams['font.size'] = '14'
folder_name = "output_files"

# Create a figure with a specific size
fig, ax = plt.subplots(figsize=(11, 6))

def plot_dim(dim, ax, folder_name):
    # Load data from CSV file
    data = pd.read_csv(f'{folder_name}/{dim}_values.csv', header=None, names=['dim', 'p_value', 'evaluations'])
    # Scale evaluations
    scale_factor = data['p_value'] / (dim * np.log(dim))
    data['evaluations'] *= scale_factor
    # Group by p_value and calculate mean and std deviation
    grouped_data = data.groupby('p_value')['evaluations'].agg(['mean', 'std']).reset_index()
    # Plot error bars
    ax.errorbar(grouped_data['p_value'], grouped_data['mean'], yerr=grouped_data['std'], label=f'$n={dim}$', marker='o', linewidth=linew, capsize=5)

dimensions = [500, 400, 300]
for dim in dimensions:
    plot_dim(dim, ax, folder_name)

# Set labels, title, and y-scale
ax.set_xlabel('p')
ax.set_ylabel('$T \cdot p / (n \cdot \ln{n})$')
ax.set_yscale('log')
ax.set_yticks([1, 10, 50])
ax.set_yticklabels(['1', '10', '50'])
ax.grid(True)

# Retrieve handles and labels and then modify the order
handles, labels = ax.get_legend_handles_labels()
# Order from smallest to largest dimension (or any custom order)
desired_order = [2, 1, 0]  # Assuming you want the reverse of the original order
new_labels = [labels[i] for i in desired_order]
new_handles = [handles[i] for i in desired_order] 

# Set the legend with the new order
ax.legend(new_handles, new_labels)

# Use tight layout to adjust plot elements
plt.tight_layout()

# Save the plot to a file
plt.savefig(f'{folder_name}/plot_diff_p.png')