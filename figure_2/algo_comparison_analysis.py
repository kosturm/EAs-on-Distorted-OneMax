import numpy as np
import matplotlib.pyplot as plt
import scienceplots 
import pandas as pd
import glob

linew = 3
plt.style.use(['ieee'])
plt.rcParams['font.size'] = '14'
folder_name = "output_files"

# Create a figure with a specific size
fig, ax = plt.subplots(figsize=(11, 6.34))

def plot_algo(algo, ax, folder_name):
    labels = {"SEAR": 'SA-$(1, \lambda)$-EA', "OLEA": '$(1, \lambda)$-EA', "OPEA": '$(1 + \lambda)$-EA'}
    file_list = glob.glob(f"{folder_name}/{algo}_*.csv")
    combined_df = pd.concat([pd.read_csv(file, header=None, names=['dim', 'p_value', 'evaluations']) for file in file_list], ignore_index=True)
    # Calculate median data
    median_data = combined_df.groupby('dim')['evaluations'].median().reset_index()
    ax.plot(median_data['dim'], median_data['evaluations'], label=labels[algo], marker='o', linewidth=3)

#Plot n ln n
n_values = np.arange(60, 520, 20)
y_values = [n * np.log(n) / ((np.e / (np.e - 1)) ** (-1.5 * np.log(n))) for n in n_values]
ax.plot(n_values, y_values, label="$(n \ln{n})/p$")

# Plot algorithms
algorithms = ["OLEA", "SEAR", "OPEA"]
for algo in algorithms:
    plot_algo(algo, ax, folder_name)

# Add cutoff horizontal line
ax.axhline(y=1_000_000, linestyle='--', label='Cutoff', linewidth=3)

# Set labels, title, and grid
ax.set_xlabel('n')
ax.set_ylabel('Median Number of Evaluations')
#ax.set_yticks([0, 200000, 400000, 600000, 800000, 1000000])
#ax.set_yticklabels(['$0$', '$2 \cdot 10^6$', '$4 \cdot 10^6$', '$6 \cdot 10^6$', '$8 \cdot 10^6$', '$10^7$'])
ax.grid(True)

# Set legend
handles, labels = ax.get_legend_handles_labels()
print(labels)
# Order from smallest to largest dimension (or any custom order)
desired_order = [4,0,2,3,1]
new_labels = [labels[i] for i in desired_order]
new_handles = [handles[i] for i in desired_order] 

# Set the legend with the new order
ax.legend(new_handles, new_labels)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(f'{folder_name}/plot.png')
plt.savefig(f'{folder_name}/plot.eps')