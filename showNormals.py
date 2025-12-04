import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Load the data from the CSV file
filename = 'Config_Results.csv'

try:
    df = pd.read_csv(filename)
    print(f"Successfully loaded data from {filename}")
except FileNotFoundError:
    print(f"Error: Could not find '{filename}' in the current directory.")
    exit()

# 2. Define the columns we want to analyze
# These must match your CSV headers exactly
columns_to_plot = ['Total Time', '# Comms', 'Total Cost']

# 3. Set up the figure (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Normal Distribution Analysis: {filename}', fontsize=16)

colors = ['blue', 'green', 'red']

for i, col in enumerate(columns_to_plot):
    # Select the specific subplot
    ax = axes[i]
    
    # Extract the data column
    if col not in df.columns:
        print(f"Warning: Column '{col}' not found in CSV. Skipping.")
        continue
        
    data = df[col]
    
    # --- STATISTICS ---
    # Fit a normal distribution to the data: gives mean (mu) and std dev (std)
    mu, std = norm.fit(data)
    
    # --- PLOTTING ---
    # 1. Histogram (Actual Data)
    # density=True normalizes it so area=1, allowing comparison with the bell curve
    ax.hist(data, bins=8, density=True, alpha=0.5, color=colors[i], label='Actual Data')
    
    # 2. Normal Distribution Curve (Theoretical)
    # Generate range of x-values from min to max of the plot
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std) # Probability Density Function
    
    ax.plot(x, p, 'k', linewidth=2, label='Normal Dist. Fit')
    
    # --- FORMATTING ---
    ax.set_title(f"{col}\n($\mu={mu:.2f}, \sigma={std:.2f}$)")
    ax.set_xlabel(col)
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()