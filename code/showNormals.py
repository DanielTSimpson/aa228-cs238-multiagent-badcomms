import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

def run():
    # --- 1. ROBUST PATH SETUP ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'Config_Results.csv')

    if not os.path.exists(filename):
        parent_file = os.path.join(script_dir, '../Config_Results.csv')
        if os.path.exists(parent_file):
            filename = parent_file

    try:
        df = pd.read_csv(filename)
        print(f"Successfully loaded data from: {filename}")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find 'Config_Results.csv'.")
        exit()

    # 2. Define the columns we want to analyze
    # Note: We handle 'Likelihood of Failure' separately as the 4th plot
    columns_to_plot = ['Total Time', '# Comms', 'Total Cost']

    # 3. Set up the figure (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Simulation Analysis & Failure Rates', fontsize=16)

    axes_flat = axes.flatten()
    colors = ['blue', 'green', 'red']

    # --- PLOT 1-3: NORMAL DISTRIBUTIONS ---
    for i, col in enumerate(columns_to_plot):
        ax = axes_flat[i]
        
        if col not in df.columns:
            ax.text(0.5, 0.5, 'Data Not Found', ha='center', va='center')
            continue
        
        # Force column to numeric. "FAILED" strings become NaN
        raw_data = pd.to_numeric(df[col], errors='coerce')
        data = raw_data.dropna() # Successes only
        
        # Skip if empty
        if len(data) < 2:
            ax.text(0.5, 0.5, 'Not Enough Data', ha='center', va='center')
            continue

        # Statistics
        mu, std = norm.fit(data)
        
        # Histogram
        ax.hist(data, bins=16, density=True, alpha=0.5, color=colors[i], label='Actual Data')
        
        # Normal Curve
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2, label='Normal Dist. Fit')
        
        # Formatting
        ax.set_title(rf"{col}" + "\n" + rf"($\mu={mu:.2f}, \sigma={std:.2f}$)")
        ax.set_xlabel(col)
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    # --- PLOT 4: LIKELIHOOD OF FAILURE ---
    ax_fail = axes_flat[3]

    # Calculate Failure Rate using 'Total Time' column
    # Any row where 'Total Time' is NOT a number is considered a failure
    total_trials = len(df)
    numeric_times = pd.to_numeric(df['Total Time'], errors='coerce')
    failed_trials = numeric_times.isna().sum()
    success_trials = total_trials - failed_trials

    if total_trials > 0:
        failure_rate = (failed_trials / total_trials) * 100
    else:
        failure_rate = 0.0

    # Create Bar Chart
    labels = ['Success', 'Failure']
    counts = [success_trials, failed_trials]
    bar_colors = ['green', 'red']

    bars = ax_fail.bar(labels, counts, color=bar_colors, alpha=0.7)
    
    # Add text labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax_fail.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')

    # Formatting
    ax_fail.set_title(f"Likelihood of Failure\n(Rate: {failure_rate:.1f}%)")
    ax_fail.set_ylabel('Number of Trials')
    ax_fail.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Adjust layout
    plt.tight_layout()
    
    save_path = os.path.join(script_dir, "ConfigEntropyvTime_TrialExample.png")
    plt.savefig(save_path)
    print(f"Figure saved to: {save_path}")
    
    plt.show()

if __name__ == '__main__':
    run()