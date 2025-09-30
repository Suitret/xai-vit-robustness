import os
import numpy as np
import pandas as pd

# Define paths and parameters
metrics = ['insertion', 'deletion', 'sparseness', 'sensitivity']
conditions = ['clean', 'brightness_5', 'gaussian_blur_5', 'gaussian_noise_5', 'motion_blur_5', 'shot_noise_5']
methods = ['rise', 'tis']
num_images = 10
base_path = 'results/metrics'

# Initialize dictionary to store average scores
results = {method: {condition: {metric: [] for metric in metrics} for condition in conditions} for method in methods}

# Process .npy files
for metric in metrics:
    for condition in conditions:
        for method in methods:
            for img_idx in range(num_images):
                # Construct file name (e.g., clean_0_vit_b16_rise.npy)
                file_name = f"{condition}_{img_idx}_vit_b16_{method}.npy"
                file_path = os.path.join(base_path, metric, file_name)
                
                if os.path.exists(file_path):
                    score = np.load(file_path)
                    # Ensure score is a scalar (mean if array)
                    score = np.mean(score) if score.size > 1 else score.item()
                    results[method][condition][metric].append(score)
                else:
                    print(f"File not found: {file_path}")
                    results[method][condition][metric].append(np.nan)

# Compute average scores across images
avg_results = {
    method: {
        condition: {metric: np.nanmean(results[method][condition][metric]) for metric in metrics}
        for condition in conditions
    } for method in methods
}

# Create summary table
table_data = []
for condition in conditions:
    row = {'Condition': condition.replace('_5', '')}
    for method in methods:
        for metric in metrics:
            score = avg_results[method][condition][metric]
            row[f"{method.upper()} {metric.capitalize()}"] = f"{score:.4f}" if not np.isnan(score) else "N/A"
    table_data.append(row)

# Convert to DataFrame and format as Markdown
df = pd.DataFrame(table_data)
markdown_table = df.to_markdown(index=False, floatfmt=".4f")

# Save table to file
with open('results/summary_table.md', 'w') as f:
    f.write("# Summary of Average Metric Scores\n\n")
    f.write(markdown_table)

print("Summary table saved to results/summary_table.md")