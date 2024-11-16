import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Plot 2D projection from a .xvg file.')
parser.add_argument('filename', type=str, help='Path to the .xvg file')
args = parser.parse_args()

# Manually parse the file to skip comment lines and only get numeric data
data = []
with open(args.filename) as file:
    for line in file:
        if line.startswith(('@', '#')):  # Skip comment lines
            continue
        parts = line.split()
        if len(parts) >= 2:  # Ensure there are enough columns
            try:
                data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                pass  # Skip lines with non-numeric data

# Convert to a DataFrame
df = pd.DataFrame(data, columns=["PC1", "PC2"])

# Plotting
plt.figure(figsize=(8, 6))
sns.scatterplot(x="PC1", y="PC2", data=df)
plt.xlabel("Projection on eigenvector 1 (nm)")
plt.ylabel("Projection on eigenvector 2 (nm)")
plt.title("2D projection of trajectory")
plt.show()
