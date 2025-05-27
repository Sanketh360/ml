# ml

# program 1

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True).frame

features = df.select_dtypes(include=[np.number]).columns

n = len(features)

rows, cols = -(-n // 3), 3

plt.figure(figsize=(15, 5 * rows))

for i, col in enumerate(features, 1):
    
    plt.subplot(rows, cols, i)
    
    sns.histplot(df[col], kde=True, bins=30, color='blue')
    
    plt.title(f'{col} Distribution')

plt.tight_layout()

plt.show()

plt.figure(figsize=(15, 5 * rows))

for i, col in enumerate(features, 1):
    
    plt.subplot(rows, cols, i)
    
    sns.boxplot(x=df[col], color='orange')
    
    plt.title(f'{col} Box Plot')

plt.tight_layout()

plt.show()

print("Outliers (IQR method):")

for col in features:
    
    q1, q3 = df[col].quantile([0.25, 0.75])
    
    iqr = q3 - q1
    
    outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
    
    print(f"{col}: {len(outliers)}")

print("\nSummary:")

print(df.describe())







#p1


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load dataset
df = fetch_california_housing(as_frame=True).frame

# Select numerical features
features = df.select_dtypes(include=[np.number]).columns

# Calculate layout for histograms
n = len(features)
rows, cols = -(-n // 3), 3  # Ceiling division to get rows needed for 3 columns

# Plot histograms of all numerical features in a grid
df[features].hist(bins=30, figsize=(15, 5 * rows), color='blue', layout=(rows, cols))
plt.suptitle("Histograms of Features", fontsize=16)
plt.tight_layout()
plt.show()

# Plot individual boxplots for each feature (one plot per figure for clarity)
for col in features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col], color='orange')
    plt.title(f'{col} Box Plot')
    plt.show()

# Outlier detection using IQR method and print counts
print("Outliers (IQR method):")
for col in features:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)}")

# Print summary statistics of dataset
print("\nSummary:")
print(df.describe())



# program 2

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset as a DataFrame
data = fetch_california_housing(as_frame=True).frame

# Plot correlation matrix using a heatmap
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Plot pairwise relationships between features
sns.pairplot(data, diag_kind='kde')
plt.show()
