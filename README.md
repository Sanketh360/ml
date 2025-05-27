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
plt.tight_layout(); plt.show()
plt.figure(figsize=(15, 5 * rows))
for i, col in enumerate(features, 1):
    plt.subplot(rows, cols, i)
    sns.boxplot(x=df[col], color='orange')
    plt.title(f'{col} Box Plot')
plt.tight_layout(); plt.show()
print("Outliers (IQR method):")
for col in features:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
    print(f"{col}: {len(outliers)}")
print("\nSummary:")
print(df.describe())

