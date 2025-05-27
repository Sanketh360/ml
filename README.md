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




# program 3 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load Iris dataset
iris = load_iris()
data = iris.data
labels = iris.target
label_names = iris.target_names

# Perform PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

# Plot the PCA-reduced data
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']

for i, label in enumerate(np.unique(labels)):
    plt.scatter(
        data_2d[labels == label, 0],  # PC1
        data_2d[labels == label, 1],  # PC2
        color=colors[i],
        label=label_names[label]
    )

plt.title('PCA on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()



# program 4


import pandas as pd

def find_s(file_path):
    data = pd.read_csv(file_path)
    print("Training Data:\n", data)

    hypothesis = ['?'] * (data.shape[1] - 1)

    for _, row in data.iterrows():
        if row.iloc[-1] == 'Yes':  # Only consider positive examples
            for i in range(len(hypothesis)):
                if hypothesis[i] == '?':
                    hypothesis[i] = row.iloc[i]
                elif hypothesis[i] != row.iloc[i]:
                    hypothesis[i] = '?'

    return hypothesis

# File path to your training data CSV
file_path = r'C:\Users\sanke\Documents\AI Lab Jupyter\training.csv'
final_hypothesis = find_s(file_path)
print("\nFinal Hypothesis:", final_hypothesis)




# program 6 


import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(x, xi, tau):
    return np.exp(- (x - xi)**2 / (2 * tau**2))

def predict(x, X, y, tau):
    m = len(X)
    
    weights = [gaussian_kernel(x, X[i], tau) for i in range(m)]
    W = np.diag(weights)
    
    X_bias = np.c_[np.ones(m), X]

    A = X_bias.T @ W @ X_bias
    b = X_bias.T @ W @ y
    theta = np.linalg.pinv(A) @ b

    return theta[0] + theta[1] * x

X = np.linspace(0, 2 * np.pi, 50)
y = np.sin(X) + 0.1 * np.random.randn(50)

x_test = np.linspace(0, 2 * np.pi, 200)
tau = 0.3

y_pred = [predict(xi, X, y, tau) for xi in x_test]

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='red', label='Data Points')
plt.plot(x_test, y_pred, color='blue', label='LWR Fit')
plt.title('Locally Weighted Regression (Easy Version)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()



# program 8


import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


data = load_breast_cancer()
X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")


rules = export_text(model, feature_names=data.feature_names.tolist())
print("\nDecision Tree Rules:\n", rules)

# 6. Predict a new case
sample = X_test[0].reshape(1, -1)
pred = model.predict(sample)[0]
print("\nPrediction for Sample 0:", "Malignant" if pred == 0 else "Benign")




# program 9

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = data.data, data.target

# 2. Split into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train the Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# 4. Predict on test data
y_pred = model.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 6. Show 15 test images with predictions
fig, axes = plt.subplots(3, 5, figsize=(12, 6))
for ax, img, true, pred in zip(axes.ravel(), X_test, y_test, y_pred):
    ax.imshow(img.reshape(64, 64), cmap='gray')
    ax.set_title(f"T:{true} P:{pred}")
    ax.axis('off')
plt.tight_layout()
plt.show()
