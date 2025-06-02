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







# p1


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.datasets import fetch_california_housing

# Load dataset
# df = fetch_california_housing(as_frame=True).frame
df = pd.read_csv("california_housing.csv")

# Select numerical features
features = df.select_dtypes(include=[np.number]).columns

# Calculate layout for histograms
n = len(features)
rows, cols = -(-n // 3), 3  # Ceiling division to get rows needed for 3 columns

# Plot histograms of all numerical features in a grid
df[features].hist(bins=30, figsize=(15, 5 * rows), color='blue', edgecolor = "black " ,layout=(rows, cols))
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
# from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset as a DataFrame
# data = fetch_california_housing(as_frame=True).frame
data = pd.read_csv("california_housing.csv")
print(data.head())
correlation_matrix = data.corr()
print("\n Correlation Martix")
print(correlation_matrix)

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



# pro 3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load dataset from CSV
iris = pd.read_csv("iris_dataset.csv")

# Separate features and labels
data = iris.iloc[:, :-1]
labels = iris.iloc[:, -1]

# Perform PCA
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)


colors = {0: 'red', 1: 'green', 2: 'blue'}
label_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

plt.figure(figsize=(8, 6))
for label in labels.unique():
    plt.scatter(
        data_2d[labels == label, 0],
        data_2d[labels == label, 1],
        color=colors[label],
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


# program 5 


import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Set seed and generate data
np.random.seed(0)
data = np.random.rand(100)

# Prepare training and test sets
train_x = data[:50].reshape(-1, 1)
train_y = ["Class1" if x <= 0.5 else "Class2" for x in data[:50]]
test_x = data[50:].reshape(-1, 1)

# Try different k values
k_values = [1, 2, 3, 4, 5, 20, 30]

for k in k_values:
    print(f"\n--- Results for k = {k} ---")
    
    # Initialize and train the KNN classifier
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_x, train_y)
    
    # Predict on test data
    predictions = model.predict(test_x)
    
    for i, (x_val, pred) in enumerate(zip(test_x.ravel(), predictions), start=51):
        print(f"x{i} = {x_val:.3f} → {pred}")






# pro 6


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("california_housing.csv").head(4000)
X = data['AveRooms'].values
y = data['HouseAge'].values
def predict(x0,x,y,tau):
    w=np.diag(np.exp(-(x-x0)**2/(2*tau**2)))
    xi=np.c_[np.ones(len(x)),x]
    theta = np.linalg.pinv(xi.T@ w @ xi)@(xi.T @ w @ y)
    return theta[0]+theta[1]*x0
    
tau=0.3
x_test=np.linspace(X.min(),X.max(),50)
y_test = [predict(i,X,y,tau) for i in x_test]

plt.scatter(X,y,color="red",label="Data",alpha = 0.5)
plt.plot(x_test,y_test,color="blue",label="lwr")
plt.title("lwr")
plt.legend()
plt.grid(True)
plt.show()

    






# program 7


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

data = pd.read_csv("california_housing.csv")
x=data[["AveRooms"]]
y=data["MedHouseVal"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

plt.scatter(x_test,y_test,color="skyblue",edgecolor="blue",label="Actual")
plt.scatter(x_test,y_pred,color="orange",edgecolor="red",label="linear")
plt.title("linear Regression")
plt.grid(True)
plt.legend()
plt.show()

poly=PolynomialFeatures(degree=3)
x_train_poly=poly.fit_transform(x_train)
x_test_poly=poly.transform(x_test)

model.fit(x_train_poly,y_train)
y_pred_poly = model.predict(x_test_poly)

plt.scatter(x_test,y_test,color="lightgreen",edgecolor="green",label="Actual")
plt.scatter(x_test,y_pred_poly,color="orange",edgecolor="red",label="poly")
plt.title("poly Regression")
plt.grid(True)
plt.legend()
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



# program 8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_text
from sklearn.metrics import accuracy_score

data = pd.read_csv("breast_cancer.csv")
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model=DecisionTreeClassifier(max_depth=4,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"accurasy score= {accuracy:.2f}\n")

rules=export_text(model,feature_names=list(x.columns))
print("Decission tree:",rules)

sample = x_test.iloc[[0]]
pred=model.predict(sample)
print("prediction for sample =" , "maligent"if pred == 0 else "Benign")



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






# program 10

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. Load the dataset from CSV file
data = pd.read_csv("breast_cancer.csv")

# 2. Separate features and (optional) labels
X = data.iloc[:, :-1]   # All columns except the last (features)
y = data.iloc[:, -1]    # Last column (actual labels – not used in clustering)

# 3. Standardize the feature values (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply K-Means Clustering with 2 clusters (malignant & benign)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_scaled)
labels = kmeans.labels_   # Predicted cluster for each data point

# 5. Use PCA to reduce dimensions to 2 for plotting
pca = PCA(n_components=2)
X_2D = pca.fit_transform(X_scaled)

# 6. Visualize the clustered data
plt.figure(figsize=(8, 6))
plt.scatter(X_2D[:, 0], X_2D[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
plt.title("K-Means Clustering (Breast Cancer Data)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster (0 or 1)")
plt.grid(True)
plt.show()




