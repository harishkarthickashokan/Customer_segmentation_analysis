import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset with whitespace separator
df = pd.read_csv("customers.csv", sep=r'\s+')

print("--- Dataset Loaded ---")
print(df.head())
print("\nColumns:", df.columns)

# Check required columns exist
required_cols = ['AnnualIncome', 'SpendingScore']

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}. Found: {df.columns}")

# Select features
X = df[['AnnualIncome', 'SpendingScore']]

# KMeans
kmeans = KMeans(n_clusters=4, random_state=0)
df['Cluster'] = kmeans.fit_predict(X)

# Plot clusters
plt.scatter(X['AnnualIncome'], X['SpendingScore'], c=df['Cluster'])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")
plt.show()

print("\n--- Cluster Results ---")
print(df[['CustomerID', 'AnnualIncome', 'SpendingScore', 'Cluster']])
