# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the Data Import the dataset to start the clustering analysis process.

2.Explore the Data Analyze the dataset to understand distributions, patterns, and key characteristics.

3.Select Relevant Features Identify the most informative features to improve clustering accuracy and relevance.

4.Preprocess the Data Clean and scale the data to prepare it for clustering.

5.Determine Optimal Number of Clusters Use techniques like the elbow method to find the ideal number of clusters.

6.Train the Model with K-Means Clustering Apply the K-Means algorithm to group data points into clusters based on similarity.

7.Analyze and Visualize Clusters Examine and visualize the resulting clusters to interpret patterns and relationships.
 

## Program:
```
/*
Program to implement customer segmentation using K-Means clustering on the Mall Customers dataset.
Developed by: P.Bhavankumar
RegisterNumber: 212225240026
*/
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")
data = pd.read_csv('CustomerData.csv')
print(data.head())
print(data.columns)
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
inertia_values = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)  # Explicit n_init to suppress warning
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia_values, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)  # Explicit n_init
kmeans.fit(X_scaled)
data['Cluster'] = kmeans.labels_
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=data,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='viridis',
    s=100,
    alpha=0.7
)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 1], centers[:, 2], c='red', s=200, alpha=0.75, marker='X', label='Centroids')

plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
```

## Output:

<img width="837" height="217" alt="Screenshot 2026-03-24 101827" src="https://github.com/user-attachments/assets/b0759d5a-f851-47e7-ae4b-8e8c06c10eee" />

<img width="940" height="469" alt="Screenshot 2026-03-24 101839" src="https://github.com/user-attachments/assets/fdab4d3c-e783-4fd1-b662-c38ccde0c473" />

<img width="909" height="505" alt="Screenshot 2026-03-24 101852" src="https://github.com/user-attachments/assets/7387403a-dea3-46dc-9e23-32795a580ef8" />


## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
