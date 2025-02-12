# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 08:17:47 2025

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset
file_path = 'D:/project/air quality reduced.xlsx'
df = pd.read_excel(file_path)

### 1️⃣ Exploratory Data Analysis (EDA)
print("Dataset Preview:\n", df.head())
print("\nDataset Information:\n")
df.info()
print("\nStatistical Summary:\n", df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values per Column:\n", missing_values)

# Visualize missing values
plt.figure(figsize=(8, 5))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

### 2️⃣ Data Preprocessing
# Fill missing values with median
df.fillna(df.select_dtypes(include=[np.number]).median(), inplace=True)

# Select relevant features for scaling
features = ['location_id', 'age_id', 'sex_id', 'val']
df_filtered = df[features].copy()

# Min-Max Normalization (0-1 Scaling)
minmax_scaler = MinMaxScaler()
df_normalized = pd.DataFrame(minmax_scaler.fit_transform(df_filtered), columns=features)
df_normalized['year'] = df['year']

# Standardization (Mean = 0, Std = 1)
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df_filtered), columns=features)
df_standardized['year'] = df['year']

# Save processed data
output_file = 'D:/project/processed_data_1.xlsx'
with pd.ExcelWriter(output_file) as writer:
    df_normalized.to_excel(writer, sheet_name="Normalized Data", index=False)
    df_standardized.to_excel(writer, sheet_name="Standardized Data", index=False)
print(f"✅ Processed data saved to: {output_file}")

### 3️⃣ Histograms and Pairplots
# Histograms of numerical features
df.hist(figsize=(10, 8), bins=20)
plt.suptitle('Histograms of Numerical Features')
plt.tight_layout()
plt.show()

# Downsample the data for visualization
df_sampled = df.sample(n=1000, random_state=42)
subset_columns = df_sampled.select_dtypes(include=['number']).columns[:5]

# Pairplot
sns.pairplot(df_sampled[subset_columns])
plt.suptitle('Pairplot of Sampled Numerical Features', y=1.02)
plt.show()

### 4️⃣ Clustering: K-Means, DBSCAN, GMM
# Elbow Method for K-Means
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(df_standardized[features])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), wcss, marker='o', linestyle='--')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_filtered['KMeans_Cluster'] = kmeans.fit_predict(df_standardized[features])

# Apply DBSCAN Clustering
dbscan = DBSCAN(eps=1.5, min_samples=5)
df_filtered['DBSCAN_Cluster'] = dbscan.fit_predict(df_standardized[features])

# Apply Gaussian Mixture Model (GMM) Clustering
gmm = GaussianMixture(n_components=3, random_state=42)
df_filtered['GMM_Cluster'] = gmm.fit_predict(df_standardized[features])

### 5️⃣ Clustering Visualizations
# Heatmap: Regional Health Impact Over Years
plt.figure(figsize=(12, 6))
heatmap_data = df.pivot_table(values='val', index='location_id', columns='year', aggfunc='mean').fillna(0)
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".1f")
plt.title("Regional Health Impact Intensity Over Years")
plt.xlabel("Year")
plt.ylabel("Location ID")
plt.show()

# Scatterplot for K-Means Clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_filtered, x='location_id', y='val', hue='KMeans_Cluster', palette='viridis', alpha=0.7)
plt.title("K-Means Cluster Map: Regional Health Impact")
plt.xlabel("Location ID")
plt.ylabel("Health Impact Value (val)")
plt.legend(title="KMeans Cluster")
plt.show()


# 🔹 Bar Chart: Health Impact by Age and Gender
plt.figure(figsize=(10, 6))
sns.barplot(x='sex_id', y='val', hue='age_id', data=df, ci=None)
plt.title("Health Impact by Gender and Age Group")
plt.xlabel("Gender (1 = Male, 2 = Female)")
plt.ylabel("Health Impact Value (val)")
plt.legend(title="Age Group")
plt.show()

# 🔹 K-Means Cluster Map
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_filtered, x='location_id', y='val', hue='KMeans_Cluster', palette='viridis', alpha=0.7)
plt.title("K-Means Cluster Map: Regional Health Impact")
plt.xlabel("Location ID")
plt.ylabel("Health Impact Value (val)")
plt.legend(title="KMeans Cluster")
plt.show()

# 🔹 DBSCAN Cluster Map
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_filtered, x='location_id', y='val', hue='DBSCAN_Cluster', palette='coolwarm', alpha=0.7)
plt.title("DBSCAN Cluster Map: Regional Health Impact")
plt.xlabel("Location ID")
plt.ylabel("Health Impact Value (val)")
plt.legend(title="DBSCAN Cluster")
plt.show()

# 🔹 GMM Cluster Map
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_filtered, x='location_id', y='val', hue='GMM_Cluster', palette='Set1', alpha=0.7)
plt.title("GMM Cluster Map: Regional Health Impact")
plt.xlabel("Location ID")
plt.ylabel("Health Impact Value (val)")
plt.legend(title="GMM Cluster")
plt.show()