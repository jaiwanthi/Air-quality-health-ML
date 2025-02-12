import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = 'D:/project/air quality reduced.xlsx'  # Replace with your actual file path
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Select relevant features for clustering
features = ['location_id', 'age_id', 'sex_id', 'val']
df_filtered = df[features].dropna()  # Remove missing values

# Standardize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_filtered)

# 🔹 1️⃣ Elbow Method to determine optimal K for K-Means
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), wcss, marker='o', linestyle='--')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# 🔹 2️⃣ Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_filtered['KMeans_Cluster'] = kmeans.fit_predict(df_scaled)

# 🔹 3️⃣ Apply DBSCAN Clustering
dbscan = DBSCAN(eps=1.5, min_samples=5)
df_filtered['DBSCAN_Cluster'] = dbscan.fit_predict(df_scaled)

# 🔹 4️⃣ Apply Gaussian Mixture Model (GMM) Clustering
gmm = GaussianMixture(n_components=3, random_state=42)
df_filtered['GMM_Cluster'] = gmm.fit_predict(df_scaled)

# 🔹 5️⃣ Heatmap: Regional Health Impact Over Years
plt.figure(figsize=(12, 6))
heatmap_data = df.pivot_table(values='val', index='location_id', columns='year', aggfunc='mean')
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".1f")
plt.title("Regional Health Impact Intensity Over Years")
plt.xlabel("Year")
plt.ylabel("Location ID")
plt.show()

# 🔹 6️⃣ Bar Chart: Health Impact by Age and Gender
plt.figure(figsize=(10, 6))
sns.barplot(x='sex_id', y='val', hue='age_id', data=df, ci=None)
plt.title("Health Impact by Gender and Age Group")
plt.xlabel("Gender (1 = Male, 2 = Female)")
plt.ylabel("Health Impact Value (val)")
plt.legend(title="Age Group")
plt.show()

# 🔹 7️⃣ K-Means Cluster Map: Regional Health Impact
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_filtered, x='location_id', y='val', hue='KMeans_Cluster', palette='viridis', alpha=0.7)
plt.title("K-Means Cluster Map: Regional Health Impact")
plt.xlabel("Location ID")
plt.ylabel("Health Impact Value (val)")
plt.legend(title="KMeans Cluster")
plt.show()

# 🔹 8️⃣ DBSCAN Cluster Map: Regional Health Impact
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_filtered, x='location_id', y='val', hue='DBSCAN_Cluster', palette='coolwarm', alpha=0.7)
plt.title("DBSCAN Cluster Map: Regional Health Impact")
plt.xlabel("Location ID")
plt.ylabel("Health Impact Value (val)")
plt.legend(title="DBSCAN Cluster")
plt.show()

# 🔹 9️⃣ GMM Cluster Map: Regional Health Impact
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_filtered, x='location_id', y='val', hue='GMM_Cluster', palette='Set1', alpha=0.7)
plt.title("GMM Cluster Map: Regional Health Impact")
plt.xlabel("Location ID")
plt.ylabel("Health Impact Value (val)")
plt.legend(title="GMM Cluster")
plt.show()
