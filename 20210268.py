import pandas as pd
import random
import math

def distance(row, centroid):
    return math.sqrt(sum((feature1 - feature2) ** 2 for feature1, feature2 in zip(row, centroid)))

def assign_to_cluster(row, centroids, clusters, prev_assignments):
    min_distance = float('inf')
    min_cluster = None
    for i in range(len(centroids)):
        centroid = centroids.iloc[i].to_numpy()
        distance_to_centroid = distance(row.to_numpy(), centroid)
        if distance_to_centroid < min_distance:
            min_distance = distance_to_centroid
            min_cluster = i
    
    current_assignment = min_cluster
    prev_assignment = prev_assignments.get(row.name, None)
    
    clusters[current_assignment].append(row.name)
    
    if current_assignment != prev_assignment:
        prev_assignments[row.name] = current_assignment
        return True  # Assignment changed
    return False  # Assignment did not change

def update_centroids(df, clusters, centroids):
    for i in range(len(centroids)):
        cluster = clusters[i]
        if len(cluster) > 0:
            new_centroid = df.loc[cluster].mean()
            centroids.iloc[i] = new_centroid

def find_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def main():
    path = "SS2025_Clustering_SuperMarketCustomers.csv"
    df = pd.read_csv(path)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    data_percentage = 1
    df = df.sample(frac=data_percentage).reset_index(drop=True)
    customer_ids = df['CustomerID']
    df = df.drop(columns=['CustomerID'])
    
    # numerical_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    # all_outliers = set()

    # for col in numerical_columns:
    #     outliers = find_outliers_iqr(df, col)
    #     print(f"Outliers in {col}: {outliers.index.tolist()}")
    #     all_outliers.update(outliers.index.tolist())  

    # df_clean = df[~df.index.isin(all_outliers)]
    
    k = 5
    random.seed(42)
    centroids_indices = random.sample(range(len(df)), k)
    centroids = df.iloc[centroids_indices].copy()
    clusters = [[] for _ in range(k)]
    
    
    prev_assignments = {}
    is_converged = False
    
    while not is_converged:
        is_converged = True
        clusters = [[] for _ in range(k)]
        
        for i in range(len(df)):
            row = df.iloc[i]
            if assign_to_cluster(row, centroids, clusters, prev_assignments):
                is_converged = False
        
        update_centroids(df, clusters, centroids)
    
    for i in range(len(clusters)):
        cluster_ids = customer_ids.loc[clusters[i]].tolist()
        print(f"Cluster {i+1} IDs: {sorted(cluster_ids)}")
        
        numeric_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        cluster = df.loc[clusters[i]]
        for column in numeric_columns:
            outliers = find_outliers_iqr(cluster, column)
            # print(f"Outliers in {column}: {outliers['CustomerID'].tolist()}")
            print(f"Outliers in {column}: {customer_ids.loc[outliers.index].tolist()}")
            
        print()

if __name__ == "__main__":
    main()