import pandas as pd
import random
import math
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox

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


class ClusteringGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Customer Clustering Tool")
        self.master.geometry("800x600")
        
        self.create_widgets()
        self.df = None
        self.customer_ids = None
        
    def create_widgets(self):
        # File Selection
        self.file_frame = ttk.LabelFrame(self.master, text="File Selection")
        self.file_frame.pack(pady=10, padx=10, fill="x")
        
        self.btn_browse = ttk.Button(self.file_frame, text="Browse CSV", command=self.load_file)
        self.btn_browse.pack(side="left", padx=5)
        
        self.file_label = ttk.Label(self.file_frame, text="No file selected")
        self.file_label.pack(side="left", padx=5)
        
        # Clustering Parameters
        self.param_frame = ttk.LabelFrame(self.master, text="Clustering Parameters")
        self.param_frame.pack(pady=10, padx=10, fill="x")
        
        ttk.Label(self.param_frame, text="Number of Clusters (k):").grid(row=0, column=0, padx=5)
        self.k_entry = ttk.Entry(self.param_frame, width=10)
        self.k_entry.grid(row=0, column=1, padx=5)
        self.k_entry.insert(0, "5")
        
        # Data Fraction
        ttk.Label(self.param_frame, text="Data Fraction:").grid(row=0, column=2, padx=5)
        self.frac_entry = ttk.Entry(self.param_frame, width=5)
        self.frac_entry.grid(row=0, column=3, padx=5)
        self.frac_entry.insert(0, "1.0")
        
        self.btn_run = ttk.Button(self.param_frame, text="Run Clustering", command=self.run_clustering)
        self.btn_run.grid(row=0, column=4, padx=5)
        
        # Results Display
        self.results_frame = ttk.LabelFrame(self.master, text="Clustering Results")
        self.results_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.results_text = scrolledtext.ScrolledText(self.results_frame, height=20)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)
        
    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_label.config(text=file_path)
            try:
                self.df = pd.read_csv(file_path)
                messagebox.showinfo("Success", "File loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def run_clustering(self):
        if self.df is None:
            messagebox.showerror("Error", "Please select a CSV file first!")
            return
            
        try:
            k = int(self.k_entry.get())
            data_fraction = float(self.frac_entry.get())
            if k <= 0 or data_fraction <= 0 or data_fraction > 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Invalid input! Ensure:\n- k is a positive integer\n- Fraction is between 0 and 1")
            return
            
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Processing...\n")
        self.master.update_idletasks()
        
        try:
            df = self.df.copy()
            df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
            df = df.sample(frac=data_fraction).reset_index(drop=True)
            customer_ids = df['CustomerID']
            df = df.drop(columns=['CustomerID'])
            
            centroids_indices = random.sample(range(len(df)), k)
            centroids = df.iloc[centroids_indices].copy()
            clusters = [[] for _ in range(k)]
            
            # Clustering algorithm
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
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            for i in range(len(clusters)):
                cluster_ids = customer_ids.loc[clusters[i]].tolist()
                result = f"Cluster {i+1} IDs: {sorted(cluster_ids)}\n"
                self.results_text.insert(tk.END, result)
                
                numeric_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
                cluster = df.loc[clusters[i]]
                for column in numeric_columns:
                    outliers = find_outliers_iqr(cluster, column)
                    outlier_ids = customer_ids.loc[outliers.index].tolist()
                    if outlier_ids:
                        self.results_text.insert(tk.END, f"Outliers in {column}: {outlier_ids}\n")
                
                self.results_text.insert(tk.END, "\n")
            
            messagebox.showinfo("Success", "Clustering completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.results_text.delete(1.0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ClusteringGUI(root)
    root.mainloop()