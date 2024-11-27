# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.exceptions import NotFittedError
from sklearn.cluster import KMeans
import seaborn as sns

# 1. Create a Custom Estimator for SOM
class SOMWrapper(BaseEstimator, ClusterMixin):
    def __init__(self, map_size_x=10, map_size_y=10, sigma=1.0, learning_rate=0.5, num_iterations=1000, random_seed=42, n_clusters=5):
        """
        Initializes the SOM Wrapper.

        Parameters:
        - map_size_x (int): Width of the SOM grid.
        - map_size_y (int): Height of the SOM grid.
        - sigma (float): Radius of the neighborhood function.
        - learning_rate (float): Initial learning rate.
        - num_iterations (int): Number of iterations for training.
        - random_seed (int): Seed for reproducibility.
        - n_clusters (int): Number of clusters for K-Means.
        """
        self.map_size_x = map_size_x
        self.map_size_y = map_size_y
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.random_seed = random_seed
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        """
        Fits the SOM to the data and clusters the neuron weights.

        Parameters:
        - X (array-like): Input data.
        - y: Ignored.

        Returns:
        - self
        """
        # Initialize MiniSom
        self.som_ = MiniSom(
            x=self.map_size_x,
            y=self.map_size_y,
            input_len=X.shape[1],
            sigma=self.sigma,
            learning_rate=self.learning_rate,
            random_seed=self.random_seed
        )

        # Train SOM
        self.som_.random_weights_init(X)
        self.som_.train_random(X, self.num_iterations)

        # Extract neuron weights
        self.weights_ = self.som_.get_weights().reshape(self.map_size_x * self.map_size_y, X.shape[1])

        # Cluster neuron weights using K-Means
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_seed)
        self.kmeans_.fit(self.weights_)
        self.cluster_centers_ = self.kmeans_.cluster_centers_
        self.labels_neurons_ = self.kmeans_.labels_

        # Assign each neuron to a cluster
        self.neuron_to_cluster_map_ = {i: self.labels_neurons_[i] for i in range(len(self.labels_neurons_))}

        return self

    def predict(self, X):
        """
        Assigns each sample to a cluster based on the closest neuron.

        Parameters:
        - X (array-like): Input data.

        Returns:
        - labels (array): Cluster labels for each sample.
        """
        if not hasattr(self, 'som_'):
            raise NotFittedError("This SOMWrapper instance is not fitted yet.")

        # Find the BMU for each sample
        bmu_indices = np.array([self.som_.winner(x) for x in X])
        # Convert 2D BMU indices to 1D indices
        bmu_flat_indices = bmu_indices[:, 0] * self.map_size_y + bmu_indices[:, 1]
        # Map BMU indices to cluster labels
        labels = np.array([self.neuron_to_cluster_map_[idx] for idx in bmu_flat_indices])

        return labels

# 2. Define a Custom Scoring Function
def som_clustering_scorer(estimator, X, y):
    """
    Custom scoring function for SOM clustering that aligns cluster labels with true labels
    using the Hungarian algorithm and computes the F1-Score.

    Parameters:
    - estimator: The clustering estimator (SOMWrapper).
    - X: Feature data.
    - y: True labels.

    Returns:
    - score: The F1-Score after label alignment.
    """
    try:
        # Fit the estimator
        estimator.fit(X)
        # Predict cluster labels
        labels = estimator.predict(X)
    except Exception as e:
        # Handle any exception during fitting/predicting
        print(f"Error during fitting/predicting: {e}")
        return -np.inf  # Assign a very low score to penalize this combination

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y, labels)

    # Use Hungarian algorithm for label alignment
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)

    # Create label mapping
    label_mapping = {pred_label: true_label for pred_label, true_label in zip(col_ind, row_ind)}

    # Align predicted labels
    y_pred_aligned = np.array([label_mapping.get(label, -1) for label in labels])

    # Compute the desired metric (F1-Score)
    score = f1_score(y, y_pred_aligned, average='weighted', zero_division=0)

    return score

# 3. Load and Prepare the Dataset
# Load the dataset
data = pd.read_csv("./train_pca.csv")

# Separate features (X) and labels (y)
# Replace 'type' with the actual name of your label column if different
X = data.drop(columns=['type'])
y = data['type']

# Encode string labels into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Set Up the Parameter Grid for GridSearchCV
param_grid = {
    'map_size_x': [13],
    'map_size_y': [6],
    'sigma': [1.9, 2.0, 2.1],
    'learning_rate': [0.1475, 0.15, 0.1525],
    'num_iterations': [875, 900, 925],
    'n_clusters': [5]
}

# 5. Initialize the Custom SOM Estimator
som_wrapper = SOMWrapper(random_seed=42)

# 6. Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=som_wrapper,
    param_grid=param_grid,
    scoring=som_clustering_scorer,  # Pass the custom scoring function directly
    cv=3,  # Number of cross-validation folds; adjust as needed
    verbose=2,
    n_jobs=-1  # Use all available cores
)

# 7. Perform Grid Search to Find the Best Hyperparameters
print("Starting Grid Search for SOM hyperparameters...")
grid_search.fit(X_scaled, y_encoded)
print("Grid Search completed.")

# Display the best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# 8. Fit the Best Model with the Best Parameters
best_som = grid_search.best_estimator_
best_som.fit(X_scaled)
y_pred = best_som.predict(X_scaled)

# 9. Align Predicted Labels with True Labels Using Hungarian Algorithm
conf_matrix = confusion_matrix(y_encoded, y_pred)
row_ind, col_ind = linear_sum_assignment(-conf_matrix)
label_mapping = {pred_label: true_label for pred_label, true_label in zip(col_ind, row_ind)}
y_pred_aligned = np.array([label_mapping.get(label, -1) for label in y_pred])

# 10. Evaluation Metrics
accuracy = accuracy_score(y_encoded, y_pred_aligned)
precision = precision_score(y_encoded, y_pred_aligned, average='weighted', zero_division=0)
recall = recall_score(y_encoded, y_pred_aligned, average='weighted', zero_division=0)
f1 = f1_score(y_encoded, y_pred_aligned, average='weighted', zero_division=0)
ari = adjusted_rand_score(y_encoded, y_pred)
nmi = normalized_mutual_info_score(y_encoded, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')
print(f'Adjusted Rand Index: {ari:.2f}')
print(f'Normalized Mutual Information: {nmi:.2f}')

# 11. Plot the Clustered Data
# Note: This visualization assumes the data has two features for 2D plotting
if X.shape[1] == 2:
    plt.figure(figsize=(8, 6))
    unique_labels = set(y_pred_aligned)
    colors = plt.cm.get_cmap('viridis', len(unique_labels))
    for label in unique_labels:
        color = colors(label)
        label_name = f'Cluster {label}'
        plt.scatter(
            X_scaled[y_pred_aligned == label, 0],
            X_scaled[y_pred_aligned == label, 1],
            s=40,
            color=color,
            label=label_name,
            alpha=0.6,
            edgecolors='w',
            linewidth=0.5
        )
    plt.title('SOM Clustering with Best Parameters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
else:
    print("The data has more than two features; skipping 2D plot.")

# 12. Plot the SOM Map
# Visualize the SOM distance map and cluster assignments

# Retrieve the trained SOM
som = best_som.som_

# Retrieve the cluster assignments for each neuron
labels_neurons = best_som.labels_neurons_
map_size_x = best_som.map_size_x
map_size_y = best_som.map_size_y

# Create a grid for plotting
plt.figure(figsize=(10, 10))
# Plot the U-Matrix (distance map)
distance_map = som.distance_map().T  # Transpose for correct orientation
sns.heatmap(distance_map, cmap='bone_r', square=True, cbar=True, linewidths=.5)

# Overlay cluster assignments
for i in range(map_size_x):
    for j in range(map_size_y):
        neuron_idx = i * map_size_y + j
        cluster = labels_neurons[neuron_idx]
        plt.text(j + 0.5, i + 0.5, str(cluster),
                 ha='center', va='center',
                 color='red', fontsize=12, fontweight='bold')

plt.title('SOM Distance Map with Cluster Assignments')
plt.xlabel('SOM X-axis')
plt.ylabel('SOM Y-axis')
plt.show()

# 13. (Optional) Plot Data Points on SOM Grid
# Visualize how data points are distributed across the SOM grid based on cluster assignments

# Get the positions of the neurons
neuron_positions = {}
for idx, position in enumerate(som.get_weights()):
    x, y = som.winner(position)
    neuron_positions[idx] = (x, y)

# Map each data point to its neuron position
data_positions = np.array([som.winner(x) for x in X_scaled])

# Assign cluster labels to each data point based on their BMU's cluster
data_clusters = np.array([labels_neurons[x * map_size_y + y] for x, y in data_positions])

# Plot the SOM grid with data points
plt.figure(figsize=(10, 10))
# Plot the U-Matrix
sns.heatmap(distance_map, cmap='bone_r', square=True, cbar=True, linewidths=.5)

# Scatter plot of data points
plt.scatter(
    data_positions[:, 1] + 0.5,  # X-axis
    data_positions[:, 0] + 0.5,  # Y-axis
    c=data_clusters,
    cmap='viridis',
    alpha=0.6,
    edgecolor='k',
    s=50
)

plt.title('SOM Distance Map with Data Point Clusters')
plt.xlabel('SOM X-axis')
plt.ylabel('SOM Y-axis')
plt.show()