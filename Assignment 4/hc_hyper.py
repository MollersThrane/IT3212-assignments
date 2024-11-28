import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
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

# 1. Create a Custom Estimator for AgglomerativeClustering
class AgglomerativeClusteringWrapper(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=2, linkage='ward', metric='euclidean'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric

    def fit(self, X, y=None):
        # Handle linkage and metric compatibility
        if self.linkage == 'ward' and self.metric != 'euclidean':
            raise ValueError("Ward linkage requires 'euclidean' metric.")
        
        self.clusterer_ = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
            metric=self.metric if self.linkage != 'ward' else 'euclidean'
        )
        self.clusterer_.fit(X)
        return self

    def predict(self, X):
        if not hasattr(self, 'clusterer_'):
            raise NotFittedError("This AgglomerativeClusteringWrapper instance is not fitted yet.")
        return self.clusterer_.labels_

# 2. Define a Custom Scoring Function
def hierarchical_clustering_scorer(estimator, X, y):
    """
    Custom scoring function for Hierarchical Clustering that aligns cluster labels with true labels
    using the Hungarian algorithm and computes the F1-Score.

    Parameters:
    - estimator: The clustering estimator (AgglomerativeClusteringWrapper).
    - X: Feature data.
    - y: True labels.

    Returns:
    - score: The F1-Score after label alignment.
    """
    try:
        # Fit the model
        estimator.fit(X)
        labels = estimator.predict(X)
    except ValueError as e:
        # Handle incompatible parameter combinations
        print(f"Parameter combination error: {e}")
        return -np.inf  # Assign a very low score to penalize this combination

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y, labels)

    # Handle cases where number of clusters is less than true labels
    unique_pred_clusters = np.unique(labels)
    unique_true_labels = np.unique(y)
    n_pred_clusters = len(unique_pred_clusters)
    n_true_labels = len(unique_true_labels)

    if n_pred_clusters < n_true_labels:
        # Not enough clusters to match all true labels
        return 0
    elif n_pred_clusters > n_true_labels:
        # More clusters than true labels; proceed without modification
        pass  # Optionally, implement a penalty for over-clustering

    # Use Hungarian algorithm for label alignment
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)

    # Create label mapping
    label_mapping = {pred_label: true_label for pred_label, true_label in zip(col_ind, row_ind)}

    # Align predicted labels
    y_pred_aligned = np.array([label_mapping.get(label, -1) for label in labels])

    # Compute the desired metric (e.g., F1-Score)
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

# 4. Set Up the Parameter Grid for GridSearchCV
param_grid = {
    'n_clusters': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],  # Adjust based on expected number of clusters
    'linkage': ['complete', 'average', 'single'],
    'metric': ['euclidean', 'manhattan', 'cosine', 'l1', 'l2']  # 'ward' only supports 'euclidean'
}

# 5. Initialize the Custom AgglomerativeClustering Estimator
agg_clustering_wrapper = AgglomerativeClusteringWrapper()

# 6. Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=agg_clustering_wrapper,
    param_grid=param_grid,
    scoring=hierarchical_clustering_scorer,  # Pass the custom scoring function directly
    cv=3,  # Number of cross-validation folds; adjust as needed
    verbose=2,
    n_jobs=-1  # Use all available cores
)

# 7. Perform Grid Search to Find the Best Hyperparameters
print("Starting Grid Search for AgglomerativeClustering hyperparameters...")
grid_search.fit(X, y_encoded)
print("Grid Search completed.")

# Display the best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# 8. Fit the Best Model with the Best Parameters
best_agg = grid_search.best_estimator_
best_agg.fit(X)
y_pred = best_agg.predict(X)

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
        # AgglomerativeClustering does not handle noise, so no label should be -1
        color = colors(label)
        label_name = f'Cluster {label}'
        plt.scatter(
            X[y_pred_aligned == label, 0],
            X[y_pred_aligned == label, 1],
            s=40,
            color=color,
            label=label_name,
            alpha=0.6,
            edgecolors='w',
            linewidth=0.5
        )
    plt.title('Hierarchical Clustering with Best Parameters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
else:
    print("The data has more than two features; skipping 2D plot.")
