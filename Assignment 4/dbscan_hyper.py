import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClusterMixin

# 1. Create a Custom Estimator for DBSCAN
class DBSCANWrapper(BaseEstimator, ClusterMixin):
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
    
    def fit(self, X, y=None):
        self.dbscan_ = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        self.dbscan_.fit(X)
        return self
    
    def predict(self, X):
        # DBSCAN does not have a predict method; return labels from the fitted model
        return self.dbscan_.labels_

# 2. Define a Custom Scoring Function
def clustering_scorer(estimator, X, y):
    """
    Custom scoring function for clustering that aligns cluster labels with true labels
    using the Hungarian algorithm and computes the F1-Score.
    
    Parameters:
    - estimator: The clustering estimator (DBSCANWrapper).
    - X: Feature data.
    - y: True labels.
    
    Returns:
    - score: The F1-Score after label alignment.
    """
    # Fit the model
    estimator.fit(X)
    labels = estimator.predict(X)
    
    # Handle noise by excluding it
    mask = labels != -1
    if np.sum(mask) == 0:
        # If all points are noise, return a minimal score
        return 0
    y_true_filtered = y[mask]
    y_pred_filtered = labels[mask]
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true_filtered, y_pred_filtered)
    
    # Check if confusion matrix is empty
    if conf_matrix.size == 0:
        return 0
    
    # Use Hungarian algorithm for label alignment
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    
    # Create label mapping
    label_mapping = {pred_label: true_label for pred_label, true_label in zip(col_ind, row_ind)}
    
    # Align predicted labels
    y_pred_aligned = np.array([label_mapping.get(label, -1) for label in labels])
    y_evaluated = y_pred_aligned[mask]
    
    # Compute the desired metric (e.g., F1-Score)
    score = f1_score(y_true_filtered, y_evaluated, average='weighted', zero_division=0)
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
    'eps': [176.3775],          # Example values; adjust as needed
    'min_samples': [4, 5, 6, 7, 8, 9],
    'metric': ['euclidean']  # Add other metrics if desired
}

# Initialize the custom DBSCAN estimator
dbscan_wrapper = DBSCANWrapper()

# 5. Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=dbscan_wrapper,
    param_grid=param_grid,
    scoring=clustering_scorer,  # Pass the custom scoring function directly
    cv=3,  # Number of cross-validation folds; adjust as needed
    verbose=2,
    n_jobs=-1  # Use all available cores
)

# 6. Perform Grid Search to Find the Best Hyperparameters
print("Starting Grid Search for DBSCAN hyperparameters...")
grid_search.fit(X, y_encoded)
print("Grid Search completed.")

# Display the best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# 7. Fit the Best Model with the Best Parameters
best_dbscan = grid_search.best_estimator_
best_dbscan.fit(X)
y_pred = best_dbscan.predict(X)

# 8. Handle Noise and Align Labels as in Original Code
# Handle noise by treating it as a separate cluster or excluding from evaluation
# Here, we'll exclude noise points (-1) from evaluation metrics
mask = y_pred != -1
y_true_filtered = y_encoded[mask]
y_pred_filtered = y_pred[mask]

# Check if there are any clusters detected (excluding noise)
unique_pred_clusters = np.unique(y_pred_filtered)
if len(unique_pred_clusters) == 0:
    print("No clusters found. All points are considered noise.")
else:
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_true_filtered, y_pred_filtered)

    # Use the Hungarian algorithm to find the optimal label mapping
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)

    # Create a mapping from predicted labels to true labels
    label_mapping = {pred_label: true_label for pred_label, true_label in zip(col_ind, row_ind)}

    # Apply the mapping to the predicted labels
    y_pred_aligned = np.array([
        label_mapping[label] if label in label_mapping else -1 for label in y_pred
    ])

    # For evaluation, exclude noise points
    y_pred_evaluated = y_pred_aligned[mask]

    # Evaluation metrics
    accuracy = accuracy_score(y_true_filtered, y_pred_evaluated)
    precision = precision_score(y_true_filtered, y_pred_evaluated, average='weighted', zero_division=0)
    recall = recall_score(y_true_filtered, y_pred_evaluated, average='weighted', zero_division=0)
    f1 = f1_score(y_true_filtered, y_pred_evaluated, average='weighted', zero_division=0)
    ari = adjusted_rand_score(y_true_filtered, y_pred_evaluated)
    nmi = normalized_mutual_info_score(y_true_filtered, y_pred_evaluated)

    # Print evaluation metrics
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-Score: {f1:.2f}')
    print(f'Adjusted Rand Index: {ari:.2f}')
    print(f'Normalized Mutual Information: {nmi:.2f}')
    print(f'Number of clusters found (excluding noise): {len(unique_pred_clusters)}')
    print(f'Number of noise points: {np.sum(y_pred == -1)}')

    # Plot the clustered data
    # Note: This visualization assumes the data has two features for 2D plotting
    if X.shape[1] == 2:
        plt.figure(figsize=(8, 6))
        unique_labels = set(y_pred_aligned)
        colors = plt.cm.get_cmap('viridis', len(unique_labels))
        for label in unique_labels:
            if label == -1:
                # Black color for noise
                color = 'k'
                label_name = 'Noise'
            else:
                color = colors(label)
                label_name = f'Cluster {label}'
            plt.scatter(
                X.iloc[y_pred_aligned == label, 0],
                X.iloc[y_pred_aligned == label, 1],
                s=40,
                color=color,
                label=label_name,
                alpha=0.6,
                edgecolors='w',
                linewidth=0.5
            )
        plt.title('DBSCAN Clustering with Best Parameters')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()
    else:
        print("The data has more than two features; skipping 2D plot.")
