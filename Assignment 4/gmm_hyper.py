import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
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

# 1. Create a Custom Estimator for GaussianMixture
class GaussianMixtureWrapper(BaseEstimator, ClusterMixin):
    def __init__(self, n_components=1, covariance_type='full', tol=1e-3, max_iter=100, init_params='kmeans', random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.max_iter = max_iter
        self.init_params = init_params
        self.random_state = random_state
    
    def fit(self, X, y=None):
        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            max_iter=self.max_iter,
            init_params=self.init_params,
            random_state=self.random_state
        )
        self.gmm_.fit(X)
        return self
    
    def predict(self, X):
        return self.gmm_.predict(X)

# 2. Define a Custom Scoring Function
def gmm_clustering_scorer(estimator, X, y):
    """
    Custom scoring function for Gaussian Mixture Model clustering that aligns
    cluster labels with true labels using the Hungarian algorithm and computes
    the F1-Score.

    Parameters:
    - estimator: The clustering estimator (GaussianMixtureWrapper).
    - X: Feature data.
    - y: True labels.

    Returns:
    - score: The F1-Score after label alignment.
    """
    # Fit the model
    estimator.fit(X)
    labels = estimator.predict(X)
    
    # Handle cases where the number of predicted clusters differs from the number of true labels
    unique_pred_clusters = np.unique(labels)
    unique_true_labels = np.unique(y)
    n_pred_clusters = len(unique_pred_clusters)
    n_true_labels = len(unique_true_labels)
    
    if n_pred_clusters < n_true_labels:
        # Not enough clusters to match all true labels
        return 0
    elif n_pred_clusters > n_true_labels:
        # More clusters than true labels; possible overfitting
        # You might choose to ignore extra clusters or penalize accordingly
        pass  # For simplicity, proceed without modification
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y, labels)
    
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
    'n_components': [5, 6, 7, 8, 9, 10],  # Adjust based on expected number of clusters
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
    'tol': [0.1, 0.01, 0.001],
    'max_iter': [100, 200],
    'init_params': ['kmeans', 'random']
}

# 5. Initialize the Custom GaussianMixture Estimator
gmm_wrapper = GaussianMixtureWrapper(random_state=42)

# 6. Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=gmm_wrapper,
    param_grid=param_grid,
    scoring=gmm_clustering_scorer,  # Pass the custom scoring function directly
    cv=3,  # Number of cross-validation folds; adjust as needed
    verbose=2,
    n_jobs=-1  # Use all available cores
)

# 7. Perform Grid Search to Find the Best Hyperparameters
print("Starting Grid Search for GaussianMixture hyperparameters...")
grid_search.fit(X, y_encoded)
print("Grid Search completed.")

# Display the best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# 8. Fit the Best Model with the Best Parameters
best_gmm = grid_search.best_estimator_
best_gmm.fit(X)
y_pred = best_gmm.predict(X)

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
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y_pred_aligned,
        s=40,
        cmap='viridis',
        zorder=2,
        edgecolor='k'
    )
    plt.title('Gaussian Mixture Model Clustering with Best Parameters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
else:
    print("The data has more than two features; skipping 2D plot.")
