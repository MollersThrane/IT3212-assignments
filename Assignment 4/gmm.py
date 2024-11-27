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

data = pd.read_csv("./train_pca.csv")

# Separate features (X) and labels (y)
# Replace 'label' with the actual name of your label column
X = data.drop(columns=['type'])
y = data['type']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode string labels into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Initialize and fit the Gaussian Mixture Model
gmm = GaussianMixture(n_components=5, covariance_type='spherical', init_params='random', max_iter=100, tol=0.004, random_state=42)
gmm.fit(X)

# Predict cluster labels
y_pred = gmm.predict(X)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_encoded, y_pred)

# Use the Hungarian algorithm to find the optimal label mapping
row_ind, col_ind = linear_sum_assignment(-conf_matrix)

# Create a mapping from predicted labels to true labels
label_mapping = {old_label: new_label for old_label, new_label in zip(col_ind, row_ind)}

# Apply the mapping to the predicted labels
y_pred_aligned = np.array([label_mapping[label] for label in y_pred])

# Evaluation metrics
accuracy = accuracy_score(y_encoded, y_pred_aligned)
precision = precision_score(y_encoded, y_pred_aligned, average='weighted')
recall = recall_score(y_encoded, y_pred_aligned, average='weighted')
f1 = f1_score(y_encoded, y_pred_aligned, average='weighted')
ari = adjusted_rand_score(y_encoded, y_pred)
nmi = normalized_mutual_info_score(y_encoded, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')
print(f'Adjusted Rand Index: {ari:.2f}')
print(f'Normalized Mutual Information: {nmi:.2f}')

# Plot the clustered data
# Note: This visualization assumes the data has two features for 2D plotting
if X.shape[1] == 2:
    plt.figure(figsize=(8, 6))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred_aligned, s=40, cmap='viridis', zorder=2)
    plt.title('Gaussian Mixture Model Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
else:
    print("The data has more than two features; skipping 2D plot.")