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
from sklearn.preprocessing import LabelEncoder
import seaborn as sns  # For improved plotting aesthetics

# 1. Load and Prepare the Dataset
data = pd.read_csv("./train_pca.csv")

# Separate features (X) and labels (y)
X = data.drop(columns=['type'])
y = data['type']

# Encode string labels into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 2. Initialize and Fit the Agglomerative Clustering Model with specified parameters
agg_clustering = AgglomerativeClustering(
    n_clusters=11,
    linkage='complete',
    metric='cosine'
)
agg_clustering.fit(X)
y_pred = agg_clustering.labels_

# 3. Align Predicted Labels with True Labels Using Hungarian Algorithm
conf_matrix = confusion_matrix(y_encoded, y_pred)
row_ind, col_ind = linear_sum_assignment(-conf_matrix)
label_mapping = {pred_label: true_label for pred_label, true_label in zip(col_ind, row_ind)}
y_pred_aligned = np.array([label_mapping.get(label, -1) for label in y_pred])

# 4. Evaluation Metrics
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

# 5. Create a DataFrame with the results
df_results = X.copy()
df_results['cluster'] = y_pred
df_results['label'] = y

# 6. Calculate the percentage of labeled points in each cluster
cluster_label_counts = pd.crosstab(df_results['cluster'], df_results['label'])
cluster_label_percentages = cluster_label_counts.div(cluster_label_counts.sum(axis=1), axis=0) * 100

print("\nPercentage of each label in each cluster:")
print(cluster_label_percentages)

# 7. Plot the percentages as a stacked bar chart
cluster_label_percentages_filled = cluster_label_percentages.fillna(0)
ax = cluster_label_percentages_filled.plot(
    kind='bar',
    stacked=True,
    figsize=(10, 7),
    colormap='tab20'
)
plt.ylabel('Percentage')
plt.title('Percentage of Each Label in Each Cluster')
plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 8. Visualization
# Plot the clustered data
unique_labels = np.unique(y_pred)
num_unique_labels = len(unique_labels)
colors = plt.cm.get_cmap('viridis', num_unique_labels)

if X.shape[1] >= 3:
    # 3D Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i, label in enumerate(unique_labels):
        indices = y_pred == label
        color = colors(i)
        label_name = f'Cluster {label}'
        ax.scatter(
            X.iloc[indices, 0],
            X.iloc[indices, 1],
            X.iloc[indices, 2],
            s=40,
            color=color,
            label=label_name,
            alpha=0.6,
            edgecolors='w',
            linewidth=0.5
        )
    ax.set_title('Agglomerative Clustering (3D)')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    plt.legend()
    plt.show()
elif X.shape[1] == 2:
    # 2D Plotting
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        indices = y_pred == label
        color = colors(i)
        label_name = f'Cluster {label}'
        plt.scatter(
            X.iloc[indices, 0],
            X.iloc[indices, 1],
            s=40,
            color=color,
            label=label_name,
            alpha=0.6,
            edgecolors='w',
            linewidth=0.5
        )
    plt.title('Agglomerative Clustering (2D)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
else:
    print("The data has less than two features; cannot plot.")
