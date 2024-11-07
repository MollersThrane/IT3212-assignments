import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def perform_pca(csv_file_path, variance_threshold=0.80):
    """
    Perform PCA on the dataset from the given CSV file to retain the specified variance.

    Parameters:
    - csv_file_path (str): Path to the CSV file containing the dataset.
    - variance_threshold (float): The amount of variance to retain (default is 0.80 for 80%).

    Returns:
    - df_pca (DataFrame): DataFrame containing the principal components.
    - num_components (int): Number of components selected to retain the specified variance.
    - explained_variance (list): List of explained variance ratios for each principal component.
    - pca (PCA): The PCA fitted to the data
    - scaler (StandardScaler): The scaler scaled to the data
    """

    # Load the dataset
    df = pd.read_csv(csv_file_path)

    # Drop non-numeric columns
    df_numeric = df.select_dtypes(include=[float, int])

    # Handle missing values by filling with the mean of each column
    df_numeric.fillna(df_numeric.mean(), inplace=True)

    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)

    # Fit PCA to determine the number of components to retain the desired variance
    pca = PCA()
    pca.fit(df_scaled)
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    num_components = (cumulative_variance >= variance_threshold).argmax() + 1

    # Apply PCA with the determined number of components
    pca = PCA(n_components=num_components)
    df_pca = pca.fit_transform(df_scaled)

    # Create a DataFrame with the principal components
    df_pca = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(num_components)])

    # Explained variance for each principal component
    explained_variance = pca.explained_variance_ratio_

    return df_pca, num_components, explained_variance, pca, scaler

# Example usage:
if __name__ == "__main__":
    csv_path = './preprocessed_stock_data.csv'  # Replace with your CSV file path
    pca_result, components, variance_ratios, _, _ = perform_pca(csv_path)

    print(f'Number of components to retain 80% variance: {components}')
    print('Explained variance ratios for each component:')
    for i, variance in enumerate(variance_ratios, start=1):
        print(f'Principal Component {i}: {variance:.2%} of variance')

    # Save the PCA result to a new CSV file
    pca_result.to_csv('pca_result.csv', index=False)
