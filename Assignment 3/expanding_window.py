import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def _perform_pca(df, variance_threshold=0.80):
    """
    Perform PCA on the given DataFrame to retain the specified variance.

    Parameters:
    - df (DataFrame): DataFrame containing the dataset.
    - variance_threshold (float): The amount of variance to retain (default is 0.80 for 80%).

    Returns:
    - df_pca (DataFrame): DataFrame containing the principal components.
    - num_components (int): Number of components selected to retain the specified variance.
    - explained_variance (list): List of explained variance ratios for each principal component.
    - pca (PCA): The PCA fitted to the data.
    - scaler (StandardScaler): The scaler fitted to the data.
    """

    # Drop non-numeric columns
    df_numeric = df.select_dtypes(include=[float, int])

    # Handle missing values by filling with the mean of each column
    df_numeric.ffill(inplace=True)

    # Standardize the data
    scaler = StandardScaler().set_output(transform="pandas")
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


class ExpandingWindowByYear:
    def __init__(self, data: pd.DataFrame, initial_train_years: int, test_years: int, result_columns: list[str], variance_to_keep: float = 0.80):
        """
        Initializes the ExpandingWindowByYear with the given data and window sizes.
        """
        if 'Year' not in data.columns:
            raise ValueError("DataFrame must contain a 'Year' column.")
        
        self.original_data = data.copy(deep=True)
        self.initial_train_years = initial_train_years
        self.test_years = test_years
        self.current_train_end_year = self.original_data['Year'].min() + initial_train_years - 1
        self.current_test_start_year = self.current_train_end_year + 1
        self.variance_to_keep = variance_to_keep
        self.result_columns = result_columns

        training_data = self.original_data[self.original_data['Year'] <= self.current_train_end_year].copy(deep=True)
        self.training_results = training_data[self.result_columns].copy(deep=True)
        training_data.drop(columns=self.result_columns, inplace=True)
        self.training_data, self.num_pca_components, _, self.pca, self.scaler = _perform_pca(training_data, self.variance_to_keep)

        test_data = self.original_data[(self.original_data['Year'] >= self.current_test_start_year) & 
                         (self.original_data['Year'] <= self.current_test_start_year + self.test_years - 1)].copy(deep=True)
        self.testing_results = test_data[self.result_columns].copy(deep=True)
        test_data.drop(columns=self.result_columns, inplace=True)
        test_data = test_data.select_dtypes(include=[float, int])
        scaled_test_data = self.scaler.transform(test_data)
        test_pca = self.pca.transform(scaled_test_data)
        self.test_data = pd.DataFrame(test_pca, columns=[f'PC{i+1}' for i in range(self.num_pca_components)])


    def train_window(self):
        """
        Returns the current training input window as a DataFrame, and the corresponding output values.
        """
        return self.training_data, self.training_results

    def test_window(self):
        """
        Returns the current testing input window as a DataFrame, and the corresponding output values.
        """
        return self.test_data, self.testing_results

    def extend_train_window(self):
        """
        Extends the training window by appending the current test window. Will raise an error if it cannot move further!
        """
        self.current_train_end_year += self.test_years
        self.current_test_start_year += self.test_years
        
        training_data = self.original_data[self.original_data['Year'] <= self.current_train_end_year].copy(deep=True)
        self.training_results = training_data[self.result_columns].copy(deep=True)
        training_data.drop(columns=self.result_columns, inplace=True)
        self.training_data, self.num_pca_components, _, self.pca, self.scaler = _perform_pca(training_data, self.variance_to_keep)

        test_data = self.original_data[(self.original_data['Year'] >= self.current_test_start_year) & 
                         (self.original_data['Year'] <= self.current_test_start_year + self.test_years - 1)].copy(deep=True)
        self.testing_results = test_data[self.result_columns].copy(deep=True)
        test_data.drop(columns=self.result_columns, inplace=True)
        test_data = test_data.select_dtypes(include=[float, int])
        
        if test_data.empty:
            raise ValueError("Test data is empty. Check the filtering conditions for test_data.")
        
        scaled_test_data = self.scaler.transform(test_data)
        test_pca = self.pca.transform(scaled_test_data)
        self.test_data = pd.DataFrame(test_pca, columns=[f'PC{i+1}' for i in range(self.num_pca_components)])
        
        if self.current_test_start_year > self.original_data['Year'].max():
            raise IndexError("Testing window exceeds available data years.")


# Example usage:
if __name__ == "__main__":
    data_df = pd.read_csv("./preprocessed_stock_data.csv")
    # Assuming you have loaded your data into a DataFrame `data_df`
    # pca_result, components, variance_ratios, _, _ = _perform_pca(data_df)

    # print(f'Number of components to retain 80% variance: {components}')
    # print('Explained variance ratios for each component:')
    # for i, variance in enumerate(variance_ratios, start=1):
    #     print(f'Principal Component {i}: {variance:.2%} of variance')

    # # Save the PCA result to a new CSV file
    # pca_result.to_csv('pca_result.csv', index=False)
    
    test = ExpandingWindowByYear(data_df, 1, 1, ["Close"])

    # print(test.train_window())
    # print(test.test_window())
    test.extend_train_window()
    # print(test.train_window())
    # print(test.test_window())
    test.extend_train_window()
    test.extend_train_window()
    test.extend_train_window()
