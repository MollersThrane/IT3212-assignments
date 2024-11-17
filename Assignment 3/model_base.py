from abc import ABC, abstractmethod
import pandas as pd

class AbstractModel(ABC):
    @abstractmethod
    def __init__(self, dataset: pd.DataFrame, initial_train_years=43, num_test_years=1):
        pass

    @abstractmethod
    def train_and_test(self):
        pass

    @abstractmethod
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_r2_rmse_mae_mape_per_year(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def plot_results(self):
        pass
