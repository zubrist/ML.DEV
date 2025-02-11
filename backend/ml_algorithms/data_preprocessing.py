import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple

class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.preprocessing_log = []
        self.encoders = {}
        
    def check_missing_values(self) -> Dict:
        """Check for missing values in the dataset."""
        missing_info = {}
        missing_counts = self.df.isnull().sum()
        
        if missing_counts.any():
            for column in missing_counts[missing_counts > 0].index:
                missing_info[column] = {
                    'count': int(missing_counts[column]),
                    'percentage': float(round((missing_counts[column] / len(self.df)) * 100, 2))
                }
        
        return missing_info
    
    def handle_missing_values(self, strategy: Dict[str, str]) -> List[str]:
        """
        Handle missing values according to specified strategy.
        strategy: Dict with column names and strategies ('mean', 'median', 'mode', 'drop')
        """
        logs = []
        for column, method in strategy.items():
            if method == 'drop':
                initial_rows = len(self.df)
                self.df.dropna(subset=[column], inplace=True)
                rows_dropped = initial_rows - len(self.df)
                logs.append(f"Dropped {rows_dropped} rows with missing values in {column}")
            elif method in ['mean', 'median', 'mode']:
                if pd.api.types.is_numeric_dtype(self.df[column]):
                    if method == 'mean':
                        value = self.df[column].mean()
                    elif method == 'median':
                        value = self.df[column].median()
                    else:  # mode
                        value = self.df[column].mode()[0]
                    self.df[column].fillna(value, inplace=True)
                    logs.append(f"Filled missing values in {column} with {method} ({value:.2f})")
                else:
                    mode_value = self.df[column].mode()[0]
                    self.df[column].fillna(mode_value, inplace=True)
                    logs.append(f"Filled missing values in {column} with mode ({mode_value})")
        
        self.preprocessing_log.extend(logs)
        return logs

    def remove_unique_columns(self) -> List[str]:
        """Remove columns with unique values (like IDs)."""
        logs = []
        columns_to_drop = []
        
        for column in self.df.columns:
            if self.df[column].nunique() == len(self.df):
                logs.append(str(f"Removed column '{column}' (unique identifier)"))
                columns_to_drop.append(column)
        
        if columns_to_drop:
            self.df.drop(columns_to_drop, axis=1, inplace=True)
        
        self.preprocessing_log.extend(logs)
        return logs

    def encode_categorical(self) -> List[str]:
        """Encode categorical columns."""
        logs = []
        for column in self.df.select_dtypes(include=['object']).columns:
            encoder = LabelEncoder()
            self.df[column] = encoder.fit_transform(self.df[column])
            self.encoders[column] = encoder
            unique_values = int(len(encoder.classes_))
            logs.append(str(f"Encoded categorical column '{column}' ({unique_values} unique values)"))
        
        self.preprocessing_log.extend(logs)
        return logs

    def get_numeric_columns(self) -> List[str]:
        """Get list of numeric columns."""
        return [str(col) for col in self.df.select_dtypes(include=['int64', 'float64']).columns]

    def get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns."""
        return [str(col) for col in self.df.select_dtypes(include=['object']).columns]

    def get_column_stats(self) -> Dict:
        """Get basic statistics for each column."""
        stats = {}
        for column in self.df.columns:
            stats[column] = {
                'type': str(self.df[column].dtype),
                'unique_values': int(self.df[column].nunique()),
                'missing_values': int(self.df[column].isnull().sum())
            }
            if pd.api.types.is_numeric_dtype(self.df[column]):
                stats[column].update({
                    'mean': float(self.df[column].mean()),
                    'std': float(self.df[column].std()),
                    'min': float(self.df[column].min()),
                    'max': float(self.df[column].max())
                })
        return stats

    def save_processed_data(self, path: str):
        """Save the processed dataset."""
        self.df.to_csv(path, index=False) 