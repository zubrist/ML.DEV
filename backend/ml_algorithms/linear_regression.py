# backend/ml_algorithms/linear_regression.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from .visualization import plot_linear_regression

def run_linear_regression(dataset_path, feature, target):
    """
    Runs linear regression on the given dataset and returns the results.

    Args:
        dataset_path (str): Path to the dataset file.
        feature (str): Name of the feature column.
        target (str): Name of the target column.

    Returns:
        dict: A dictionary containing various metrics and visualization data.
    """
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Select the feature and target columns
    X = df[[feature]].values
    y = df[target].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,  # 80% training, 20% testing
        random_state=42
    )

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on both training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate various metrics
    metrics = {
        # Basic metrics
        "mse": mean_squared_error(y_test, y_test_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "mae": mean_absolute_error(y_test, y_test_pred),
        "r2_score": r2_score(y_test, y_test_pred),
        
        # Model parameters
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        
        # Dataset info
        "train_size": len(X_train),
        "test_size": len(X_test),
        
        # Additional statistics
        "mean_target": float(np.mean(y)),
        "std_target": float(np.std(y)),
        "feature_range": {
            "min": float(np.min(X)),
            "max": float(np.max(X))
        }
    }

    # Generate regression equation string
    equation = f"y = {metrics['coefficients'][0]:.4f}x + {metrics['intercept']:.4f}"
    metrics["equation"] = equation

    # Calculate prediction intervals (95% confidence)
    residuals = y_test - y_test_pred
    std_residuals = np.std(residuals)
    metrics["prediction_interval"] = float(1.96 * std_residuals)  # 95% confidence interval

    # Generate plot
    plot_filename = plot_linear_regression(
        X_test, 
        y_test, 
        y_test_pred,
        feature_name=feature,
        target_name=target,
        algorithm_name="linear_regression"
    )

    metrics["plot_path"] = plot_filename

    return metrics