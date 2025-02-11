import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt
import os
import uuid
from datetime import datetime


STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

def plot_linear_regression(X, y, y_pred, feature_name, target_name, algorithm_name="linear_regression", save_dir=STATIC_DIR):
    """
    Plots the actual vs predicted values for linear regression and saves the plot as an image with a unique filename.

    Args:
        X (array-like): Feature values.
        y (array-like): Actual target values.
        y_pred (array-like): Predicted target values.
        feature_name (str): Name of the feature column for x-axis label.
        target_name (str): Name of the target column for y-axis label.
        algorithm_name (str): Name of the algorithm (e.g., "linear_regression").
        save_dir (str): Directory to save the plot image.

    Returns:
        str: The path to the saved plot image.
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Generate a unique filename using a timestamp and UUID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]  # Short unique identifier
    filename = f"{algorithm_name}_{timestamp}_{unique_id}.png"
    save_path = os.path.join(save_dir, filename)

    # Set the style for better visibility on dark backgrounds
    plt.style.use('dark_background')
    
    # Generate and save the plot
    plt.figure(figsize=(20,10))
    plt.scatter(X, y, color="#00ddb3", alpha=0.6, label="Actual Data")
    plt.plot(X, y_pred, color="#ff6b6b", linewidth=2, label="Regression Line")
    
    # Set dynamic labels
    plt.xlabel(feature_name)
    plt.ylabel(target_name)
    plt.title(f"Linear Regression: {feature_name} vs {target_name}")
    
    # Customize the plot
    plt.legend(framealpha=0.8)
    plt.grid(True, alpha=0.2)
    
    # Add a light border around the plot
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['top'].set_color('white')
    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    
    # Save the plot with tight layout
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor='#31363f')
    plt.close()

    return filename.replace("\\", "/")  # Ensure correct path formatting
    #return filename
    #return save_path.replace("\\", "/")