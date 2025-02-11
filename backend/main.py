from fastapi import FastAPI, HTTPException,  APIRouter, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sqlite3
from ml_algorithms.linear_regression import run_linear_regression
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
import base64
import io
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from enum import Enum
from typing import List
import shutil
from ml_algorithms.data_preprocessing import DataPreprocessor
from pathlib import Path


app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# API Router with prefix /api
api_router = APIRouter(prefix="/api")

# Database connection
DATABASE_PATH = "database/mlalgolab.db"


dataset_folder = os.path.join(os.path.dirname(__file__), "datasets")

# Ensure the datasets directory exists
os.makedirs(dataset_folder, exist_ok=True)







# Define supported algorithm types
class AlgorithmType(str, Enum):
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE = "decision_tree"
    SVM = "svm"
    RANDOM_FOREST = "random_forest"
    # Add more algorithms as needed

# Mapping of algorithm types to their dataset suffixes
ALGORITHM_DATASET_MAPPING = {
    AlgorithmType.LINEAR_REGRESSION: "_lin_reg",
    AlgorithmType.LOGISTIC_REGRESSION: "_log_reg",
    AlgorithmType.DECISION_TREE: "_decision_tree",
    AlgorithmType.SVM: "_svm",
    AlgorithmType.RANDOM_FOREST: "_random_forest"
    # Add more mappings as needed
}

@api_router.get("/datasets/{algorithm}")
def get_datasets(algorithm: AlgorithmType) -> dict:
    """
    Get datasets for a specific algorithm type.
    
    Args:
        algorithm: The type of algorithm (e.g., 'linear_regression', 'logistic_regression')
    
    Returns:
        dict: Dictionary containing filtered datasets for the specified algorithm
    """
    if not os.path.exists(dataset_folder):
        raise HTTPException(
            status_code=404, 
            detail="Datasets directory not found"
        )
    
    suffix = ALGORITHM_DATASET_MAPPING.get(algorithm)
    if not suffix:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported algorithm type: {algorithm}"
        )
    
    # Get all datasets and filter for those ending with the appropriate suffix
    all_datasets = os.listdir(dataset_folder)
    filtered_datasets = [
        dataset for dataset in all_datasets 
        if dataset.endswith(f"{suffix}.csv")
    ]
    
    if not filtered_datasets:
        return {
            "datasets": [],
            "message": f"No datasets found for {algorithm.replace('_', ' ').title()}"
        }
    
    return {
        "datasets": filtered_datasets,
        "count": len(filtered_datasets)
    }

# Optional: Add an endpoint to get all supported algorithms
@api_router.get("/algorithms")
def get_supported_algorithms() -> dict:
    """Get list of all supported algorithms"""
    return {
        "algorithms": [algo.value for algo in AlgorithmType],
        "count": len(AlgorithmType)
    }

# Optional: Add an endpoint to get dataset info
@api_router.get("/dataset_info/{dataset_name}")
def get_dataset_info(dataset_name: str) -> dict:
    """Get information about a specific dataset"""
    dataset_path = os.path.join(dataset_folder, dataset_name)
    
    if not os.path.exists(dataset_path):
        raise HTTPException(
            status_code=404,
            detail="Dataset not found"
        )
    
    try:
        df = pd.read_csv(dataset_path)
        return {
            "name": dataset_name,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "size_kb": round(os.path.getsize(dataset_path) / 1024, 2),
            "algorithm_type": next(
                (algo for algo, suffix in ALGORITHM_DATASET_MAPPING.items() 
                 if dataset_name.endswith(f"{suffix}.csv")),
                "unknown"
            )
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading dataset: {str(e)}"
        )






@api_router.get("/dataset_columns/{dataset_name}")
def get_dataset_columns(dataset_name: str):
    # Define a function to get the column names of a dataset
    dataset_path = f"{dataset_folder}/{dataset_name}"
    # Define the path to the dataset
    if not os.path.exists(dataset_path):
        # If the dataset path does not exist, raise an HTTP exception
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = pd.read_csv(dataset_path)
    # Read the dataset
    return {"columns": df.columns.tolist()}





@api_router.get("/dataset_preview/{dataset_name}")
async def get_dataset_preview(dataset_name: str):
    dataset_path = f"{dataset_folder}/{dataset_name}"

    if not os.path.exists(dataset_path):
        return JSONResponse(status_code=404, content={"error": "Dataset not found"})

    try:
        df = pd.read_csv(dataset_path)
        preview = df.head().to_dict(orient="records")
        shape = df.shape  # Tuple: (number_of_rows, number_of_columns)

        return {
            "preview": preview,
            "shape": shape,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})




@app.get("/api/scatter_plot/{dataset}/{feature}/{target}")
async def get_scatter_plot(dataset: str, feature: str, target: str):
    try:
        # Load the dataset
        df = pd.read_csv(f"datasets/{dataset}")
        
        # Create a shorter filename using hash
        filename = f"scatter_{hash(f'{dataset}{feature}{target}')}.png"
        save_path = os.path.join("static", filename)
        
        # Generate scatter plot
        plt.figure(figsize=(10, 6))
        plt.style.use('dark_background')
        plt.scatter(df[feature], df[target], color="#00ddb3", alpha=0.6)
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.title(f"Scatter Plot of {feature} vs {target}")
        plt.grid(True, alpha=0.2)
        
        # Add styling
        plt.gca().spines['bottom'].set_color('white')
        plt.gca().spines['top'].set_color('white')
        plt.gca().spines['right'].set_color('white')
        plt.gca().spines['left'].set_color('white')
        
        # Save with tight layout
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor='#31363f')
        plt.close()

        return {"plot": filename}
    except Exception as e:
        return {"error": str(e)}





# Models


class LinearRegressionRequest(BaseModel):
    dataset: str
    feature: str  # Selected feature column
    target: str   # Selected target column

@app.post("/api/run_linear_regression")
def run_linear_regression_api(request: LinearRegressionRequest):
    dataset_path = f"{dataset_folder}/{request.dataset}"
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    result = run_linear_regression(dataset_path, request.feature, request.target)
    return result


@api_router.post("/upload_dataset/{algorithm}")
async def upload_dataset(algorithm: AlgorithmType, file: UploadFile = File(...)):
    try:
        # Create filename with appropriate suffix
        original_name = file.filename
        name_without_ext = original_name.rsplit('.', 1)[0]
        new_filename = f"{name_without_ext}{ALGORITHM_DATASET_MAPPING[algorithm]}.csv"
        file_path = os.path.join(dataset_folder, new_filename)
        
        # Save the uploaded file first
        with open(file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        
        # Now read the saved file
        df = pd.read_csv(file_path)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(df)
        
        # Check for missing values
        missing_info = preprocessor.check_missing_values()
        
        return {
            "filename": str(new_filename),
            "message": "File uploaded successfully",
            "preprocessing_summary": {
                "missing_values": missing_info,
                "unique_columns_removed": [],  # Empty initially
                "encoded_columns": []  # Empty initially
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Add this class for request validation
class MissingValuesRequest(BaseModel):
    filename: str
    strategies: dict

@api_router.post("/handle_missing_values")
async def handle_missing_values(request: MissingValuesRequest):
    try:
        # Load the dataset
        file_path = os.path.join(dataset_folder, request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dataset file not found")
            
        df = pd.read_csv(file_path)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(df)
        
        # Handle missing values
        logs = preprocessor.handle_missing_values(request.strategies)
        
        # Process the rest of the data
        unique_cols_removed = preprocessor.remove_unique_columns()
        encoded_cols = preprocessor.encode_categorical()
        
        # Create new filename for processed file
        name_without_ext = request.filename.rsplit('.', 1)[0]
        processed_filename = f"{name_without_ext}_processed.csv"
        processed_file_path = os.path.join(dataset_folder, processed_filename)
        
        # Save processed data
        preprocessor.save_processed_data(processed_file_path)
        
        return {
            "filename": processed_filename,
            "message": "Missing values handled successfully",
            "preprocessing_summary": {
                "missing_values_handled": logs,
                "unique_columns_removed": [str(col) for col in unique_cols_removed],
                "encoded_columns": [str(col) for col in encoded_cols],
                "numeric_columns": preprocessor.get_numeric_columns(),
                "categorical_columns": preprocessor.get_categorical_columns()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))







# Create users table if it doesn't exist
def init_db():
    db_path = Path("database/mlalgolab.db")
    db_path.parent.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

# Call this when your application starts
init_db()

# Add this class for the signup request
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

# Add this endpoint to handle signup
@api_router.post("/signup")
async def signup(request: SignupRequest):
    try:
        conn = sqlite3.connect("database/mlalgolab.db")
        cursor = conn.cursor()
        
        # Check if email already exists
        cursor.execute("SELECT email FROM users WHERE email = ?", (request.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Insert new user
        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (request.name, request.email, request.password)
        )
        
        conn.commit()
        return {"message": "Signup successful"}
    
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# Add this endpoint to view users (for development/testing only)
@api_router.get("/users")
async def get_users():
    try:
        conn = sqlite3.connect("database/mlalgolab.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, email FROM users")  # Excluding password for security
        users = cursor.fetchall()
        
        # Convert to list of dictionaries for better readability
        users_list = [
            {
                "id": user[0],
                "name": user[1],
                "email": user[2]
            }
            for user in users
        ]
        
        return {"users": users_list}
    
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# Add this class for signin request validation
class SigninRequest(BaseModel):
    email: str
    password: str

@api_router.post("/signin")
async def signin(request: SigninRequest):
    try:
        conn = sqlite3.connect("database/mlalgolab.db")
        cursor = conn.cursor()
        
        # Check if user exists with matching email and password
        cursor.execute(
            "SELECT id, name FROM users WHERE email = ? AND password = ?", 
            (request.email, request.password)
        )
        user = cursor.fetchone()
        
        if user:
            return {
                "message": "Signin successful",
                "user": {
                    "id": user[0],
                    "name": user[1]
                }
            }
        else:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
            
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# Add these new endpoints to get counts
@api_router.get("/stats")
async def get_stats():
    try:
        # Get users count
        conn = sqlite3.connect("database/mlalgolab.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        users_count = cursor.fetchone()[0]
        conn.close()

        # Get datasets count
        dataset_count = len([f for f in os.listdir(dataset_folder) if f.endswith('.csv')])

        # Calculate total size of datasets in TB
        total_size = 0
        for file in os.listdir(dataset_folder):
            if file.endswith('.csv'):
                file_path = os.path.join(dataset_folder, file)
                total_size += os.path.getsize(file_path)
        
        # Convert bytes to TB (rounded to 2 decimal places)
        total_size_tb = round(total_size / (1024 ** 4), 2)  # 1024^4 for TB

        return {
            "active_users": users_count,
            "dataset_count": dataset_count,
            "total_size_tb": total_size_tb
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Important: Include the API router BEFORE mounting static files
app.include_router(api_router)

# Mount static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "../frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# Root redirect
@app.get("/")
async def root():
    return RedirectResponse(url="/index.html")


# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)        