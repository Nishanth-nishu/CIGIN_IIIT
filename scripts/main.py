import pandas as pd
import numpy as np
from train import run_kfold_cv
import warnings
warnings.filterwarnings("ignore")

def load_and_preprocess_data(csv_path):
    """Load and preprocess the dataset"""
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Strip whitespace from column names as requested
    df.columns = df.columns.str.strip()
    
    # Strip whitespace from string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    
    # Remove any rows with missing values
    df = df.dropna()
    
    # Filter out problematic SMILES if any
    df = df[df['SoluteSMILES'].str.len() > 0]
    df = df[df['SolventSMILES'].str.len() > 0]
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Unique solutes: {df['SoluteSMILES'].nunique()}")
    print(f"Unique solvents: {df['SolventSMILES'].nunique()}")
    print(f"Solvation free energy range: {df['delGsolv'].min():.2f} to {df['delGsolv'].max():.2f} kcal/mol")
    
    return df

def main():
    """Main training function following CIGIN paper methodology"""
    print("CIGIN Model Training")
    print("=" * 50)
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("GPU Not Available - Using CPU")
    print("=" * 50)
    
    # Load dataset - replace with your actual CSV path
    csv_path = "https://github.com/adithyamauryakr/CIGIN-DevaLab/raw/master/CIGIN_V2/data/whole_data.csv"
    
    try:
        # Try to load from URL first
        df = pd.read_csv(csv_path)
    except:
        # If URL fails, try local file
        print("Loading from URL failed, trying local file...")
        df = pd.read_csv("whole_data.csv")
    
    # Preprocess data
    data = load_and_preprocess_data(csv_path if 'df' in locals() else "whole_data.csv")
    df = data
    
    # Run k-fold cross validation as described in the paper
    # Paper mentions: "10-fold cross validation scheme was used to assess the model"
    # "We made 5 such 10 cross validation splits and trained our model independently"
    print("\nStarting 10-fold cross validation (5 independent runs)...")
    
    mean_rmse, std_rmse = run_kfold_cv(df, k=10, n_runs=5)
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"CIGIN Model Performance:")
    print(f"RMSE: {mean_rmse:.2f} ± {std_rmse:.2f} kcal/mol")
    print("\nPaper reported RMSE: 0.57 ± 0.10 kcal/mol")
    print("=" * 50)

if __name__ == "__main__":
    main()
