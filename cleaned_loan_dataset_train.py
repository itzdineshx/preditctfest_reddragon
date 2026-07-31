import pandas as pd
import numpy as np

def clean_data(input_file, output_file):
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # 1. Standardize strings: strip leading/trailing whitespaces and convert to Title Case for categorical columns
    categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Strip whitespace
            df[col] = df[col].astype(str).str.strip()
            
            # Convert NaN represented as strings back to actual NaN
            df[col] = df[col].replace(['nan', 'NaN', 'None', ''], np.nan)
            
            if col in categorical_columns:
                df[col] = df[col].str.title()
                
    # Specific case for Loan_Status as it should probably be 'Y' or 'N' instead of 'Y'/'N'
    if 'Loan_Status' in df.columns:
        df['Loan_Status'] = df['Loan_Status'].str.upper()

    # 2. Handle missing values
    # Categorical: fill with mode
    for col in categorical_columns:
        if col in df.columns:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            
    # Numerical: fill with median
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        
    # Also handle 'Dependents' as mode, since it has '3+'
    if 'Dependents' in df.columns:
        mode_val = df['Dependents'].mode()[0]
        df['Dependents'] = df['Dependents'].fillna(mode_val)

    # 3. Remove Duplicates
    initial_shape = df.shape
    df = df.drop_duplicates()
    final_shape = df.shape
    if initial_shape[0] != final_shape[0]:
        print(f"Removed {initial_shape[0] - final_shape[0]} duplicate rows.")

    # Verification prints
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    print("\nData info after cleaning:")
    print(df.info())

    print(f"\nSaving cleaned data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    input_filepath = "train.csv"
    output_filepath = "train_cleaned.csv"
    clean_data(input_filepath, output_filepath)
