import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def train_accurate_model(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop(columns=['Loan_Status', 'Loan_ID'], errors='ignore')
    y = df['Loan_Status']
    
    # Encode target variable
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    
    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Fill any remaining missing values with 0 or median (if any)
    X = X.fillna(X.median())
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Gradient Boosting model (for better accuracy)
    print("Training Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_y.classes_))
    
    return model

if __name__ == "__main__":
    train_accurate_model("cleaned_data/cleaned_loan_dataset_train.csv")
