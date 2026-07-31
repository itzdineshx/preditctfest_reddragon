import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

def train_accurate_model(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop(columns=['Loan_Status', 'Loan_ID', 'Remarks'], errors='ignore')
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
    
    # Train an Ensemble model (Voting Classifier)
    print("Training Ensemble Model (Random Forest + XGBoost + Logistic Regression)...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='logloss')
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    
    model = VotingClassifier(estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('lr', lr_model)
    ], voting='soft')
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_y.classes_))
    
    # Save the model
    print("\nSaving models...")
    joblib.dump(model, 'model.pkl')
    # Save a copy as .h5 (note: sklearn models are usually saved as .pkl, saving as .h5 using joblib for compatibility)
    joblib.dump(model, 'model.h5')
    print("Models saved as 'model.pkl' and 'model.h5'.")
    
    return model

if __name__ == "__main__":
    train_accurate_model("cleaned_data/cleaned_loan_dataset_train.csv")
