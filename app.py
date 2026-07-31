import streamlit as st
import pandas as pd
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load('model.pkl')
    except Exception as e:
        return None

model = load_model()

st.title("Loan Approval Prediction App")
st.write("Enter the applicant's details below to predict if their loan will be approved.")

if model is None:
    st.error("Model not found! Please run the training script first to generate 'model.pkl'.")
else:
    # Create input fields for user
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        
    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=150)
        loan_amount_term = st.number_input("Loan Amount Term", min_value=12, value=360)
        credit_history = st.selectbox("Credit History", [1.0, 0.0])
        branch_code = st.text_input("Branch Code (e.g. BR-DEL)", value="BR-DEL")

    if st.button("Predict Loan Status"):
        # Create a dataframe from inputs
        input_data = {
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area],
            'Branch_Code': [branch_code]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Load the original dataset to align the columns (to get the exact dummy variables)
        try:
            train_df = pd.read_csv('cleaned_data/cleaned_loan_dataset_train.csv')
            X_train = train_df.drop(columns=['Loan_Status', 'Loan_ID', 'Remarks'], errors='ignore')
            categorical_cols = X_train.select_dtypes(include=['object']).columns
            
            # Combine input with train data to ensure all dummy columns are created correctly
            combined = pd.concat([X_train, input_df], ignore_index=True)
            combined_encoded = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)
            
            # Extract the encoded input (last row)
            final_input = combined_encoded.tail(1)
            
            # Fill NaNs with median of training data just in case
            final_input = final_input.fillna(X_train.median(numeric_only=True))
            
            # Make prediction
            prediction = model.predict(final_input)
            
            if prediction[0] == 1:
                st.success("🎉 Prediction: Loan Approved (Y)")
            else:
                st.error("❌ Prediction: Loan Rejected (N)")
                
        except Exception as e:
            st.error(f"Error processing input: {e}")
