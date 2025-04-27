# Bank Customer Churn Prediction Project

## Project Overview
This project aims to predict customer churn for a bank using machine learning techniques. Churn prediction helps banks identify customers who are likely to leave, enabling proactive retention strategies.

The workflow includes:
- Downloading and loading the dataset
- Data preprocessing and feature engineering
- Handling class imbalance with SMOTE
- Training a Random Forest Classifier
- Evaluating model performance
- Saving the trained model for future use

## Dataset Used
- **Source:** [Kaggle - Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)
- **File:** `Churn_Modelling.csv`
- **Features:**
  - CreditScore
  - Age
  - Tenure
  - Balance
  - NumOfProducts
  - HasCrCard
  - IsActiveMember
  - EstimatedSalary
  - Gender (encoded)
  - Exited (target: 1 = churned, 0 = not churned)
- **Dropped Columns:** RowNumber, CustomerId, Surname, Geography (to avoid data leakage and focus on relevant features)

## Features and Drawbacks of the Dataset
### Features
- Contains demographic, financial, and account activity data for 10,000 customers.
- Target variable (`Exited`) clearly indicates churn status.

### Drawbacks
- **Class Imbalance:** The dataset has more non-churned than churned customers, which can bias model training.
- **Categorical Variables:** Some features are categorical and require encoding.
- **Potential Data Leakage:** Columns like CustomerId, Surname, and Geography do not contribute to prediction and may leak information if not dropped.

### How Drawbacks Are Addressed
- **Class Imbalance:** Addressed using SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes in both training and test sets.
- **Categorical Variables:** One-hot encoding is applied to categorical features (e.g., Gender).
- **Data Leakage:** Irrelevant columns are dropped during preprocessing.

## How to Run This Project
1. **Clone the repository and navigate to the project directory.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up Kaggle API credentials** to enable dataset download (see [Kaggle API docs](https://www.kaggle.com/docs/api)).
4. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook churn_predd_model_training.ipynb
   ```
   Follow the notebook cells to download the dataset, preprocess data, train the model, and evaluate results.
5. **The trained model will be saved as** `model/Bank_Churn_pred_model.joblib`.
6. **Run the Streamlit app for interactive predictions:**
   ```bash
   streamlit run app.py
   ```
   This will launch a web interface where you can enter customer details and predict the likelihood of churn using the trained model.

## Streamlit App - Bank Customer Churn Predictor
A new Streamlit app (`app.py`) has been added for interactive customer churn prediction. The app allows users to input customer details and instantly see the prediction (whether the customer is likely to stay or leave) using the trained model. The app provides a user-friendly interface and visual feedback for predictions.

### Features
- Input fields for all model features (credit score, age, tenure, balance, etc.)
- Real-time prediction with visual feedback (success, error, balloons)
- Uses the trained model saved at `model/Bank_Churn_pred_model.joblib`

## Try the App Online
The Streamlit app is deployed and available online. You can access and try out the app [**here**](https://bank-churnpredictor.streamlit.app)

## License
See [LICENSE](LICENSE) for details.
