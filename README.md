❤️ Heart Disease Classification

This project predicts the presence of heart disease using multiple machine learning classification algorithms. The goal is to evaluate and compare several models — including Logistic Regression, KNN, Naive Bayes, and Random Forest — and identify the most effective one based on accuracy, precision, recall, F1-score, and ROC-AUC.

📁 Dataset

Source: heart.csv

Target variable: HeartDisease (0 = No, 1 = Yes)

Total records: 918

Features:Includes clinical attributes such as age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, ECG, max heart rate, exercise-induced angina, oldpeak, and ST slope.

🛠 Tools and Libraries Used

Python (Pandas, NumPy)

scikit-learn (Logistic Regression, KNN, Naive Bayes, Random Forest, GridSearchCV)

StandardScaler

Matplotlib / Seaborn (for plots)

Metrics: Accuracy, Precision, Recall, F1, AUC-ROC

🔧 Preprocessing

Handled missing values and duplicates

LabelEncoded categorical features

Standardized numerical features (for LR, KNN, NB)

Train-Test Split: 70% training, 30% testing

🧠 Models Used

Model                   Accuracy   AUC Score   Precision (1)   Recall (1)    F1-Score (1)    Best Parameters

Logistic Regression     88.0%       0.928      0.92            0.87         0.89             C=1, solver=liblinear

K-Nearest Neighbors     91.3%       0.949      0.93            0.91         0.92             n_neighbors=10, metric=manhattan, weights=distance

Naive Bayes             87.3%       0.933      0.91            0.87         0.89             Default

Random Forest           88.0%       0.947      0.91            0.89         0.90             n_estimators=100, max_depth=None, min_samples_split=5, min_samples_leaf=2

📈 Evaluation Metrics

Each model was evaluated using:

* Confusion Matrix

* Classification Report (Precision, Recall, F1-Score)

* ROC Curve

* AUC (Area Under Curve)
 
* ROC-AUC curve analysis showed KNN had the highest separability, followed closely by Random Forest.

✅ Final Recommendation

Best Performing Model: ✅ KNN (Highest Accuracy & AUC)

Alternative for Scalability: 🔀 Random Forest

Interpretable Baseline: ℹ️ Logistic Regression


Fast Lightweight Model: ⚡ Naive Bayes

🧠 Future Work

Add SHAP/LIME explainability

Include Support Vector Machine and XGBoost

Deploy best model using Streamlit or Flask

Add cross-validation ROC/AUC plots

▶️ How to Run

Clone the repo

Install dependencies

pip install -r requirements.txt

Run the notebook or Python script

jupyter notebook heart_analysis.ipynb


