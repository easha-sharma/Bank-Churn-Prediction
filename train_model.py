import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# 1. Load dataset
df = pd.read_csv("Bank_Churn.csv")
data = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore").dropna()

# 2. Features and target
X = data.drop(columns=["Exited"])
y = data["Exited"]

# 3. Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Preprocessing
categorical_cols = X_train.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("encoder", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder="passthrough")

# 5. Pipeline with SMOTE + GradientBoosting
model = Pipeline([
    ("preprocessing", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", GradientBoostingClassifier(random_state=42))
])

print("Training model...")
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

# 7. Save model
joblib.dump(model, "model.pkl")
print("Saved: model.pkl successfully!")

