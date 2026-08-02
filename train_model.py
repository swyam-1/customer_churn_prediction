import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
# Dataset load
df = pd.read_csv("data/customer_churn.csv")

# print("=" * 50)
# print("First 5 Rows")
# print(df.head())

# print("=" * 50)
# print("Dataset Shape")
# print(df.shape)

# print("=" * 50)
# print("Dataset Information")
# print(df.info())

# print("=" * 50)
# print("Missing Values")
# print(df.isnull().sum())

# print("=" * 50)
# print("Statistical Summary")
# print(df.describe())

# print("=" * 50)
# print("Target Variable")
# print(df["Churn"].value_counts())


# print(df["TotalCharges"].unique())
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# print(df.isnull().sum())
df.dropna(inplace=True)
# print(df.shape)
# print(df.isnull().sum())
# plt.figure(figsize=(6,4))
# sns.countplot(x="Churn", data=df)
# plt.title("Customer Churn Distribution")
# plt.savefig("plot1.png",dpi=300,bbox_inches="tight",transparent=True)
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(6,4))
# sns.countplot(x="gender", hue="Churn", data=df)
# plt.title("Gender vs Churn")
# plt.savefig("plot2.png",dpi=300,bbox_inches="tight",transparent=True)
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(8,5))
# sns.countplot(x="Contract", hue="Churn", data=df)
# plt.title("Contract Type vs Churn")
# plt.xticks(rotation=15)
# plt.savefig("plot3.png",dpi=300,bbox_inches="tight",transparent=True)
# plt.tight_layout()
# plt.show()

# # Tenure Distribution
# plt.figure(figsize=(8,5))
# sns.histplot(df["tenure"], bins=30, kde=True)
# plt.title("Tenure Distribution")
# plt.savefig("plot4.png",dpi=300,bbox_inches="tight",transparent=True)
# plt.tight_layout()
# plt.show()

# # Monthly Charges vs Churn
# plt.figure(figsize=(8,5))
# sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
# plt.title("Monthly Charges vs Churn")
# plt.savefig("plot5.png",dpi=300,bbox_inches="tight",transparent=True)
# plt.tight_layout()
# plt.show()

# # Correlation Heatmap
# plt.figure(figsize=(6,4))
# sns.heatmap(df.select_dtypes(include=["int64","float64"]).corr(),
#             annot=True, cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.savefig("plot6.png",dpi=300,bbox_inches="tight",transparent=True)
# plt.tight_layout()
# plt.show()
# Remove customerID because it is just an identifier
df = df.drop("customerID", axis=1)

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# print("Features Shape:", X.shape)
# print("Target Shape:", y.shape)
# print(X.dtypes)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)

# print(y[:10])
X = pd.get_dummies(X, drop_first=True)

# print(X.head())
# print("New Shape:", X.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# print("Training Data :", X_train.shape)
# print("Testing Data  :", X_test.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

# print("Accuracy :", accuracy)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# print(cm)
from sklearn.metrics import classification_report

# print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

print(classification_report(y_test, rf_pred))
import joblib

# Save model
joblib.dump(model, "models/churn_model.pkl")

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(X.columns.tolist(), "models/model_columns.pkl")

print("✅ Model and preprocessing files saved successfully!")