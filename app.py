import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
st.set_page_config(
    page_title="📊 Customer Churn Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

/* Background */
.stApp{
background:linear-gradient(135deg,#dbeafe,#eff6ff,#f8fafc);
}

/* Main Title */
h1{
text-align:center;
color:#1e3a8a;
font-size:45px;
font-weight:bold;
}

/* Subtitle */
h3{
color:#2563eb;
}

/* Metric Cards */
div[data-testid="stMetric"]{
background:rgba(255,255,255,0.85);
padding:18px;
border-radius:18px;
box-shadow:0 8px 20px rgba(0,0,0,.15);
transition:0.3s;
}

div[data-testid="stMetric"]:hover{
transform:scale(1.03);
}

/* Buttons */
.stButton>button{
background:#2563eb;
color:white;
font-size:18px;
font-weight:bold;
height:55px;
width:100%;
border-radius:12px;
border:none;
}

.stButton>button:hover{
background:#1d4ed8;
}

/* Dataframe */
[data-testid="stDataFrame"]{
border-radius:15px;
overflow:hidden;
}

</style>
""",unsafe_allow_html=True)
@st.cache_resource
def load_model():
    model = joblib.load("models/churn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    columns = joblib.load("models/model_columns.pkl")
    return model, scaler, columns

model, scaler, model_columns = load_model()
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("""
### 🤖 AI Powered Telecom Customer Retention System
Predict whether a customer is likely to **Stay** or **Churn** using Machine Learning.
""")
c1,c2,c3,c4=st.columns(4)

c1.metric("📁 Dataset","7032")
c2.metric("🤖 Model","Random Forest")
c3.metric("🎯 Accuracy","78.5%")
c4.metric("📋 Columns", "21")

st.divider()
st.sidebar.title("⚙ Customer Information")

st.sidebar.info(
"""
Enter customer details and click **Predict**.

This model predicts customer churn using
Machine Learning.
"""
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Information")

st.sidebar.write("📁 Rows : 7032")
st.sidebar.write("📌 Columns : 21")
st.sidebar.write("🎯 Target : Churn")
st.sidebar.write("🏢 Domain : Telecom")

def get_dataset():
    with open("data/customer_churn.csv","rb") as f:
        return f.read()
    st.sidebar.download_button(
        "⬇️ Download Dataset",
        file,
        file_name="customer_churn.csv",
        mime="text/csv"
    )
st.sidebar.subheader("📋 Dataset Columns")

columns = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]
# Dataset load
@st.cache_data
def load_data():
    return pd.read_csv("data/customer_churn.csv")

df = load_data()
for col in df.columns:
    st.sidebar.write(f"✅ {col}")
    # Dataset Load
st.sidebar.subheader("📋 Dataset Columns")

selected_column = st.sidebar.selectbox(
    "Select a Column",
    df.columns
)

st.sidebar.write(f"**Data Type:** {df[selected_column].dtype}")
st.sidebar.write(f"**Unique Values:** {df[selected_column].nunique()}")
st.sidebar.write("**📄 Values:**")

unique_values = sorted(df[selected_column].dropna().unique())

for value in unique_values:
    st.sidebar.write(f"• {value}")
st.header("Customer Details")
gender = st.selectbox("Gender", ["Male", "Female"])

senior = st.selectbox("Senior Citizen", [0, 1])

partner = st.selectbox("Partner", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.slider("Tenure (Months)", 0, 72, 12)

monthly = st.number_input("Monthly Charges", 18.0, 120.0, 70.0)

total = st.number_input("Total Charges", 0.0, 9000.0, 1000.0)
phone = st.selectbox("Phone Service", ["Yes", "No"])

multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)
input_data = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": device,
    "TechSupport": support,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}

input_df = pd.DataFrame([input_data])

st.subheader("📋 Customer Details")
st.dataframe(
    input_df,
    use_container_width=True
)
# One-Hot Encoding
input_encoded = pd.get_dummies(input_df)

# Missing columns add karo
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Same order as training
input_encoded = input_encoded[model_columns]

# st.subheader("Encoded Data")
# st.write(input_encoded)
if st.button("Predict"):

    # Scale input
    input_scaled = scaler.transform(input_encoded)

    # Prediction
    prediction = model.predict(input_scaled)

    # Probability
    probability = model.predict_proba(input_scaled)
    prob = probability[0]
    if prediction[0] == 1:
       st.error("⚠️ Customer is likely to Churn")
    else:
       st.success("✅ Customer is likely to Stay")
       st.divider()
    c1,c2,c3=st.columns(3) 
    c1.metric("💚 Stay",
    f"{prob[0]*100:.2f}%")

    c2.metric("❤️ Churn",
    f"{prob[1]*100:.2f}%")

    c3.metric("📅 Tenure",
    f"{tenure} Months")

    st.progress(probability[0][1])

    st.write(f"Stay Probability : {probability[0][0]*100:.2f}%")
    st.write(f"Churn Probability : {probability[0][1]*100:.2f}%")
    chart = pd.DataFrame({
    "Result": ["Stay", "Churn"],
    "Probability": [prob[0]*100, prob[1]*100]
    })

    # ===================== Side by Side Charts ======================

    left, right = st.columns(2)

# -------- Gauge Chart --------
    with left:
       fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob[1] * 100,
        title={"text": "🚨 Churn Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red"},
            "steps": [
                {"range": [0, 40], "color": "green"},
                {"range": [40, 70], "color": "orange"},
                {"range": [70, 100], "color": "red"}
            ]
        }
    ))

       st.plotly_chart(fig, use_container_width=True)

# -------- Donut Chart --------
    with right:

        pie = px.pie(
        names=["Stay", "Churn"],
        values=[prob[0], prob[1]],
        hole=0.65,
        color=["Stay", "Churn"],
        color_discrete_map={
            "Stay": "#00C853",
            "Churn": "#D50000"
        },
        title="📊 Prediction Probability"
    )

        pie.update_traces(
        textposition="inside",
        textinfo="percent+label"
        )
        bar = px.bar(
        chart,
        x="Result",
        y="Probability",
        text="Probability",
        color="Result",
        color_discrete_map={
        "Stay": "#00C853",
        "Churn": "#D50000"
        }
        )

        bar.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
        )

        bar.update_layout(
        title="📈 Prediction Probability",
        xaxis_title="Prediction",
        yaxis_title="Probability (%)"
        )

        st.plotly_chart(bar, use_container_width=True)
    st.plotly_chart(pie,use_container_width=True)
if st.button("🔄 Reset"):
    st.rerun()
    st.markdown("---")

st.markdown(
"""
<center>

Made with ❤️ by **Swyam Shukla**

MCA | Data Analyst | Machine Learning

</center>
""",
unsafe_allow_html=True
)