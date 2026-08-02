import streamlit as st

st.set_page_config(
    page_title="📊 Customer Churn Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#eef2ff,#dbeafe,#bfdbfe);
}

h1{
text-align:center;
font-size:42px;
color:#1e3a8a;
font-weight:bold;
}

div[data-testid="stMetric"]{
background:white;
padding:18px;
border-radius:15px;
box-shadow:0px 8px 18px rgba(0,0,0,.15);
text-align:center;
}

.stButton>button{
background:#2563eb;
color:white;
border-radius:10px;
height:55px;
font-size:20px;
font-weight:bold;
width:100%;
}

.stButton>button:hover{
background:#1d4ed8;
}

</style>
""",unsafe_allow_html=True)
st.title("📊 Customer Churn Prediction Dashboard")

st.markdown(
"""
<center>

### 🤖 Machine Learning Based Customer Retention System

Predict whether a telecom customer will **Stay** or **Churn**

</center>
""",
unsafe_allow_html=True
)
c1,c2,c3,c4=st.columns(4)

c1.metric("📁 Dataset","7032")
c2.metric("🤖 Model","Random Forest")
c3.metric("🎯 Accuracy","78.5%")
c4.metric("⚡ Features","30")
st.sidebar.image(
"https://img.icons8.com/color/96/artificial-intelligence.png",
width=80
)

st.sidebar.title("Customer Details")

st.sidebar.markdown("---")
if prediction[0]==1:

    st.error("🚨 High Churn Risk")

else:

    st.success("🎉 Customer is likely to Stay")
    a,b=st.columns(2)

a.metric("💚 Stay Probability",
f"{prob[0]*100:.2f}%")

b.metric("❤️ Churn Probability",
f"{prob[1]*100:.2f}%")
fig=px.pie(
names=["Stay","Churn"],
values=[prob[0],prob[1]],
hole=.65,
color=["Stay","Churn"],
color_discrete_map={
"Stay":"#16a34a",
"Churn":"#dc2626"
})

st.plotly_chart(fig,use_container_width=True)
bar=px.bar(

chart,

x="Result",

y="Probability",

text="Probability",

color="Result",

color_discrete_map={
"Stay":"green",
"Churn":"red"
})

bar.update_traces(
texttemplate='%{text:.1f}%',
textposition='outside'
)

st.plotly_chart(bar,use_container_width=True)
st.markdown("---")

st.markdown(
"""
<center>

Made with ❤️ by **Swyam Shukla**

Python | Streamlit | Scikit-Learn | Plotly

</center>
""",
unsafe_allow_html=True
)