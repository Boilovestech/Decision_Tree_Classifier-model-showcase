import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split

# Custom CSS for subtle gradients and flat UI
st.markdown("""
<style>
@keyframes subtle-gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.stApp {
    background: linear-gradient(-45deg, #060606FF, #161925FF, #283162FF, #1E5289FF);
    background-size: 400% 400%;
    animation: subtle-gradient 15s ease infinite;
}
.stButton>button {
    color: #333333;
    background-color: #ffffff;
    border: none;
    border-radius: 4px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}
.stSubheader {
    color: black;
}
</style>
""", unsafe_allow_html=True)

st.title("Salary Prediction App")

@st.cache_data
def load_data():
    data = pd.read_csv("salaries.csv")
    return data

data = load_data()

st.subheader("Data Overview")
st.write(data.head())

st.subheader("Data Preprocessing")
X = data.drop('salary_more_then_100k', axis='columns')
y = data["salary_more_then_100k"]

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

X["company_n"] = le_company.fit_transform(X["company"])
X["job_n"] = le_job.fit_transform(X["job"])
X["degree_n"] = le_degree.fit_transform(X["degree"])

X_e = X.drop(['company', 'job', 'degree'], axis='columns')

st.write("Encoded features:")
st.write(X_e.head())

st.subheader("Data Splitting")
X_train, X_test, y_train, y_test = train_test_split(X_e, y, test_size=0.2, random_state=42)
st.markdown("""
<style>
.metric-card {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
}
.metric-value {
    font-size: 24px;
    font-weight: bold;
    color: #333333;
}
.metric-label {
    font-size: 16px;
    color: #666666;
}
.accuracy-card {
    background-color: #f0f8ff;
    border-left: 5px solid #007bff;
}
.accuracy-value {
    font-size: 36px;
    font-weight: bold;
    color: #007bff;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{X_train.shape[0]} x {X_train.shape[1]}</div>
        <div class="metric-label">Training set shape</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{X_test.shape[0]} x {X_test.shape[1]}</div>
        <div class="metric-label">Testing set shape</div>
    </div>
    """, unsafe_allow_html=True)

model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

st.header("Model Evaluation")
score = model.score(X_test, y_test)
st.markdown(f"""
<div class="metric-card accuracy-card">
    <div class="metric-label">Model Accuracy</div>
    <div class="accuracy-value">{score:.2f}</div>
</div>
""", unsafe_allow_html=True)

st.subheader("Salary Prediction")
st.sidebar.header("Input Features")

company = st.sidebar.selectbox("Company", data['company'].unique())
job = st.sidebar.selectbox("Job", data['job'].unique())
degree = st.sidebar.selectbox("Degree", data['degree'].unique())

if st.button("Predict Salary"):
    company_n = le_company.transform([company])[0]
    job_n = le_job.transform([job])[0]
    degree_n = le_degree.transform([degree])[0]
    
    prediction = model.predict([[company_n, job_n, degree_n]])
    
    result = "More than 100k" if prediction[0] == 1 else "Less than 100k"
    
    st.success(f"The predicted salary is: {result}")

st.sidebar.markdown("---")
st.sidebar.write("Created with ❤️ by [BoiCodes](https://github.com/Boilovestech)")
