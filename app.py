
import streamlit as st
import joblib
import re

st.set_page_config(
    page_title="Fake Internship Detection System",
    page_icon="🕵️",
    layout="centered"
)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
st.sidebar.title("📌 Project Information")

st.sidebar.success("🎯 Model Accuracy: 95%")

st.sidebar.write("🤖 Model: Logistic Regression")
st.sidebar.write("📚 Algorithm: TF-IDF + Logistic Regression")
st.sidebar.write("🧑‍💻 Developer: Sapna Rani")
st.sidebar.write("🎓 B.Tech CSE")

st.title("🕵️ Fake Internship Detection System")

st.markdown("""
Welcome to the **Fake Internship Detection System**.

This application uses a **Machine Learning** model to analyze internship and job postings and predict whether they are **Genuine** or **Fraudulent**.

Please enter the job details below and click **Detect Fraud**.
""")

st.divider()

st.subheader("Enter Job Details")
if "title" not in st.session_state:
    st.session_state.title = ""
    st.session_state.company = ""
    st.session_state.description = ""
    st.session_state.requirements = ""
    st.session_state.benefits = ""

if st.button("📄 Load Sample Job"):
    st.session_state.title = "Data Analyst Intern"
    st.session_state.company = "ABC Technologies Pvt Ltd"
    st.session_state.description = "Looking for a Data Analyst Intern with knowledge of Python, SQL and Excel."
    st.session_state.requirements = "Python, SQL, Excel, Communication Skills"
    st.session_state.benefits = "Certificate, Stipend, Flexible Working Hours"

title = st.text_input("💼 Job Title", key="title")

company = st.text_area("🏢 Company Profile", key="company")

description = st.text_area("📄 Job Description", key="description")

requirements = st.text_area("📋 Requirements", key="requirements")

benefits = st.text_area("🎁 Benefits", key="benefits")

if st.button("🔍 Detect Fraud", use_container_width=True):

    if not any([title.strip(), company.strip(), description.strip(), requirements.strip(), benefits.strip()]):
        st.warning("⚠️ Please enter at least one job detail before running the prediction.")
        st.stop()

    full_text = title + " " + company + " " + description + " " + requirements + " " + benefits

    full_text = str(full_text).lower()
    full_text = re.sub(r'[^a-zA-Z ]', '', full_text)

    # Convert text to numbers
    vector_input = vectorizer.transform([full_text])

    # Prediction with loading spinner
    with st.spinner("Analyzing job posting..."):
        progress = st.progress(0)

        for i in range(100):
            progress.progress(i + 1)

        prediction = model.predict(vector_input)
        probability = model.predict_proba(vector_input)

        progress.empty()

    # Show result
    if prediction[0] == 1:
        st.error("⚠️ Fake / Fraud Internship Detected")
        st.warning(
            "This posting contains characteristics commonly found in fraudulent job advertisements. Please verify the company before applying."
        )
    else:
        st.success("✅ Genuine Internship")
        st.info(
            "This internship appears to be genuine based on the Machine Learning model."
        )

    confidence = max(probability[0]) * 100
    st.metric("Prediction Confidence", f"{confidence:.2f}%")

st.divider()
st.caption("🚀 Developed by Sapna Rani | Fake Internship Detection System")