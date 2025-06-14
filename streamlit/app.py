import streamlit as st
import joblib
from shared.preprocessing import clean_text

model = joblib.load('logitsic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')



st.title("Fake News Detection")


user_input = st.text_area("Enter news text here:")

if st.button("Predict"):
    if user_input:
        cleaned_input = clean_text(user_input)  # Apply the same cleaning
        vect_input = vectorizer.transform([cleaned_input])
        
        prediction = model.predict(vect_input)[0]
        probs = model.predict_proba(vect_input)[0]

        st.write(f"Fake probability: {probs[1]:.2f}, Real probability: {probs[0]:.2f}")

        if prediction == 0:
            st.error("Fake News ⚠️")
        else:
            st.success("Real News ✅")
