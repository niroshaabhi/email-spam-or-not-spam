import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import streamlit as st


# Load dataset
data = pd.read_csv("email.csv")


# Basic checks
print(data.head())
print(data.shape)

# Remove duplicates
data.drop_duplicates(inplace=True)
print(data.shape)

# Check null values
print(data.isnull().sum())

# Rename labels
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# Split data
X = data['Message']
y = data['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# Test model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "/ 100")

# Prediction function
def predict(message):
    message_tfidf = tfidf.transform([message])
    result = model.predict(message_tfidf)
    return result[0]
st.title("📧 Email Spam Detection")
st.write("Accuracy:", round(accuracy * 100, 2), "/ 100")

user_input = st.text_area("Enter an email message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = predict(user_input)

        if prediction == "Spam":
            st.error("Spam")
        else:
            st.success("Not Spam")


# Test prediction
output = predict("Congratulations, you won a lottery")
print("Prediction:", output)
