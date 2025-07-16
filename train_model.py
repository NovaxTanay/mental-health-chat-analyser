import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load dataset (make sure train.txt is in the same folder)
df = pd.read_csv("train.txt", sep=";", header=None, names=["text", "emotion"])

# Show first 5 rows
print(df.head())
# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["emotion"], test_size=0.2, random_state=42)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate the model
accuracy = model.score(X_test_vec, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Save model and vectorizer for use in app.py
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

