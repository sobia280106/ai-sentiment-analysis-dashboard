import pandas as pd
import re
import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# Load dataset
data = pd.read_csv(
    "socialmedia.csv",
    encoding='latin-1',
    header=None
)

# Dataset column names
data.columns = [
    'sentiment',
    'id',
    'date',
    'query',
    'user',
    'text'
]

# Keep only needed columns
data = data[['sentiment', 'text']]

# Use smaller sample for faster execution
data = data.sample(50000, random_state=42)

# Convert labels
# 0 = Negative
# 4 = Positive

data['sentiment'] = data['sentiment'].replace(4, 1)

# Stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove mentions
    text = re.sub(r'@\w+', '', text)

    # Remove hashtag symbol
    text = re.sub(r'#', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Split into words
    words = text.split()

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Apply cleaning
data['clean_text'] = data['text'].apply(clean_text)

# Features
X = data['clean_text']

# Labels
y = data['sentiment']

# Improved TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)
)

# Convert text to vectors
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Better NLP model
model = LinearSVC()

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# User input loop
while True:

    print("\n----------------------------")

    user_input = input(
        "Enter social media post (or type exit): "
    )

    # Exit option
    if user_input.lower() == "exit":
        print("Program Ended")
        break

    # Clean input
    cleaned_input = clean_text(user_input)

    # Convert to vector
    vector_input = vectorizer.transform(
        [cleaned_input]
    )

    # Predict
    prediction = model.predict(vector_input)

    # Output result
    if prediction[0] == 1:
        print("Predicted Sentiment: POSITIVE")

    else:
        print("Predicted Sentiment: NEGATIVE")