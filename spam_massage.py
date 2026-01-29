from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Setup and Data Preparation
# -----------------------------

# Download stopwords (only first time it will actually download)
nltk.download('stopwords')

# Load dataset (make sure spam.csv is in same folder)
df = pd.read_csv("spam.csv", encoding='latin1')

# Keep only needed columns and rename them
df = df[['v1', 'v2']]
df.columns = ['Category', 'Message']

# Drop missing values and keep only "ham" and "spam"
df = df.dropna(subset=['Message', 'Category'])
df['Message'] = df['Message'].astype(str)
df = df[df['Category'].isin(['ham', 'spam'])]

# Simple text cleaning function
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = ''.join(char for char in text if char not in string.punctuation)  # remove punctuation
    stop = stopwords.words('english')
    text = ' '.join(word for word in text.split() if word not in stop)  # remove stopwords
    return text

# Apply preprocessing
df['Message'] = df['Message'].apply(preprocess_text)

# -----------------------------
# 2. Feature Extraction + Model
# -----------------------------

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Message'])
y = df['Category'].map({'ham': 0, 'spam': 1})

# Train-test split (for better model training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a simple Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 3. Flask App
# -----------------------------

app = Flask(__name__)

@app.route('/')
def home():
    return 'Spam Detection App is running. Go to /form to check a message.'

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint: send JSON {'message': 'your text'} and get spam/ham."""
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'error': 'Message field is required'}), 400

    message = data['message']
    processed = preprocess_text(message)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)[0]
    label = 'Spam' if prediction == 1 else 'Ham'
    return jsonify({'prediction': label})

@app.route('/form', methods=['GET', 'POST'])
def form():
    """Simple HTML form to check one message."""
    result = None
    message = ''
    if request.method == 'POST':
        message = request.form.get('message', '')
        processed = preprocess_text(message)
        vector = vectorizer.transform([processed])
        prediction = model.predict(vector)[0]
        result = 'Spam ❌' if prediction == 1 else 'Not Spam ✅'

    html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Spam Message Detector</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #74ebd5, #ACB6E5);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Segoe UI', sans-serif;
        }
        .container-box {
            background-color: #ffffff;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            max-width: 550px;
            width: 100%;
        }
        .logo {
            width: 70px;
            margin-bottom: 10px;
        }
        .result {
            margin-top: 20px;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div class="container-box text-center">
        <img src="https://cdn-icons-png.flaticon.com/512/561/561127.png" class="logo" alt="Mailbox Icon">
        <h2 class="mb-3">Spam Message Detector</h2>
        <form method="POST">
            <div class="form-floating">
                <textarea name="message" class="form-control" placeholder="Enter a message" id="floatingTextarea" style="height: 120px;" required>{{ message }}</textarea>
                <label for="floatingTextarea">Your Message</label>
            </div>
            <button type="submit" class="btn btn-primary mt-3 w-100 shadow">Check for Spam</button>
        </form>
        {% if result %}
        <div class="result alert alert-{{ 'Spam' in result and 'danger' or 'success' }} mt-4">
            <strong>Prediction:</strong> {{ result }}
        </div>
        {% endif %}
    </div>
</body>
</html>
'''
    return render_template_string(html, result=result, message=message)

if __name__ == '__main__':
    app.run(debug=True)
