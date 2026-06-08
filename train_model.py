import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math

nltk.download('punkt')
nltk.download('stopwords')

data = {
    'text': [
        "Win a free iphone now! Click here to claim your cash prize.",
        "Hey, are we still meeting for lunch today at 1 PM?",
        "URGENT: Your account has been suspended. Verify your details.",
        "Can you send me the lecture notes for yesterday's class?",
        "Free entry in 2 a weekly comp to win FA Cup final tickets 21st May.",
        "Dear student, your assignment submission has been received successfully."
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}
df = pd.DataFrame(data)

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    cleaned_words = [word for word in words if word.isalnum() and word not in stop_words]
    return cleaned_words

df['cleaned_text'] = df['text'].apply(preprocess_text)

spam_words = []
ham_words = []

for idx, row in df.iterrows():
    if row['label'] == 'spam':
        spam_words.extend(row['cleaned_text'])
    else:
        ham_words.extend(row['cleaned_text'])

total_spam_words = len(spam_words)
total_ham_words = len(ham_words)

spam_freq = nltk.FreqDist(spam_words)
ham_freq = nltk.FreqDist(ham_words)

vocab = set(spam_words + ham_words)
vocab_size = len(vocab)

def predict_spam_or_ham(text):
    words = preprocess_text(text)
    
    p_spam = math.log(0.5)
    p_ham = math.log(0.5)
    
    for word in words:
        spam_count = spam_freq[word] + 1
        ham_count = ham_freq[word] + 1
        
        p_spam += math.log(spam_count / (total_spam_words + vocab_size))
        p_ham += math.log(ham_count / (total_ham_words + vocab_size))
        
    if p_spam > p_ham:
        return "SPAM"
    else:
        return "HAM"

print("=== Model Training ===")
print("Naively trained custom text classification model successfully!\n")

print("=== Real-time Testing ===")
test_emails = [
    "Congratulations! You won a free lottery prize of $1000.",
    "Hello bro, call me when you are free."
]

for email in test_emails:
    prediction = predict_spam_or_ham(email)
    print(f"Email: '{email}' -> Prediction: {prediction}")