# ===================================
# 1. Import libraries
# ===================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# ===================================
# 2. Load dataset
# ===================================

data = pd.read_csv("spam.csv", encoding='latin-1')

print(data.columns)

# xử lý dataset Kaggle
if 'v1' in data.columns:
    data = data[['v1','v2']]
    data.columns = ['label','message']

print(data.head())


# ===================================
# 3. Convert label
# ===================================

data['label'] = data['label'].map({'ham':0,'spam':1})


# ===================================
# 4. Train Test Split
# ===================================

X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label'],
    test_size=0.2,
    random_state=42
)


# ===================================
# 5. Text Vectorization
# ===================================

vectorizer = TfidfVectorizer(stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ===================================
# 6. Naive Bayes Model
# ===================================

nb_model = MultinomialNB()

nb_model.fit(X_train_vec, y_train)

nb_pred = nb_model.predict(X_test_vec)

print("Naive Bayes Accuracy:")
print(accuracy_score(y_test, nb_pred))

print(classification_report(y_test, nb_pred))


# ===================================
# 7. Logistic Regression
# ===================================

lr_model = LogisticRegression()

lr_model.fit(X_train_vec, y_train)

lr_pred = lr_model.predict(X_test_vec)

print("Logistic Regression Accuracy:")
print(accuracy_score(y_test, lr_pred))

print(classification_report(y_test, lr_pred))


# ===================================
# 8. Confusion Matrix
# ===================================

cm = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(5,4))

sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Ham','Spam'],
            yticklabels=['Ham','Spam'])

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()


# ===================================
# 9. ROC Curve
# ===================================

prob = lr_model.predict_proba(X_test_vec)[:,1]

fpr, tpr, threshold = roc_curve(y_test, prob)

roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="AUC = %.2f" % roc_auc)

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()


# ===================================
# 10. Test with new message
# ===================================

def predict_message(msg):

    msg_vec = vectorizer.transform([msg])

    pred = lr_model.predict(msg_vec)[0]

    if pred == 1:
        print("Spam")
    else:
        print("Not Spam")


predict_message("Congratulations you won a free ticket")
predict_message("Hey are we meeting today?")