import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def text_transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]

    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS spam classifier")
input_sms = st.text_area("Enter the Massage")

if st.button("predict"):
    transfored_text = text_transform(input_sms)

    vector_input = tfidf.transform([transfored_text])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")
