from sklearn.feature_extraction.text import CountVectorizer
import time

st = time.time()

with open('full_text.txt', 'r', encoding='utf-8') as file:
    text = file.read()

vectorizer = CountVectorizer()

X = vectorizer.fit_transform([text])

feature_names = vectorizer.get_feature_names_out()

bag_of_words = dict(zip(feature_names, X.toarray()[0]))

et = time.time()

print(bag_of_words)

print("sklearn time: ", et - st)