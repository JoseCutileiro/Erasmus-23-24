from collections import Counter
import time 

st = time.time()

with open('full_text.txt', 'r', encoding='utf-8') as file:
    text = file.read().lower()

words = text.split()

word_counts = Counter(words)

s = sorted(word_counts)

ret = []
for e in s:
    ret.append([e,word_counts[e]])    

et = time.time()

print("My results: ", et - st)
