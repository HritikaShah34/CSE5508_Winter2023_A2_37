import math
import os
from collections import Counter
import pandas as pd

path = "D://ir//assignment2//CSE508_Winter2023_Dataset"
dir_list = os.listdir(path)

# Create a corpus containing the contents of all 1400 documents
corpus = []
for file in dir_list:
    x = 'D://ir//assignment2//CSE508_Winter2023_Dataset//'+file
    with open(x, 'r') as file:
        corpus.append(file.read())


# Split all the words to create a tokenized corpus
tokenized_corpus = [x.split() for x in corpus]

# Find the set containing all the unique words from tokenized corpus
unique_words = set(word for doc in tokenized_corpus for word in doc)

# Calculate documnet frequency of all the unique terms
doc_freq = {word: sum(1 for doc in tokenized_corpus if word in doc)
            for word in unique_words}

# Calculating idf values of all the unique words
idf = {word: math.log10(
    len(corpus) / (1 + doc_freq[word])) for word in unique_words}


# Creating a list to store the output
output=[]

# ----------------------------------------------------------BINARY------------------------------------------------------#
def binary():
    words=[]
    for x in unique_words:
        tfidf = []
        for y1 in tokenized_corpus:
            if x not in y1:
                # append 0 if word is not present
                tfidf.append(0)
            else:
                # append 1 * idf[word] to get its idf value
                tfidf.append(idf[x])
        words.append(x)
        output.append(tfidf)

    df = pd.DataFrame(output)
    df.index=words
    df.columns = range(1,len(df.columns)+1)
    return df

# ----------------------------------------------------------RAW COUNT------------------------------------------------------#
def rawcount():
    words=[]
    for x in unique_words:
        tfidf = []
        for y1 in tokenized_corpus:
            if x not in y1:
                # append 0 if word is not present
                tfidf.append(0)
            else:
                # append word count in particular document * idf[word]
                tfidf.append(y1.count(x)*idf[x])
        words.append(x)
        output.append(tfidf)

    df = pd.DataFrame(output)
    df.index=words
    df.columns = range(1,len(df.columns)+1)
    return df

# ----------------------------------------------------------TERM FREQUENCY------------------------------------------------------#
def term_freq():
    words=[]
    for x in unique_words:
        tfidf = []
        for y1 in tokenized_corpus:
            l = Counter(y1)
            temp=0
            for j in l:
                temp = temp + l[j]
            if x not in y1:
                # append 0 if word is not present
                tfidf.append(0)
            else:
                # append word count in particular document/total number of terms * idf[word]
                tfidf.append((y1.count(x)/temp)*idf[x])
        words.append(x)
        output.append(tfidf)

    df = pd.DataFrame(output)
    df.index=words
    df.columns = range(1,len(df.columns)+1)
    return df

# ---------------------------------------------------------LOG NORMALIZATION-------------------------------------------------------#
def log_normalize():
    words=[]
    for x in unique_words:
        tfidf = []
        count1 = 0
        for y1 in tokenized_corpus:
            if x not in y1:
                # append 0 if word is not present
                tfidf.append(0)
            else:
                # append (log of word count in particular document + 1) * idf[word]
                tfidf.append((math.log10(y1.count(x)+1))*idf[x])
        words.append(x)
        output.append(tfidf)

    df = pd.DataFrame(output)
    df.index=words
    df.columns = range(1,len(df.columns)+1)
    return df

# ------------------------------------------------------double normalization ----------------------------------------------------#

def double_normalize():
    words=[]
    for x in unique_words:
        tfidf = []
        for y1 in tokenized_corpus:
            l = Counter(y1)
            max=0
            for j in l:
                if(max<l[j]):
                    max=l[j]
            if x not in y1:
                # append 0 if word is not present
                tfidf.append(0)
            else:
                # append 0.5 + (0.5 * word count in particular document/max count among all the terms)* idf[word]
                tfidf.append((0.5 + (0.5 * (y1.count(x)/max)))*idf[x])
        words.append(x)
        output.append(tfidf)

    df = pd.DataFrame(output)
    df.index=words
    df.columns = range(1,len(df.columns)+1)
    return df


print("1 for binary")
print("2 for raw count")
print("3 for term frequancy")
print("4 for log normalization")
print("5 for double normalization")

x=input("Enter number here:")

df1=pd.DataFrame()
if(x=='1'):
    df1=binary()
elif(x=='2'):
    df1=rawcount()
elif(x=='3'):
    df1=term_freq()
elif(x=='4'):
    df1=log_normalize()
elif(x=='5'):
    df1=double_normalize()

print(df1)