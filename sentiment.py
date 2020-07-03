import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
import numpy as np

df = pd.read_csv('amazon.txt', sep='\t', names=['txt','liked'])

# Word tokenizer  and removal of stop words
stopset=set(stopwords.words('english')) 

# transformation from upper case to lower case 
vectorizer = TfidfVectorizer(use_idf=True,lowercase=True,strip_accents='ascii',stop_words=stopset)

#  Detetminig independent variable 
y=df.liked
# Determining the dependent variable 
x=vectorizer.fit_transform(df.txt)

# Training and testing the data with some random set of data 
''' Random state is a seed used by the random number generator 
	If random state instance , random state is the random number generator;
	If 'none' the random number generator is the random state instance used by np.random. '''
# to maximize the accuracy we have taken random number generator as 42
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)  
# Applying the Naive Bayes Multinomial Algorithm
clf=naive_bayes.MultinomialNB()
#Training the data by applying Naive Bayes Algorithm
clf.fit(x_train,y_train)

#Review = input()
#reviews_arr=np.array([Review])
#reviews_vector=vectorizer.transform(reviews_arr) 
#Ans = clf.predict(reviews_vector)

#if Ans==1:
#   print("Positive Review!")
#if Ans==0:
#    print("Negative Review!")

