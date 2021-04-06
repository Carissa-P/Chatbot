from flask import Flask, render_template, request
#from chatterbot import ChatBot
#from chatterbot.trainers import ChatterBotCorpusTrainer
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords  
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 

def cleanseData(text):
  
  #Get stop words
  stop_words = stopwords.words('english')

  #Tokenize text
  allWords = re.split(r'\W+', text.lower())

  #Remove stop words
  cleanedData = [word for word in allWords if not word in stop_words]
  # print(cleanedData)
  return cleanedData
  
df = pd.read_csv('FinalDataset.csv')
df['description'] = df['description'].str.replace(r"[^A-Za-z0-9.'!,? ]+",' ')
df['description'] = df['description'].apply(func = cleanseData)
df['book_name'] = df['book_name'].str.replace(r"[^A-Za-z0-9.'!,? ]+",' ')


def make_prediction(input):
  #Get genre prediction
  input_genre = prediction(input)

  #Turn genre prediction into string
  x = str(input_genre)
  x = re.sub('[^A-Za-z0-9]+', '',x)

  #Filtering data frame to only contain the related genres
  df2 = df.loc[df['genres'] == x]
  df2 = df2[['genres','description', 'book_name','image_url']]
  input = [input.split()]

  new_row = pd.DataFrame({'genres': x, 'description': input}, index=[0])
  df2 = pd.concat([new_row, df2[:]]).reset_index(drop = True)
  return df2

import pickle
def prediction(input):
    # load the vectorizer
    loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))

    # load the model
    loaded_model = pickle.load(open('classification.model', 'rb'))

    # make a prediction
    Prediction = loaded_model.predict(loaded_vectorizer.transform([input]))
    #print(prediction)

    return Prediction

def get_index(array):
  #print(array)
  i = 0
  for x in array:
    y = array[i][1][0]
    if y == 0: 
      return x[1]
    i = i+1   
  #print('array[0][1][0]', array[0][1][0])


def kmeansBoW2(data):
  import random
  df = pd.DataFrame(data, columns =['genres', 'description'])
  df['description'] = df['description'].str.join(' ')
  text = df.description
 
  vectorizer = CountVectorizer()
  # calculate the feature matrix
  feature_matrix = vectorizer.fit_transform(text)
  pca = PCA()
  X = pca.fit_transform(feature_matrix.todense())

  K = KMeans(n_clusters = 3).fit(X)
  labels = K.predict(X)
  labels = labels.tolist()

  # !! Get the indices of the points for each corresponding cluster
  #mydict = {i: np.where(clf.labels_ == i)[0] for i in range(clf.n_clusters)}
  mydict = {i: np.where(K.labels_ == i)[0] for i in range(K.n_clusters)}

  # Transform the dictionary into list
  dictlist = []
  for key, value in mydict.items():
      temp = [key,value]
      dictlist.append(temp)

  my_dict2 = {K.cluster_centers_[i, 0]: np.where(K.labels_ == i)[0] for i in range(K.n_clusters)}
  #print(my_dict2)

  data2 = list(my_dict2.items())
  array_clusters = np.array(data2)
  like_books = get_index(array_clusters)
  random_book = random.choice(like_books)
  #print(random_book)
  #print('Book Reccomendation')
  #print(data.loc[random_book]['book_name'], data.loc[random_book]['image_url'])
  book = data.loc[random_book]['book_name']
  link = data.loc[random_book]['image_url']

  return book, link
def reccomend_book(input):
  df3 = make_prediction('Harry potter')
  df3 = df3.reset_index(drop=True)
  book,link = kmeansBoW2(df3)
  return book

app = Flask(__name__)
#create chatbot
#englishBot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
#trainer = ChatterBotCorpusTrainer(englishBot)

#define app routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
#function for the bot response
def get_bot_response():
  userText = request.args.get('msg')
  book = reccomend_book(userText)
  return(book)
  return str(englishBot.get_response(userText))

if __name__ == '__main__':
  app.run(host="localhost", port=8000, debug=True)
  #app.run(host='0.0.0.0') 

