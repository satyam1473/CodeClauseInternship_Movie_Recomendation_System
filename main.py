import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


movies = pd.read_csv('C:/Users/SATYAM KUMAR/Desktop/folder/data/tmdb_5000_movies.csv')
credits = pd.read_csv('C:/Users/SATYAM KUMAR/Desktop/folder/data/tmdb_5000_credits.csv') 


movies.merge(credits, on='title').shape

movies = movies.merge(credits, on='title')

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

movies.dropna(inplace=True)

movies.isna().sum()
movies.duplicated().sum()


import ast
def convert(object):
    l = []
    for i in ast.literal_eval(object):
        l.append(i['name'])
    return l


movies['genres'] = movies['genres'].apply(convert)

movies['keywords'] = movies['keywords'].apply(convert)


def convert_cast(object):
    l = []
    for i in ast.literal_eval(object)[:3]:
        l.append(i['name'])
    return l

movies['cast'] = movies['cast'].apply(convert_cast)


import ast
def fetch_director(object):
    L = []
    for i in ast.literal_eval(object):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

df = movies[['movie_id', 'title', 'tags']]

df['tags'] = df['tags'].apply(lambda x:" ".join(x))

df['tags'] = df['tags'].apply(lambda x:x.lower())


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


df['tags'] = df['tags'].apply(stem)

import sklearn
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')
cv.fit_transform(df['tags']).toarray().shape
vectors = cv.fit_transform(df['tags']).toarray()


from sklearn.metrics.pairwise import cosine_similarity


similarity = cosine_similarity(vectors)
similarity[1]

sorted(similarity[1], reverse=True)[:10]
sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x:x[1])[1:6]
df[df['title'] == "Spectre"].index[0]
df.iloc[1216]['title']

def recommender(movie):
    movie_index = df[df['title'] == movie].index[0]
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x:x[1])[1:6]
    movie_recommend = []
    movie_ids = []
    for i in movies_list:
        movie_id = df.iloc[i[0]]['movie_id']
        movie_ids.append(movie_id)
        # print(df.iloc[movie_id]['title'])
        movie_recommend.append(df.iloc[i[0]]['title'])
    return movie_recommend, movie_ids


name, id = recommender('Aliens')

name


list(df['title'])[:10]

import pickle
pickle.dump(df, open('movies.pkl', 'wb'))

new_df = pickle.load(open('movies.pkl', 'rb'))
new_df.head()

new_df['title'].values

new_df.to_csv('data.csv')

pickle.dump(similarity, open('similarity.pkl', 'wb'))

import requests
def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=b8c96e534866701532768a313b978c8b')
    data = response.json()
    # print(data)
    poster_path = data['poster_path']
    full_path = 'https://image.tmdb.org/t/p/w500/' + poster_path
    return full_path

fetch_poster(49529)

def recommender(movie):
    movie_index = df[df['title'] == movie].index[0]
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x:x[1])[1:6]
    movie_recommend = []
    movie_recommend_posters = []
    for i in movies_list:
        movie_id = df.iloc[i[0]]['movie_id']
        # print(df.iloc[idx]['title'])
        movie_recommend.append(df.iloc[i[0]]['title'])
        movie_recommend_posters.append(fetch_poster(movie_id))

    return movie_recommend, movie_recommend_posters

name, poster = recommender('Avatar')

name