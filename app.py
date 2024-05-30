import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests

# df = pickle.load(open('movies.pkl', 'rb'))
df = pd.read_csv('data.csv')
titles = df['title'].values
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()
similarity = cosine_similarity(vectors)
# similarity = pickle.load(open('similarity.pkl', 'rb'))

API_KEY_AUTH = "b8c96e534866701532768a313b978c8b"


def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY_AUTH}')
    data = response.json()
    poster_path = data['poster_path']
    full_path = 'https://image.tmdb.org/t/p/w500/' + poster_path
    return full_path


def recommender(movie):
    movie_index = df[df['title'] == movie].index[0]
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:21]
    movie_recommend = []
    movie_recommend_posters = []
    for i in movies_list:
        movie_id = df.iloc[i[0]]['movie_id']
        movie_recommend.append(df.iloc[i[0]]['title'])
        movie_recommend_posters.append(fetch_poster(movie_id))

    return movie_recommend, movie_recommend_posters


st.set_page_config(layout="wide")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)


st.title('Movie Recommendation System')
selected_movie = st.selectbox('Type a Movie', options=titles)
if st.button('Recommend'):
    recommended_movie_names, recommended_movie_posters = recommender(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])
    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])

    col6, col7, col8, col9, col10 = st.columns(5)
    with col6:
        st.text(recommended_movie_names[5])
        st.image(recommended_movie_posters[5])
    with col7:
        st.text(recommended_movie_names[6])
        st.image(recommended_movie_posters[6])
    with col8:
        st.text(recommended_movie_names[7])
        st.image(recommended_movie_posters[7])
    with col9:
        st.text(recommended_movie_names[8])
        st.image(recommended_movie_posters[8])
    with col10:
        st.text(recommended_movie_names[9])
        st.image(recommended_movie_posters[9])

    col11, col12, col13, col14, col15 = st.columns(5)
    with col11:
        st.text(recommended_movie_names[10])
        st.image(recommended_movie_posters[10])
    with col12:
        st.text(recommended_movie_names[11])
        st.image(recommended_movie_posters[11])
    with col13:
        st.text(recommended_movie_names[12])
        st.image(recommended_movie_posters[12])
    with col14:
        st.text(recommended_movie_names[13])
        st.image(recommended_movie_posters[13])
    with col15:
        st.text(recommended_movie_names[14])
        st.image(recommended_movie_posters[14])

    col16, col17, col18, col19, col20 = st.columns(5)
    with col16:
        st.text(recommended_movie_names[15])
        st.image(recommended_movie_posters[15])
    with col17:
        st.text(recommended_movie_names[16])
        st.image(recommended_movie_posters[16])
    with col18:
        st.text(recommended_movie_names[17])
        st.image(recommended_movie_posters[17])
    with col19:
        st.text(recommended_movie_names[18])
        st.image(recommended_movie_posters[18])
    with col20:
        st.text(recommended_movie_names[19])
        st.image(recommended_movie_posters[19])