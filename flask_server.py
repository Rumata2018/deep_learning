from flask import Flask, request, render_template
import pickle as pickle
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load the saved dictionary with movies
movie2index = pickle.load(open('movies2index.pkl', 'rb'))
# Names of movies with and without comma
movie2movie_comma = {re.sub(r',', '', movie): movie for movie in movie2index.keys()}
# Names of all movies
movie_names = list(movie2movie_comma.keys())
# Dictionary index-movie
index2movie = {i: movie for movie, i in movie2index.items()}

# Load embeddings for movies
movie_weights = pickle.load(open('movies_emb50.pkl', 'rb'))
# Dimension of embeddings
dim_embedding = 50

# Load rating for movies
movie2raiting = pickle.load(open('movie2raiting.pkl', 'rb'))


def prepare_classes(positive_movies, negative_movies):
    '''Create two datasets: favorite and least favorite movies'''

    selected_movies = positive_movies + negative_movies
    X = np.zeros((len(selected_movies), dim_embedding))
    Y = np.zeros(len(selected_movies))

    for i, movie in enumerate(selected_movies):
        X[i] = movie_weights[movie2index[movie]]
        if movie in positive_movies:
            Y[i] = 1
        else:
            Y[i] = -1

    return X, Y


def recommendation(positive_movies, negative_movies=[]):
    recommended_movies = []
    mean_rating = []

    '''If there is only one movies in one of the classes, or there are no negative movies, 
    then the recommendation is made according to the similarity of the embeddings to the embeddings of the positive class.
    If there are at least two movies in each of the classes, then a KNN classifier is built to recommend movies.
    Rating is also shown for recommended movies.'''

    # Only one movie is in a positive class
    if len(positive_movies) == 1:
        movie = positive_movies[0]
        #distances = np.dot(movies_norm, movies_norm[movie2index[movie]])
        distances = cosine_similarity(movie_weights, [movie_weights[movie2index[movie]]])[:, 0]
        similar = np.argsort(distances)[-11:]

        for k in reversed(similar):
            mov = index2movie[k]
            recommended_movies.append(mov)
            mean_rating.append(movie2raiting[mov])
        return recommended_movies[1:], mean_rating[1:]

    # Several movies are in a positive class
    if len(positive_movies) >= 2:

        # One movie is in a negative class and several movies are in a positive class
        if len(negative_movies) < 2:
            X = np.zeros((len(positive_movies), dim_embedding))
            for i, movie in enumerate(positive_movies):
                X[i] = movie_weights[movie2index[movie]]
            # Average value for the vector of ebmeddings
            X = np.mean(X, axis=0)

            #distances = np.dot(movies_norm, X)
            distances = cosine_similarity(movie_weights, [X])[:, 0]
            similar = np.argsort(distances)[-(10 + len(positive_movies)):]

            for k in reversed(similar):
                mov = index2movie[k]
                if mov not in positive_movies:
                    recommended_movies.append(mov)
                    mean_rating.append(movie2raiting[mov])
            return recommended_movies, mean_rating

        # Several movies are in each of the classes
        if len(negative_movies) >= 2:
            X, Y = prepare_classes(positive_movies, negative_movies)

            # Train KNN classifier
            clf = KNeighborsClassifier(n_neighbors=len(positive_movies) + 1,
                                       weights='distance', metric='minkowski', p=2)
            clf.fit(X, Y)

            # Calculate the probability of attribution to the second class (positive)
            movie_rating = clf.predict_proba(movie_weights)[0:, 1]
            # Sorting movie indices by increasing the probability of being classified as a positive class
            sorted_movies = np.argsort(movie_rating)[-(10 + len(positive_movies)):]

            # Top 10 recommended movies
            for k in reversed(sorted_movies):
                mov = index2movie[k]
                if mov not in positive_movies:
                    recommended_movies.append(mov)
                    mean_rating.append(movie2raiting[mov])
            return recommended_movies, mean_rating

@app.route('/movies')
def my_form2():
    return render_template('movie1.html', values=movie_names)

@app.route('/movies', methods=['POST'])
def my_form_post2():
    # Result output
    recommend_text = []

    # Reading text from forms
    text1 = request.form.get('name')
    text2 = request.form.get('name2')

    if text1 != '':
        # List of positive and negative movies
        pos_movies = text1.split(',')
        neg_movies = text2.split(',')

        # Go to the usual names with comma
        if len(pos_movies) > 1:
            pos_movies = [movie2movie_comma[mov] for mov in pos_movies]
        elif len(pos_movies) == 1:
            pos_movies = [movie2movie_comma[pos_movies[0]]]
        if len(neg_movies) > 1:
            neg_movies = [movie2movie_comma[mov] for mov in neg_movies]

        print('Positive:', pos_movies)
        print('Negative:', neg_movies)

        # Calling the function that makes recommendations
        recommended_movies, rate_movies = recommendation(pos_movies, neg_movies)

        for movie, rate in zip(recommended_movies, rate_movies):
            if rate == None:
                rate = ''
            else:
                rate = ' (' + str(rate) + ')'
            rec = movie + rate
            recommend_text.append(rec)

        print(recommend_text)

    return render_template('movie2.html', text=recommend_text)

@app.route("/")
def hello():
    return "Movie recommendation system"

if __name__ == '__main__':
    #app.debug = True
    app.run(host='0.0.0.0')