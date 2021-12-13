from flask import Flask, request, render_template
import pickle as pickle
import numpy as np
import re
from sklearn.neighbors import KNeighborsClassifier

# Задаем имя серверу
app = Flask(__name__)

# Загружаем сохранённый словарь с фильмами
movie2index = pickle.load(open('movies2index.pkl', 'rb'))
# Словарь включающий названия фильмов с удалённой запятой
movie2movie_comma = {re.sub(r',', '', movie): movie for movie in movie2index.keys()}
# Названия всех фильмов
movie_names = list(movie2movie_comma.keys())
# Словарь индекс-фильм
index2movie = {i: movie for movie, i in movie2index.items()}

# Загружаем сохранённый словарь с ссылками на фильмы в Википедии
#movie2wiki = pickle.load(open('movie2wiki.pkl', 'rb'))

# Загружаем эмбеддинги
movies_norm = pickle.load(open('movies_emb50.pkl', 'rb'))
# Размерность эмбеддингов
dim_embedding = 50

# Загружаем рейтинги фильмов
movie2raiting = pickle.load(open('movie2raiting.pkl', 'rb'))


def prepare_classes(positive_movies, negative_movies):
    '''Создаёт два набора данных: любимые и нелюбимые фильмы'''

    selected_movies = positive_movies + negative_movies
    X = np.zeros((len(selected_movies), dim_embedding))
    Y = np.zeros(len(selected_movies))

    for i, movie in enumerate(selected_movies):
        X[i] = movies_norm[movie2index[movie]]
        if movie in positive_movies:
            Y[i] = 1
        else:
            Y[i] = -1

    return X, Y


def recommendation(positive_movies, negative_movies=[]):
    recommended_movies = []
    mean_rating = []

    '''Если только один фильм в одном из классов, либо нет отрицательных фильмов,
    то рекоммендация производится по близости эмбеддингов к эмбеддингам положительного класса.
    Если по крайней мере есть два фильма в каждом из классов, то строится KNN классификатор
    для рекоммендации фильмов.
    Для рекомендуемых фильмов также выводится их средний рейтинг.'''

    # Один пример в положительном классе
    if len(positive_movies) == 1:
        movie = positive_movies[0]
        distances = np.dot(movies_norm, movies_norm[movie2index[movie]])
        similar = np.argsort(distances)[-11:]

        for k in reversed(similar):
            mov = index2movie[k]
            recommended_movies.append(mov)
            mean_rating.append(movie2raiting[mov])
        return recommended_movies[1:], mean_rating

    # Несколько примеров в положительном классе
    if len(positive_movies) >= 2:

        # Один пример в отрицательном классе и несколько в положительном
        if len(negative_movies) < 2:
            X = np.zeros((len(positive_movies), dim_embedding))
            for i, movie in enumerate(positive_movies):
                X[i] = movies_norm[movie2index[movie]]
            # Среднее значение для вектора эбмеддингов
            X = np.mean(X, axis=0)

            distances = np.dot(movies_norm, X)
            similar = np.argsort(distances)[-(10 + len(positive_movies)):]

            for k in reversed(similar):
                mov = index2movie[k]
                if mov not in positive_movies:
                    recommended_movies.append(mov)
                    mean_rating.append(movie2raiting[mov])
            # return recommended_movies[len(positive_movies):]
            return recommended_movies, mean_rating

        # Несколько примеров в каждом из классов
        if len(negative_movies) >= 2:
            X, Y = prepare_classes(positive_movies, negative_movies)

            # Обучаем KNN классификатор
            clf = KNeighborsClassifier(n_neighbors=len(positive_movies) + 1,
                                       weights='distance', metric='minkowski', p=2)
            clf.fit(X, Y)

            # Считаем вероятность отнесения ко второму классу (положительному)
            movie_rating = clf.predict_proba(movies_norm)[0:, 1]
            # Сортируем индексы фильмов по увеличению вероятности отнесения к положительному классу
            sorted_movies = np.argsort(movie_rating)[-(10 + len(positive_movies)):]

            # 10 наиболее рекомендуемых фильмов
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
    # Для вывода результата
    recommend_text = []

    # Считываем текст из форм
    text1 = request.form.get('name')
    text2 = request.form.get('name2')

    if text1 != '':
        # Список положительных и отрицательных фильмов
        pos_movies = text1.split(',')
        neg_movies = text2.split(',')

        # Перейдём к обычным названиям с запятой
        if len(pos_movies) > 1:
            pos_movies = [movie2movie_comma[mov] for mov in pos_movies]
        elif len(pos_movies) == 1:
            pos_movies = [movie2movie_comma[pos_movies[0]]]
        if len(neg_movies) > 1:
            neg_movies = [movie2movie_comma[mov] for mov in neg_movies]

        print('Positive:', pos_movies)
        print('Negative:', neg_movies)

        # Вызываем функцию, делающую рекоммендации
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
    #app.run()
    app.run(host='0.0.0.0')