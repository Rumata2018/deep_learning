movie_recommendation_flask is a web application for movie recommendation system based on the links in Wikipedia (file wp_movies_10k.ndjson). 
Recommendations are based on neural network-trained embeddings for movies.

Jupyter notebook Train_model.ipynb contains data processing and embedding training. 
The application itself (flask_server.py) is implemented using Flask and uses trained embeddings (movies_emb50.pkl).

## Usage
- start the server: flask_server.py
- navigate to http://127.0.0.1:5000/movies
