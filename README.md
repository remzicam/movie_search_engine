# SEARCH ENGINE&amp;MOVIE RECOMMENDER: IMDB TOP 250 MOVIES
Discover the best movies from the IMDB Top 250 list with advanced semantic search engine and movie recommender. Simply enter a keyword, phrase, or even plot. It provides you with a personalized selection of top-rated films!

**App demo:** https://huggingface.co/spaces/remzicam/movie_search_engine
App video: 

## Business Problem
Users face difficulties in finding high-quality films that align with their interests among the vast number of movies available. Additionally, users may not have the time or resources to thoroughly research and compare different films in order to make informed decisions about what to watch. Within this app's advanced semantic search engine and personalized movie recommendations could help users discover the best movies from the IMDB Top 250 list more efficiently and effectively, saving them time and effort in finding films to watch.

## Dataset
imdb top 250 movies

## Tools Used

sentence-transformers: to create embeddings for movie plots and user queries
cosine-similarity: to find similarities between films or user query-films
streamlit: web app and deployment
pickle with xz compression: to store embeddings in very low size


