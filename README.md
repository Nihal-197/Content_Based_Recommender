# Content_Based_Recommender
Building a content based recommender on the imdb dataset.

The model built uses the info of genres, crew members and few keywords and finally vectorized using the meta soup on CountVectorizer 
cause TF-IDFVectorizer would have assigned low weightage to directors and actors(selecting top 3)
