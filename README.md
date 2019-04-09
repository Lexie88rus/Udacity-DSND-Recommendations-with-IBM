# Udacity-DSND-Recommendations-with-IBM
Project for building recommendations for IBM Watson platform (part of Udacity DSND)

## About the Project

### Goals of the Project
The goal of the project is to build a recommender system engine to recommend articles to users of IBM Watson platform.
The following data is provided:
* `user-item-interactions.csv` dataset containing log of interactions between users and articles;
* `articles_community.csv` dataset containing articles' full names, descriptions and full bodies.

### Project Results
As a result of the project following classes were made:
* `CollaborativeRecommender` class, which makes recommendations basing on similarity between different users.
Similarity between users is calculated as a dot product (cosine distance) of user-article vectors, which contain 1 if user
interacted with the article and 0 otherwise.
* `ContentBasedRecommender` class, which makes content-based recommendations by processing the contents of the articles.
For each user it recommends articles, which are similar to articles, the user has already interacted with. Similarity between
articles is computed basing on article's keywords (extracted with nltk-rake library) and cosine distanse.

## Repository Contents
The repository has the following structure:
```
- data
|- user-item-interactions.csv  # dataset containing log of interactions between users and articles 
|- articles_community.csv  # dataset containing articles' full names, descriptions and full bodies

- models
|- collaborative_recommender.py # Script contains class CollaborativeRecommender for making user-user collaborative recommendations
|- recommender_helper_functions.py  # Script contains helper functions for both recommenders
|- content_based_recommender.py  # Script contains class ContentBasedRecommender for making content-based recommendations
|- content_based_helpers.py  # Script contains helpers for content-based recommender

- README.md
```

## Code Examples
1. CollaborativeRecommender usage example:
```python
# import recommender
from collaborative_recommender import CollaborativeRecommender 

# instanciate recommender 
rec = CollaborativeRecommender()

# fit recommender to data
rec.fit('../data/user-item-interactions.csv')

# make 10 predictions for user_id = 2
rec.make_recs(2, 10)
```

2. ContentBasedRecommender usage example:
```python
# import recommender
from content_based_recommender import ContentBasedRecommender 

# instanciate recommender 
rec = ContentBasedRecommender()

# fit recommender to data
rec.fit('../data/user-item-interactions.csv', '../data/articles_community.csv')

# make 10 predictions for user_id = 2
rec.make_recs(2, 10)
```
## Demo
![demo](https://github.com/Lexie88rus/Udacity-DSND-Recommendations-with-IBM/blob/master/demo/demo.gif)

## External Libraries
* [NLTK](http://www.nltk.org) library for message text processing,
* [NLTK-Rake](https://github.com/csurfer/rake-nltk) library used for extraction of keywords for building content-based recommendations.
