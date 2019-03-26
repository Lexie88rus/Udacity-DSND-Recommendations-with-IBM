#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:51:34 2019

@author: aleksandraastakhova

File contains helper functions for content-based recommender class.
"""

import numpy as np
import pandas as pd
from rake_nltk import Rake # import rake to extract keywords from article text
from sklearn.metrics.pairwise import cosine_similarity # import cosine similarity to calculate similarity between articles
from sklearn.feature_extraction.text import CountVectorizer # import count vectorizer for vercorization of keywords from article

def get_keywords(row):
    '''
    Function used to extract keywords from article content
    INPUT:
    row - row from df_new pandas dataframe
    
    OUTPUT:
    keywords - string, containing keywords from 'doc_body' column separated by spaces
    keywords are extracted using nltk_rake library
    '''
    decsription = row['doc_body']
     
    # instantiating Rake, by default it uses english stopwords from NLTK
    # and discards all puntuation characters as well
    r = Rake()

    # extracting the words by passing the text
    r.extract_keywords_from_text(decsription)

    # getting the dictionary whith key words as keys and their scores as values
    key_words_dict_scores = r.get_word_degrees()
    
    # assigning the key words to the new column for the corresponding movie
    keywords = list(key_words_dict_scores.keys())
    keywords = ' '.join(keywords)
    
    return keywords

def prepare_data(df_content):
    '''
    Creates pandas dataframe, which is used for making content-based recommendations
    INPUT:
    df_content - pandas dataframe containing details on articles' content
    
    OUTPUT:
    df_new - pandas dataframe which contains following columns: 'article_id' - id of article from df_content,
    'keywords' - contains strings with keywords extracted from article content separated by spaces
    
    '''    
    # subset df_content to relevant columns for making recommendations
    df_new = df_content[['article_id', 'doc_body']]
    df_new = df_new.dropna(subset=['doc_body'])
    
    # the new column for keywords
    df_new['keywords'] = df_new.apply(lambda row: get_keywords(row),axis=1)

    # dropping the Plot column
    df_new.drop(columns = ['doc_body'], inplace = True)
    
    return df_new

def get_similar_articles(article_id, df_new):
    '''
    Returns list of similar articles using content-based approach  
    INPUT:
    article_id - id of the article to provide similar articles
    df_new - pandas dataframe which contains following columns: 'article_id' - id of article from df_content,
    'keywords' - contains strings with keywords extracted from article content separated by spaces
    
    OUTPUT:
    similar_articles - list of similar articles, the list is empty if article with passed article_id
    is not in df_new dataset
    
    '''    
    try:
    
        # instantiating and generating the count matrix
        count = CountVectorizer()
        count_matrix = count.fit_transform(df_new['keywords'])
    
        # find row number corresponding to article id
        article_idx = df_new[df_new['article_id'] == article_id].index[0]
    
        # find vector from cosine_sim matrix with similarity
        article_similarity = cosine_similarity(count_matrix, count_matrix, dense_output = True)[article_idx, :]
    
        # create dataframe with article ids and corresponding similarity to passed article
        similar_articles = pd.DataFrame(columns = ['article_id', 'similarity'])
        similar_articles.article_id = df_new.article_id.index
        similar_articles.similarity = article_similarity
    
        # sort dataframe by similarity
        similar_articles = similar_articles.sort_values('similarity', ascending = False)
    
        # get ids of similar articles
        similar_articles = similar_articles['article_id'].values.tolist()
    
        # remove article_id from the list
        similar_articles.remove(article_id)
        
    except:
        # if article_id from df is not found in df_new 
        return []
    
    return similar_articles