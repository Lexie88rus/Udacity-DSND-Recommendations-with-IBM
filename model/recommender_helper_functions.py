#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:45:27 2019

@author: aleksandraastakhova

File contains helper functions for collaborative recommender class.
"""

import numpy as np
import pandas as pd

def email_mapper(df):
    '''
    Function to to map the user email to a user_id.
    
    INPUT:
        df - pandas dataframe with 'email' column with anonymized emails
        to be mapped to user ids
        
    OUTPUT:
        email_encoded - list of user ids 
    '''
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

def get_top_articles(n, df):
    '''
    Function returns names of most popular articles (articles which have
    the largest number of interactions with users)
    
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df which contains user interactions with articles
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    # Count number of iteractions for each article_id
    article_user = df.groupby('article_id')['user_id'].count()
    
    # Sort articles by number of iteractions
    article_user = article_user.sort_values(ascending = False)
    
    # Get top-n article ids
    top_n = article_user.iloc[:n].index
    
    # Get article titles for top-n article ids
    top_articles = df[df['article_id'].isin(top_n)]['title'].unique()
    
    return top_articles # Return the top article titles from df (not df_content)

def get_top_article_ids(n, df):
    '''
    Function returns ids of most popular articles (articles which have
    the largest number of interactions with users)
    
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df which contains user interactions with articles 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    # Count number of iteractions for each article_id
    article_user = df.groupby('article_id')['user_id'].count()
    
    # Sort articles by number of iteractions
    article_user = article_user.sort_values(ascending = False)
    
    # Get top-n article ids
    top_articles = article_user.iloc[:n].index
    top_articles = top_articles.astype(str)
 
    return top_articles # Return the top article ids

def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    # Copy df dataframe and create a dummy column
    user_item = df.copy()
    user_item['cnt'] = 1
    
    # Create matrix with users in rows and articles in columns, if user interacted with article - 1 where
    # user row meets article column and NaN if user didn't interact with the article
    user_item = user_item.groupby(['user_id', 'article_id'])['cnt'].max().unstack()

    # Turn NaN into 0 where there was no interaction between user and article
    user_item = user_item.fillna(0)
    
    return user_item # return the user_item matrix 

def find_similar_users_similarity(user_id, user_item, df):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    similar_users_df - (pandas dataframe) where the closest users (largest dot product users)
                    are listed first which contains user id in neighbor_id column and similarity in 'similarity'
                    column
    
    Description:
    Computes the similarity of every pair of users based on the dot product
    Returns an ordered
    
    '''
    # if user_id is not in df then return empty list of similar users
    if user_id not in df.user_id.unique().tolist():
        return []
    
    # compute similarity of each user to the provided user
    user_articles = np.array(user_item)
    dot_prod_users = user_articles.dot(np.transpose(user_articles))[user_id - 1]
    similarity = pd.DataFrame({'neighbor_id':user_item.index,'similarity':dot_prod_users})

    # sort by similarity
    similar_users_df = similarity.sort_values('similarity', ascending = False)
   
    # remove the own user's id
    similar_users_df = similar_users_df.drop(similar_users_df[similar_users_df.neighbor_id == user_id].index)
       
    return similar_users_df # return a dataframe with user ids and similarity to the user with specified user_id

def find_similar_users(user_id, user_item, df):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest dot product users)
                    are listed first
    
    Description:
    Computes the similarity of every pair of users based on the dot product
    Returns an ordered
    
    '''
    # if user_id is not in df then return empty list of similar users
    if user_id not in df.user_id.unique().tolist():
        return []
   
    # obtain dataframe of users sorted by similarity and their
    most_similar_users = find_similar_users_similarity(user_id, user_item, df)['neighbor_id'].values
       
    return most_similar_users # return a list of the users in order from most to least similar

def get_article_names(article_ids, df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    '''
    article_names = df[df['article_id'].isin(article_ids)]['title'].unique()
    
    return article_names # Return the article names associated with list of article ids


def get_user_articles(user_id, user_item, df):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the doc_full_name column in df_content)
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    # find article_ids user interacted with
    article_ids = user_item.loc[user_id][user_item.loc[user_id] == 1].index.astype(str).tolist()
    
    #find article names for articles user interacted with
    article_names = get_article_names(article_ids, df)
    
    return article_ids, article_names # return the ids and names

def get_number_of_interactions(user_id, df):
    '''
    INPUT:
    user_id - (int)
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    Number of iteractions of the user with given user_id
    '''
    return df[df['user_id'] == user_id]['user_id'].count()

def get_number_of_interactions_for_articles(article_ids, df):
    '''
    INPUT:
    user_id - (int)
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    Dataframe with article_id and number of iteractions for each article
    '''
    df_articles = df.groupby(['article_id']).count().reset_index()
    
    article_interactions = df_articles[df_articles['article_id'].isin(article_ids)]
    article_interactions = article_interactions[['article_id', 'title']].rename(index=str, columns={"title": "num_interactions"})

    return article_interactions

def get_top_sorted_users(user_id, df, user_item):
    '''
    INPUT:
    user_id - (int)
    df - pandas dataframe with article_id, title, user_id columns
    user_item - (pandas dataframe) matrix of users by articles: 
            1's when a user has interacted with an article, 0 otherwise
    
            
    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u
                    
    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe
     
    '''
    # Get dataframe with similar users ids and their similarity sorted by similarity
    neighbors_df = find_similar_users_similarity(user_id, user_item, df)
    
    # Add column with number of interactions in descending order
    neighbors_df['num_interactions'] = neighbors_df.apply(lambda row: get_number_of_interactions(row['neighbor_id'], df),axis=1)
    
    # Sort by similarity then by number of interactions in descending order
    neighbors_df = neighbors_df.sort_values(['similarity', 'num_interactions'], ascending=[False, False])
    
    return neighbors_df # Return the dataframe specified in the doc_string