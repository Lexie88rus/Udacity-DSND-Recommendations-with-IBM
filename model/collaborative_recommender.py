#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:58:18 2019

@author: aleksandraastakhova

File contains class for making user-user collaborative recommendations.
"""

import numpy as np
import pandas as pd
import recommender_helper_functions as hf

class CollaborativeRecommender():
    '''
    Class which contains methods to make user-user recommendations for
    articles.
    '''
    
    def fit(self, user_articles_pth):
        '''
        Fits the recommender to data which contains interactions between users
        and articles.
        
        INPUT:
            user_articles_pth - (str) path to dataset, which contains
            information on interactions between users and articles
            
        '''
        # read dataset with interactions with articles
        self.df = pd.read_csv(user_articles_pth)
        
        # map emails to user_ids
        email_encoded = hf.email_mapper(self.df)
        del self.df['email']
        self.df['user_id'] = email_encoded
        
        # create user-article matrix
        self.user_item = hf.create_user_item_matrix(self.df)
        
    
    def make_recs(self, user_id, rec_num = 5):
        '''
        Makes recommendations for articles for provided user.
        
        INPUT:
            user_id - id of the user to make recommendations for
            rec_num - number of recommended articles
            
        OUTPUT:
            recs - list of recommended article_ids
            rec_names - list of recommended article names
            
        '''
        recs = [] 
    
        # if provided user has no views then recommend top m most popular articles
        if user_id not in self.df.user_id.unique().tolist():
            recs = hf.get_top_article_ids(rec_num, self.df)
            rec_names = hf.get_article_names(recs, self.df)
        
            return recs, rec_names
    
        # get similar users sorted by similarity and then by number of interactions
        similar_users = hf.get_top_sorted_users(user_id, self.df, self.user_item)['neighbor_id'].values
    
        # get articles user already interacted with
        user_articles = hf.get_user_articles(user_id, self.user_item, self.df)[0]
    
        # loop through similar users
        for similar_user in similar_users:
            # find articles for each user
            recommended_articles = hf.get_user_articles(similar_user, self.user_item, self.df)[0]
        
            # for each article of the user
            for article in recommended_articles:
                # if acticle is not already viewed by the user and not in list already then append the article to recs
                if (article not in user_articles) and (article not in recs):
                    recs.append(article)
        
        # if number of recommendations exceeds required then sort articles by number of interactions
        if (len(recs) > rec_num):
            article_interactions_df = hf.get_number_of_interactions_for_articles(recs, self.df)
            article_interactions_df = article_interactions_df.sort_values(['num_interactions'], ascending = False)
        
            recs = article_interactions_df['article_id'].values
    
        # if number of recommendations is less than required then recommend top viewed articles
        if len(recs) < rec_num:
            top_articles = hf.get_top_article_ids(rec_num * 2, self.df)
            for article in top_articles:
                # if acticle is not already viewed by the user and not in list already then append the article to recs
                if (article not in user_articles) and (article not in recs):
                    recs.append(article)
             
                # if exceed the number of required recommendations then break the loop and return results
                if len(recs) >= rec_num:
                    break
                
        rec_names = hf.get_article_names(recs, self.df)
    
        return recs[:rec_num], rec_names[:rec_num]