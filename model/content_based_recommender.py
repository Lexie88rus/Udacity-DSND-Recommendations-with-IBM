#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:36:55 2019

@author: aleksandraastakhova

File contains class for making content based recommendations using NLP.
"""

import numpy as np
import pandas as pd
import recommender_helper_functions as hf
import content_based_helpers as cbh

class ContentBasedRecommender():
    '''
    Class which contains methods to make content-based recommendations for
    articles using NLP.
    '''
    
    def fit(self, user_articles_pth, articles_content_pth):
        '''
        Fits the recommender to data which contains details about articles'
        content.
        
        INPUT:
            user_articles_pth - (str) path to dataset, which contains
            information on interactions between users and articles
            
            articles_content_pth - (str) ath to dataset, which contains
            information on content of articles
        '''
        # read dataset with interactions with articles
        self.df = pd.read_csv(user_articles_pth)
        
        # map emails to user_ids
        email_encoded = hf.email_mapper(self.df)
        del self.df['email']
        self.df['user_id'] = email_encoded
        
        self.df_content = pd.read_csv(articles_content_pth)
        
        self.df_new = cbh.prepare_data(self.df_content)
        
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
        
        else:
            # get list of user articles
            user_articles = hf.get_user_articles(user_id, self.user_item, self.df)[0]
        
            # for each user article find similar article using content-based approach
            for article in user_articles:
                similar_articles_int = cbh.get_similar_articles(round(float(article)), self.df_new)
                similar_articles = [str(float(a)) for a in similar_articles_int]
            
                # for each of similar articles append article to recommendations if it is not in recs so far and
                # user didn't interact with this article before
                for sim_article in similar_articles:
                    if (sim_article not in user_articles) and (sim_article not in recs):
                        recs.append(sim_article)
                    
                    if len(recs) >= rec_num:
                        break
                    
                # if found recommendations less than required  add articles from top viewed articles
                if len(recs) < rec_num:
                    top_articles = hf.get_top_article_ids(2 * rec_num)
                    for article in top_articles:
                        # if acticle is not already viewed by the user and not in list already then append the article to recs
                        if (article not in user_articles) and (article not in recs):
                            recs.append(article)
             
                    # if exceed the number of required recommendations then break the loop and return results
                    if len(recs) >= rec_num:
                        break
        
        recs = recs[:rec_num]
    
        # get article names
        rec_names = hf.get_article_names(recs, self.df) 
    
        return recs , rec_names



