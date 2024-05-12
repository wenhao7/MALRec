import numpy as np
import pandas as pd
import math

from ast import literal_eval
from collections import defaultdict
#from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from scipy.sparse.linalg import svds
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from multiprocessing.pool import ThreadPool as Pool
from datetime import datetime
import random

def mask_user_ratings(user_ratings_df, random_state=42, frac=0.5):
    """
    Split user ratings data into input and val splits for evaluating model performance.
    Input split will act as input while model recommendations will be evaluated on the validation split.

    Parameters
    ----------
    user_ratings_df : DataFrame
        Processed user ratings dataframe.
    random_state : int, optional
        Random seed. The default is 42.
    frac : float, optional
        Fraction of dataset to sample for input split. The default is 0.5.

    Returns
    -------
    input_df : DataFrame
        Input split.
    val_df : DataFrame
        Validation split.

    """
    # Sample half of rated titles as input
    input_df = user_ratings_df[user_ratings_df['Rating_Score']>0].sample(frac=frac, random_state=random_state)
    val_df = user_ratings_df.drop(input_df.index)
    return input_df, val_df

def running_avg(scores):
    """
    Calculate cumulative average score during model evaluation

    Parameters
    ----------
    scores : List
        List of scores per evaluated sample.

    Returns
    -------
    avgs : float
        Current average score.

    """
    avgs = np.cumsum(scores)/np.array(range(1, len(scores) + 1))
    return avgs

def select_users(user_ratings_df, min_n=20):
    """
    Select users with at least a minimum number of interactions in their ratings list.

    Parameters
    ----------
    user_ratings_df : DataFrame
        Processed user ratings data.
    min_n : int, optional
        Minimum number of interactions. The default is 20.

    Returns
    -------
    df : DataFrame
        User ratings data containing only users with at least min_n interactions.

    """
    tmp = (user_ratings_df.value_counts('User_Id') >= min_n).reset_index()
    tmp = tmp[tmp['count'] == True]
    df = user_ratings_df[user_ratings_df.User_Id.isin(tmp.User_Id)]
    return df

class ModelEvaluator:
    def evaluate_mrr(self, ranked_rec_df, user_input_df, user_val_df, weight=1, topn=10, left_on='MAL_Id', right_on='Anime_Id'):
        scoring_df = ranked_rec_df.merge(user_val_df, how='left', left_on=left_on, right_on=right_on)
        scoring_df = scoring_df.loc[~scoring_df[right_on].isna()][:topn]
        matched_idx = list(scoring_df[scoring_df[right_on].isin(user_val_df[right_on])].index)
        if not matched_idx:
            return 0
        return (1 * weight) / (matched_idx[0] + 1)
    
    def evaluate_ndcg(self, ranked_rec_df, user_input_df, user_val_df, weight=1, topn=10, left_on='MAL_Id', right_on='Anime_Id'):
        scoring_df = ranked_rec_df.merge(user_val_df, how='left', left_on=left_on, right_on=right_on)
        scoring_df = scoring_df.iloc[:topn]
        # Calculate relevance score based on how well the user interaction went
        for i in range(len(scoring_df)):
            scoring_df['rel'] = 0.0
            scoring_df.loc[scoring_df.Rating_Score == 0, 'rel'] = 0.5
            scoring_df.loc[scoring_df.Rating_Score > 0, 'rel'] = 1
            scoring_df.loc[scoring_df.Rating_Score > 5, 'rel'] = 2
            scoring_df.loc[scoring_df.Rating_Score > 8, 'rel'] = 3
            
        cg, icg = list(scoring_df['rel']) , sorted(scoring_df['rel'], reverse=True)            
        if not cg or max(cg) == 0:
            return 0
        icg = sorted(cg, reverse=True)
        cg = list(np.array(cg) / np.array([math.log(i+1, 2) for i in range(1,len(cg) + 1)]))
        icg = list(np.array(icg) / np.array([math.log(i+1, 2) for i in range(1,len(icg) + 1)]))
        ndcg = sum(cg) / sum(icg)
        return ndcg
    
    def evaluate(self, rec_object, df_content, df_ratings, number_of_samples=30, topn=10, left_on="Anime_Id", random_state=1):
        mrr_list, ndcg_list = [], []
        s = datetime.now()
        for i in np.random.choice(df_ratings.User_Id.unique(), number_of_samples, replace=False):
            user_input_df, user_val_df = mask_user_ratings(df_ratings[df_ratings['User_Id'] ==i], random_state=random_state)
            if len(user_input_df) == 0:
                continue
            pred_df = rec_object.predict(user_input_df, 10)
            mrr = self.evaluate_mrr(pred_df, user_input_df, user_val_df, topn=10, left_on="Anime_Id")
            ndcg = self.evaluate_ndcg(pred_df, user_input_df, user_val_df, topn=10, left_on="Anime_Id")

            mrr_list.append(mrr)
            ndcg_list.append(ndcg)
        s = datetime.now() - s
        return mrr_list, ndcg_list, s
    
class PopularityRec:
    def __init__(self, df, anime_info_df=None):
        self.popularity = df
        self.anime_info = anime_info_df
        self.name = 'Popularity'
        
    def predict(self, user_ratings_df, topn=10, left_on='MAL_Id', right_on='Anime_Id'):
        rec_df = self.popularity.sort_values('Members', ascending=False)
        rec_df = rec_df.merge(user_ratings_df, how='left', left_on=left_on, right_on=right_on)
        return rec_df.loc[rec_df[right_on].isna()][self.popularity.columns][:topn]
    
class ContentBasedRecommender:
    def __init__(self, df_content):
        self.df_content = df_content
        self.df_dense, self.sparse_vec, self.sparse_tfidf = self.process_df(self.df_content)
        self.ref_weights = [1/math.log(len(self.df_content)-i+1, 10) + 1 for i in range(len(self.df_content))]        
        self.name = 'Content'
        
    def process_df(self, df_content):
        
        explode_cols = ['Genres','Themes','Demographic']
        for col in explode_cols:
            try:
                tmp = df_content[col].apply(literal_eval).explode()
            except:
                tmp = df_content[col].explode()
            tmp = str(col).lower() + '_' + tmp
            tmp = tmp.fillna(str(col).lower() + '_na')
            df_content = df_content.drop(col, axis=1).join(pd.crosstab(tmp.index, tmp))
        
        #labelencode
        for col in ['Type','Source']:
            le=LabelEncoder()
            df_content[col] = le.fit_transform(df_content[col])
        #Vectorize
        sparse_vec=[]
        #for col in ['Name','Producers','Licensors','Studios','Voice_Actors']:
        for col in ['Name','Studio']:
            df_content[col].apply(lambda x: '' if pd.isna(x) else x.strip('[]'))
            vec = CountVectorizer()
            tmp = df_content[col]
            sparse_tmp = vec.fit_transform(tmp)
            if isinstance(sparse_vec,list):
                sparse_vec = sparse_tmp
            else:
                sparse_vec = hstack((sparse_vec, sparse_tmp))
        
        tfidf_vec = TfidfVectorizer(analyzer='word',
                            ngram_range=(1,2),
                            max_df=0.5,
                            min_df=0.001,
                            #stop_words=stopwords.words('english')
                            )
        sparse_tfidf = tfidf_vec.fit_transform(df_content['Synopsis'])
        
        #df_dense = df_content.drop(['Name','Producers','Licensors','Studios','Voice_Actors','Synopsis'], axis=1)
        df_dense = df_content.drop(['Image','Start_Date','18plus'], axis=1)
        df_dense = df_dense.drop(['Name','Studio','Synopsis'], axis=1)
        df_dense.Episodes = df_dense.Episodes.fillna(0)
        #scale_cols = ['Score','Members','Favorites','Episodes']
        scale_cols = ['Score','Members','Episodes','Duration']
        ss = StandardScaler()
        for col in scale_cols:
            df_dense[col] = pd.to_numeric(df_dense[col], errors='coerce')
            df_dense[col] = df_dense[col].fillna(0)
        df_dense[scale_cols] = ss.fit_transform(df_dense[scale_cols])
        
        return df_dense, sparse_vec, sparse_tfidf
    
        self.df_dense = df_dense
        self.sparse_vec = sparse_vec
        self.sparse_tfidf = sparse_tfidf
    
    def get_entry(self, MAL_Id):
        title_dense = self.df_dense[self.df_dense['MAL_Id'] == MAL_Id]
        idx = title_dense.index[0]
        title_vec = self.sparse_vec[idx]
        title_tfidf = self.sparse_tfidf[idx]
        return title_dense, title_vec, title_tfidf
    
    def calc_sim(self, MAL_Id):
        try:
            title_dense, title_vec, title_tfidf = self.get_entry(MAL_Id)
        except:
            return None
        sim_dense = cosine_similarity(title_dense, self.df_dense)
        sim_vec = cosine_similarity(title_vec, self.sparse_vec)
        sim_tfidf = cosine_similarity(title_tfidf, self.sparse_tfidf)
        total = (sim_dense + sim_vec + sim_tfidf).argsort().flatten()
        return total
    
    
    def predict_weights(self, user_list):
        weights_df = pd.DataFrame({'Preds': self.df_content.MAL_Id, 'Weights':0})
        for MAL_Id in user_list:
            recs = self.calc_sim(MAL_Id)
            if recs is None:
                continue
            idx_recs = list(recs)
            weights_zip = list(zip(idx_recs, self.ref_weights))
            weights_zip = sorted(weights_zip)
            weights_zip = list(zip(*weights_zip))
            weights_df['Weights'] += weights_zip[1]
        weights_df['Weights'] = (weights_df['Weights'] - weights_df['Weights'].min()) / (weights_df['Weights'].max() - weights_df['Weights'].min())
        return weights_df
    
    def par_weights(self, user_list):
        weights_df = pd.DataFrame({'Preds': self.df_content.MAL_Id, 'Weights':0})
        recs_list=[]
        with Pool() as pool:
            for recs in pool.imap(self.calc_sim, user_list):
                if recs is None:
                    continue
                recs_list.append(recs)
        for recs in recs_list:
            idx_recs = list(recs)
            weights_zip = list(zip(idx_recs, self.ref_weights))
            weights_zip = sorted(weights_zip)
            weights_zip = list(zip(*weights_zip))
            weights_df['Weights'] += weights_zip[1]
        weights_df['Weights'] = (weights_df['Weights'] - weights_df['Weights'].min()) / (weights_df['Weights'].max() - weights_df['Weights'].min())
        return weights_df
    
    def par_predict(self, user_df, topn=10):
        user_list = list(user_df['Anime_Id'])
        weights_df = self.par_weights(user_list)
        res = weights_df.merge(self.df_content, how='left', left_on='Preds', right_on='MAL_Id')
        res = res.sort_values('Weights', ascending=False).loc[~res['MAL_Id'].isin(user_list)][:topn]
        return res
    
    def predict(self, user_df, topn=10):
        user_list = list(user_df['Anime_Id'])
        weights_df = self.predict_weights(user_list)
        res = weights_df.merge(self.df_content, how='left', left_on='Preds', right_on='MAL_Id')
        res = res.sort_values('Weights', ascending=False).loc[~res['MAL_Id'].isin(user_list)][:topn]
        return res        
    

class CollaborativeRecommender:
    def __init__(self, df_cf):
        self.df_original = df_cf
        self.df_anime_id = df_cf.groupby(['Anime_Id','Anime_Title']).count().reset_index()[['Anime_Id','Anime_Title']]
        self.name = 'Collaborative'       
        
    def process_df(self, df):
        df['Mean_Score'] = 0
        mean_df = df[df['Rating_Score']>0].groupby("User_Id")['Rating_Score'].mean().reset_index().rename(columns={'Rating_Score':'mean_score'})
        df = df.merge(mean_df)
        df['Interactions'] = 0.0
        df.loc[df.Rating_Score == 0, 'Interactions'] = 2
        df.loc[df.Rating_Score-df.Mean_Score < 0, 'Interactions'] = 1
        df.loc[df.Rating_Score-df.Mean_Score == 0, 'Interactions'] = 3
        df.loc[df.Rating_Score-df.Mean_Score > 0, 'Interactions'] = 4
        df = df.pivot(index='User_Id', columns='Anime_Id', values='Interactions').fillna(0)
        return df
    
    def predict_dec(self, user_df, k=15):
        max_uid = self.df_original.User_Id.max()
        for i, uid in enumerate(user_df.User_Id.unique()):
            user_df.loc[user_df.User_Id==uid, 'User_Id'] = max_uid + 1 + i
        user_df = pd.concat([self.df_original, user_df])
        user_cf = self.process_df(user_df)
        sparse_cf = csr_matrix(user_cf)
        U, sigma, Vt = svds(sparse_cf)
        return U, sigma, Vt, user_cf.columns, user_cf.index
    
    def predict(self, user_df, topn=10, k=15):
        # Reconstruct matrix to find similarities
        U, sigma, Vt, new_col, new_index = self.predict_dec(user_df, k)
        sigma = np.diag(sigma)
        all_ratings = np.dot(np.dot(U,sigma), Vt)
        all_ratings = (all_ratings - all_ratings.min()) / (all_ratings.max() - all_ratings.min())
        
        # Construct output dataframe, collecting weights from the number of user we have predicted on
        df_cf_pred = pd.DataFrame(all_ratings, columns=new_col, index=new_index)     
        num_users = user_df.User_Id.nunique()
        res = df_cf_pred.iloc[-num_users:].T
        if num_users == 1:
            res = res.sort_values(res.columns[0],ascending=False).reset_index()
            res = res.loc[~res['Anime_Id'].isin(user_df['Anime_Id'])][:topn]
        else:
            res = res.reset_index()
            res = res.loc[~res['Anime_Id'].isin(user_df['Anime_Id'])]
        return res
    
    
class HybridRecommender:
    def __init__(self, cb_model, cf_model, df_content, df_ratings, cb_weight=0.5):
        self.cb_model = cb_model(df_content)
        self.cf_model = cf_model(df_ratings)
        self.cb_weight = cb_weight
        self.cf_weight = 1 - cb_weight
        self.n = df_ratings.Anime_Id.nunique()
        self.name = 'Hybrid'
    
    def predict(self, user_df, topn=10):   
        num_users = user_df.User_Id.nunique()
        cb_pred = self.cb_model.predict(user_df, self.n)
        cf_pred = self.cf_model.predict(user_df, self.n)
        
        # Normalize scores from both predictions
        ss = StandardScaler()
        cb_pred['ss'] = ss.fit_transform(cb_pred['Weights'].values.reshape(-1,1))
        cf_cols = ['ss_' + str(col) for col in cf_pred.columns[-1:]]
        if num_users == 1:
            cf_pred[cf_cols] = ss.fit_transform(cf_pred[cf_pred.columns[-1]].values.reshape(-1,1))
        else:
            cf_pred[cf_cols] = ss.fit_transform(cf_pred[cf_pred.columns[1:]])
        
        combined_pred = cf_pred.merge(cb_pred[['ss','MAL_Id']], how='left', left_on='Anime_Id', right_on='MAL_Id')
        combined_pred['Final_Score'] = self.cf_weight*combined_pred[cf_cols].sum(axis=1) + self.cb_weight*combined_pred['ss']
        combined_pred = combined_pred.sort_values('Final_Score', ascending=False)
        return combined_pred[['MAL_Id','Final_Score']]#combined_pred[:topn]