import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel,sigmoid_kernel

# red=pd.read_csv('black.csv')
black=pd.read_csv('black.csv')

tf=TfidfVectorizer()
tf_mat=tf.fit_transform(black.soup)

sig=sigmoid_kernel(tf_mat,tf_mat)
idx=pd.Series(black.index,index=black['book_title']).drop_duplicates()

def get_recom(title,sig=sig):
    idcs=idx[title]
    sim_sc=list(enumerate(sig[idcs]))
    sim_sc=sorted(sim_sc,key=lambda x: x[1],reverse=True)
    sim_sc=sim_sc[1:21]
    book_indices=[i[0] for i in sim_sc]
    wss=[]
    for i in book_indices:
        wss.append(black['book_title'].iloc[i])
        
    return wss

# get_recom("Ender's Game")

# red=pd.read_csv('book_data.csv')
# red.drop(red[red['book_desc'].isna()].index,inplace=True)
# red=red.drop(['book_edition','book_isbn','image_url','book_pages'],1)
# red=red.dropna()

# def to_string(x):
#     if x:
#         return x.replace('|',' ')
#     else:
#         return ''
    
# red['book_authors']=red['book_authors'].apply(lambda x: to_string(x))
# red['genres']=red['genres'].apply(lambda x: to_string(x))

# def soup(x):
#     return x.book_authors+' '+x.genres

# red['soup']=red.apply(soup, 1)

# black = (red.sample(replace = False,frac = 0.2).sort_index().reset_index())
# black.drop_duplicates(subset ="book_title",keep = False, inplace = True) 
# black.to_csv('black.csv')

# black=pd.read_csv('black.csv')


# from sklearn.feature_extraction.text import TfidfVectorizer
# tf=TfidfVectorizer()
# tf_mat=tf.fit_transform(black.soup)

# sig=sigmoid_kernel(tf_mat,tf_mat)
# idx=pd.Series(black.index,index=black['book_title']).drop_duplicates()

# def get_recom(title,sig=sig):
#     sim_sc=list(enumerate(sig[idx[title]]))
#     sim_sc=sorted(sim_sc,key=lambda x: x[1],reverse=True)
#     sim_sc=sim_sc[1:21]
#     book_indices=[i[0] for i in sim_sc]
#     wss=[]
#     for i in book_indices:
#         wss.append(black['book_title'].iloc[i])
        
#     return wss


