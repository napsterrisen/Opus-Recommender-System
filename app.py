import urllib.request
import numpy as np
import pandas as pd
import csv
from flask import Flask, request, jsonify, render_template
import pickle
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel,sigmoid_kernel
# import book_recommender
# from book_recommender import get_recom

app = Flask(__name__)

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
    img_arr=[]
    rec_arr = []

    for i in book_indices:
        rec_arr.append({'title': black['book_title'].iloc[i], 'link': black['image_url'].iloc[i]})
        
    return rec_arr

data = pd.read_csv('black.csv')
suggestions = []

for x in data['book_title']:
    suggestions.append(x)


def get_suggestions():
    data = pd.read_csv('black.csv')
    return list(data['book_title'].str.capitalize())


@app.route('/')
def home():
    title_suggestion_arr = []
    
    # for x in data['book_title']:
    #     title_suggestion_arr.append(x)

    # for x in data['image_url']:
    #     url_suggestion_arr.append(x)

    return render_template('main.html', suggestions=suggestions, img_url="")

@app.route('/search', methods=['POST'])
def search():
    '''
    For rendering results on HTML GUI
    '''
    data = pd.read_csv('black.csv')
    # user_input = 100
    user_input = request.form.get('search_box')
    # return render_template('main.html', img_url=user_input)
    # if(data['book_title'] == "Harry Potter and the Philosopher's Stone"])
    #     res = data
    value = [user_input]
    # value.insert(0,user_input)
    image_url = data[data.book_title.isin(value)]['image_url']
    image_url = image_url.values
    author_name = data[data.book_title.isin(value)]['book_authors']
    author_name = author_name.values
    # publication_year = data[data.book_title.isin(value)]['original_publication_year']
    # publication_year = publication_year.values
    # avg_rating = data[data.book_title.isin(value)]['average_rating']
    # avg_rating = avg_rating.values
    genres = data[data.book_title.isin(value)]['genres']
    genres = genres.values
    book_rating = data[data.book_title.isin(value)]['book_rating']
    book_rating = book_rating.values
    book_desc = data[data.book_title.isin(value)]['book_desc']
    book_desc = book_desc.values
    # img_url_val = ""
    # author_name_val = ""
    # publication_year_val = ""
    # avg_rating_val = ""

    for x in image_url:
        img_url_val = x

    for x in author_name:
        author_name_val = x

    publication_year_val=""
    avg_rating_val=""
    # for x in publication_year:
    #     publication_year_val = x

    # for x in avg_rating:
    #     avg_rating_val = x

    for x in genres:
        genres_val = x

    for x in book_rating:
        book_rating_val = x

    for x in book_desc:
        book_desc_val = x
    # link = "https://en.wikipedia.org/wiki/Suzanne_Collins"
    # f = urllib.request.urlopen(link)
    # the_page = f.read()
    # getpage = requests.get("https://en.wikipedia.org/wiki/Suzanne_Collins")


    # getpage_soup = BeautifulSoup(getpage.text, 'html.parser')

    # all_id_para1 = getpage_soup.findAll('table', {'class': 'infobox'})
    # the_page = ""

    # for para in all_id_para1:
    #     the_page = para
    # img_url_val=""
    # author_name_val=""
    # publication_year_val=""
    # avg_rating_val=""
    # the_page=""
    recom_arr = get_recom(user_input, sig)
    # recom_arr = []
    # for recom_val in recom_arr:
    #     rec_movies = recom_val
# , image_suggestions=recom_arr, recom_length = len(recom_arr)
    return render_template('main.html', img_url=img_url_val, author=author_name_val, publ_year=publication_year_val, 
    avg_rating=avg_rating_val, genres=genres_val, book_rating=book_rating_val, book_desc=book_desc_val, suggestions=suggestions, recommended_list=recom_arr)

if __name__ == "__main__":
    app.run(debug=True)
