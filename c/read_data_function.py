import pandas
from pandas import read_csv


def get_dataframe_ratings_base(text):
    r_cols = ['user_id', 'item_id', 'rating']
    ratings = pandas.read_csv(text, sep='\t', names=r_cols, encoding='latin-1')
    Y_data = ratings.values
    return Y_data


def get_name_movie(text):
    r_cols = ['name', 'year', 'imdb', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19','20','21','22']
    list_movie = pandas.read_csv(text, sep='|', names=r_cols, encoding='latin-1')
    list_name_movie = list_movie['name'].values
    return list_name_movie


def get_year_movie(text):
    r_cols = ['name', 'year', 'imdb', '3','4', '5', '6','7', '8', '9','10', '11', '12','13', '14', '15','16','17', '18', '19','20','21','22']
    list_movie = pandas.read_csv(text, sep='|', names=r_cols, encoding='latin-1')
    list_year_movie = list_movie['year'].values
    return list_year_movie

