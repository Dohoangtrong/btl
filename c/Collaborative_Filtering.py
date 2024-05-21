import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import read_data_function as read_data_function

class Collaborative_Filtering(object):
    def __init__(self, data_matrix, k, dist_func=cosine_similarity, uuCF=1):
        self.uuCF = uuCF 
        self.Y_data = data_matrix if uuCF else data_matrix[:, [1, 0, 2]]
        self.k = k
        self.dist_func = dist_func
        self.Ybar_data = None
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    def add(self, new_data):
        self.Y_data = np.concatenate((self.Y_data, new_data), axis=0)

    def normalize_matrix(self):
        users = self.Y_data[:, 0]
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        for n in range(self.n_users):
            ids = np.where(users == n)[0].astype(np.int32)
            ratings = self.Y_data[ids, 2]
            m = np.mean(ratings)
            if np.isnan(m):
                m = 0 
            self.mu[n] = m
            self.Ybar_data[ids, 2] = ratings - self.mu[n]
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
                                       (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)

    def refresh(self):
        self.normalize_matrix()
        self.similarity()

    def fit(self):
        self.refresh()

    def __pred(self, u, i, normalized=1):
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        sim = self.S[u, users_rated_i]
        a = np.argsort(sim)[-self.k:]
        nearest_s = sim[a]
        r = self.Ybar[i, users_rated_i[a]]
        if normalized:
            return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8)

        return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8) + self.mu[u]

    def recommend_top(self, u, top_x):
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        item = {'id': None, 'similar': None}
        list_items = []

        def take_similar(elem):
            return elem['similar']

        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                item['id'] = i
                item['similar'] = rating
                list_items.append(item.copy())

        sorted_items = sorted(list_items, key=take_similar, reverse=True)
        sorted_items.pop(top_x)
        return sorted_items

if __name__ == "__main__":
    data_matrix = read_data_function.get_dataframe_ratings_base('ub.base')
    cf_rs = Collaborative_Filtering(data_matrix, k=2, uuCF=1)
    cf_rs.fit()
    list_name_movie = read_data_function.get_name_movie('u.item')
    list_year_movie = read_data_function.get_year_movie('u.item')
    # nhập id của người dùng
    user = int(input("Nhập id của người dùng:\n"))
    list_movies = cf_rs.recommend_top(user,200)
    for i in range(200):
        print( list_movies[i]['similar'], 2,
                              list_name_movie[list_movies[i]['id']],
                              list_year_movie[list_movies[i]['id']] )
