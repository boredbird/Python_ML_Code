import graphlab as gl
url = 'http://s3.amazonaws.com/dato-datasets/movie_ratings/training_data.csv'
data = gl.SFrame.read_csv(url, column_type_hints={"rating":int})
data.show()
model = gl.recommender.create(data, user_id="user", item_id="movie", target="rating")
results = model.recommend(users=None, k=5)
