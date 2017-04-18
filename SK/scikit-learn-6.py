from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data = datasets.load_boston()
data_X = loaded_data.data 
data_Y = loaded_data.target

model = LinearRegression()
model.fit(data_X,data_Y)

print(model.predict(data_X[:4,:]))
print(data_Y[:4])

print(model.coef_)
print(model.intercept_)
print(model.score(data_X,data_Y))
print(model.get_params())