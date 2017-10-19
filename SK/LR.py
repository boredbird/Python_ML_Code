from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
model = LogisticRegression()

a = load_iris()
X = a.data
y = a.target
model.fit(X, y)
model.score(X, y)

print('Coefficient: n', model.coef_)
print('Intercept: n', model.intercept_)

# predicted= model.predict(x_test)
