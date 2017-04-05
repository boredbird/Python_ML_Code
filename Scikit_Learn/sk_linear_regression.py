from sklearn import  datasets
import numpy as np
diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
print(regr.coef_)

# The mean square error
print np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)
# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and y.
print regr.score(diabetes_X_test, diabetes_y_test)

X = np.c_[ .5, 1].T
y = [.5, 1]
test = np.c_[ 0, 2].T

#linear regression
regr = linear_model.LinearRegression()
import matplotlib.pyplot as plt
plt.figure()
np.random.seed(0)
for _ in range(6):
    this_X = .1 * np.random.normal(size=(2, 1)) + X
    regr.fit(this_X, y)
    plt.plot(test, regr.predict(test))
    plt.scatter(this_X, y, s=3)

#ridge regression
#the larger the ridge alpha parameter, the higher the bias and the lower the variance
regr = linear_model.Ridge(alpha=.1)
plt.figure()
np.random.seed(0)
for _ in range(6):
    this_X = .1 * np.random.normal(size=(2, 1)) + X
    regr.fit(this_X, y)
    plt.plot(test, regr.predict(test))
    plt.scatter(this_X, y, s=3)

#lasso regression
alphas = np.logspace(-4, -1, 6)
regr = linear_model.Lasso()
scores = [regr.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train).score(diabetes_X_test, diabetes_y_test) for alpha in alphas]
best_alpha = alphas[scores.index(max(scores))]
regr.alpha = best_alpha
regr.fit(diabetes_X_train, diabetes_y_train)
print(regr.coef_)

#logistic regression
logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(iris_X_train, iris_y_train)

#The C parameter controls the amount of regularization in the LogisticRegression object: a large value
#for C results in less regularization. penalty="l2" gives Shrinkage (i.e. non-sparse coefficients), while
#penalty="l1" gives Sparsity

