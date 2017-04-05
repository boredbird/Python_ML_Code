from sklearn import  datasets
iris = datasets.load_iris()
data = iris.data
data.shape

digits = datasets.load_digits()
digits.images.shape
import  matplotlib.pyplot as plt
plt.imshow(digits.images[-1],cmap=plt.cm.gray_r)
data = digits.images.reshape((digits.images.shape[0], -1))

#add x,y,x**2,xy,y**2... features
from sklearn.preprocessing import  PolynomialFeatures
import numpy as np
X = np.arange(6).reshape(3,2)
print X
poly = PolynomialFeatures(degree=2)
poly.fit_transform(X)

#This sort of preprocessing can be streamlined with the Pipeline tools
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
model = Pipeline([('poly',PolynomialFeatures(degree=3)),
                  ('linear',LinearRegression(fit_intercept=False))])
#fit to an order-3 polynomial data
x = np.arange(5)
y = 3-2*x+x**2-x**3
model = model.fit(x[:,np.newaxis],y)
print model.named_steps['linear'].coef_


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train) # Don't cheat - fit only on training data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test) # apply same transformation to test data


