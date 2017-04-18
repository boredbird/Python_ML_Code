from sklearn.learning_curve import validation_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
import numpy as np 

digits = load_digits()
X = digits.data 
y = digits.target

param_range = np.logspace(-6,-2.3,5)
train_loss,test_loss = validation_curve(
    SVC(),X,y,param_name='gamma',param_range=param_range,
    cv=10,scoring='mean_squared_error')

train_loss_mean = -np.mean(train_loss,axis=1)
test_loss_mean = -np.mean(test_loss,axis=1)

plt.plot(param_range,train_loss_mean,'o-',color="r",label="Training")
plt.plot(param_range,test_loss_mean,'o-',color="g",label="Cross-Validation")

plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()