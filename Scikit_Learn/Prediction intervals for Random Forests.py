from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.datasets import load_boston
boston = load_boston()
X = boston["data"]
Y = boston["target"]
size = len(boston["data"])
trainsize = 400
idx = range(size)

#shuffle the data np.random.shuffle(idx)
rf = RandomForestRegressor(n_estimators=1000, min_samples_leaf=1)
rf.fit(X[idx[:trainsize]], Y[idx[:trainsize]])

#We can now define a function to calculate prediction intervals for every prediction:
def pred_ints(model, X, percentile=95):
    err_down = []
    err_up = []
    for x in range(len(X)):
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict(X[x])[0])
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    return err_down, err_up

#Let’s compute 90% prediction intervals and test how many observations in the test set fall into the interval.
err_down, err_up = pred_ints(rf, X[idx[trainsize:]], percentile=90)
truth = Y[idx[trainsize:]]
correct = 0.
for i, val in enumerate(truth):
    if err_down[i] <= val <= err_up[i]:
        correct += 1
print(correct/len(truth))
