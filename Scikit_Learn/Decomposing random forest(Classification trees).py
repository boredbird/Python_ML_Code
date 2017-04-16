#Exactly the same method can be used for classification trees,
#where features contribute to the estimated probability of a given class

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris iris = load_iris()
rf = RandomForestClassifier(max_depth = 4)
idx = range(len(iris.target))
np.random.shuffle(idx)

rf.fit(iris.data[idx][:100], iris.target[idx][:100])

#Let’s predict now for a single instance.
instance = iris.data[idx][100:101]
print(rf.predict_proba(instance))

#Breakdown of feature contributions:
prediction, bias, contributions = ti.predict(rf, instance)
print("Prediction", prediction)
print("Bias (trainset prior)", bias)
print("Feature contributions:")
for c, feature in zip(contributions[0], iris.feature_names):
    print(feature, c)
