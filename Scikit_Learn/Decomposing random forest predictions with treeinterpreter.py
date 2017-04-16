from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.datasets import load_boston
boston = load_boston()
rf = RandomForestRegressor()
rf.fit(boston.data[:300], boston.target[:300])

#Lets pick two arbitrary data points that yield different price estimates from the model.
instances = boston.data[[300, 309]]
print ("Instance 0 prediction:", rf.predict(instances[0]))
print ("Instance 1 prediction:", rf.predict(instances[1]))

#decompose the predictions into the bias term (which is just the trainset mean) and individual feature contributions,
#so we see which features contributed to the difference and by how much.
#simply call the treeinterpreter predict method with the model and the data
prediction, bias, contributions = ti.predict(rf, instances)

#Printint out the results:
for i in range(len(instances)):
    print "Instance", i
    print "Bias (trainset mean)", biases[i]
    print "Feature contributions:"
    for c, feature in sorted(zip(contributions[i], boston.feature_names), key=lambda x: -abs(x[0])):
        print(feature, round(c, 2))
    print("-"*20 ) 

#Comparing too datasets
ds1 = boston.data[300:400]
ds2 = boston.data[400:]
print (np.mean(rf.predict(ds1)))
print (np.mean(rf.predict(ds2)))

#break down the contributors to this difference: which features contribute to this different and by how much.
prediction1, bias1, contributions1 = ti.predict(rf, ds1)
prediction2, bias2, contributions2 = ti.predict(rf, ds2)

#just print out the differences of the contributions in the two datasets
for c, feature in sorted(zip(totalc1 - totalc2, boston.feature_names), reverse=True):
    print feature, round(c, 2)


