#Import Library
from sklearn import svm
 
#Assumed you have, X (predic
tor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object
model = svm.svc() # there is various option associated with it, this is simple for classification. You can refer link, for mo# re detail.
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
 
#Predict Output
predicted= model.predict(x_test)
