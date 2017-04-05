#Import Library
from sklearn.ensemble import RandomForestClassifier
 
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Random Forest object
model= RandomForestClassifier()
 
# Train the model using the training sets and check score
model.fit(X, y)
 
#Predict Output
predicted= model.predict(x_test)
