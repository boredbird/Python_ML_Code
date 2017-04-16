#Import Library
from sklearn.ensemble import GradientBoostingClassifier
 
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Gradient Boosting Classifier object
model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
 
# Train the model using the training sets and check score
model.fit(X, y)
 
#Predict Output
predicted= model.predict(x_test)
