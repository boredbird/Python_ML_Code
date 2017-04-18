#Import Library
from sklearn import decomposition
 
#Assumed you have training and test data set as train and test
# Create PCA obeject pca= decomposition.PCA(n_components=k) #default value of k =min(n_sample, n_features)
# For Factor analysis
#fa= decomposition.FactorAnalysis()
# Reduced the dimension of training dataset using PCA
train_reduced = pca.fit_transform(train)
 
#Reduced the dimension of test dataset
test_reduced = pca.transform(test)
 
#For more detail on this, please refer  this link.
