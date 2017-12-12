## LOOK AT http://scikit-learn.org/stable/modules/preprocessing.html


#A diffculty is that different algorithms make different assumptions about your data 
#and may require different transforms. Further, when you follow all of the rules 
#and prepare your data, sometimes algorithms can deliver better results without pre-processing.
# 1. Load the dataset from a URL.
# 2. Split the dataset into the input and output variables for machine learning. 
# 3. Apply a pre-processing transform to the input variables.
# 4. Summarize the data to show the change.

from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer 

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8] ## Input
Y = array[:, 8]  ## Output

# MinMaxScaler -- Scaling features to a range
# Transforms features by scaling each feature to a given range.
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
# Scaling features to a range
print("### MinMaxScaler ###")
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(X)
set_printoptions(precision=3)
print(rescaledX[0:5,:])

# StandardScaler
# Standardized values (also called standard scores or normal deviates) are the same thing as z-scores. 
# A standardized value is what you get when you take a data point and scale it by population data. 
# It tells us how far from the mean we are in terms of standard deviations
# z = (observation - mean)/sigma
#For instance, suppose you went to college in New York and your best friend went to college in Georgia. 
#You might get a grade of 87 in a test with a mean of 77 and a standard deviation of 5, 
# and the same day your friend might get a grade of 612 (mean 600, standard deviation 100). 
# Although the two grades (87 and 612) cant be compared directly the standardized values will 
# allow you to immediately see who is doing better compared with the rest of the class.
print("### StandardScaler ###")
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
print(rescaledX[0:5,:])


print("### Normalizer ###")
scaler = Normalizer().fit(X)
rescaledX = scaler.transform(X)
print(rescaledX[0:5,:])

print("### Binarizer ###")
scaler = Binarizer().fit(X)
rescaledX = scaler.transform(X)
print(rescaledX[0:5,:])