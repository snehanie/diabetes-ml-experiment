#peekAtYourData
from pandas import read_csv
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
print("========================================================================================================================")
print("peekAtYourData")
print("========================================================================================================================")
peek = data.head(20)
print(peek)

#lookAtDimensionsOfData
print("========================================================================================================================")
print("lookAtDimensionsOfData")
print("========================================================================================================================")
shape = data.shape
print(shape)


#dataTypeForEachAttribute\
print("========================================================================================================================")
print("dataTypeForEachAttribute")
print("========================================================================================================================")
dtypes = data.dtypes
print(dtypes)


#DescriptiveStatus
print("========================================================================================================================")
print("DescriptiveStatus")
print("========================================================================================================================")
description = data.describe()
print(description)

#classDistribution (Classification Only)
print("========================================================================================================================")
print("classDistribution")
print("========================================================================================================================")

class_counts = data.groupby('class')
print(class_counts)
#Correlation between attributes
# Correlation refers to the relationship between two variables and 
# how they may or may not change together. The most common method 
# for calculating correlation is Pearsons Correlation Coefficient
# that assumes a normal distribution of the attributes involved. 
# A correlation of -1 or 1 shows a full negative or positive 
# correlation respectively. Whereas a value of 0 shows no 
# correlation at all. Some machine learning algorithms like linear 
# and logistic regression can suffer poor performance if there are 
# highly correlated attributes in your dataset. As such, it is a 
# good idea to review all of the pairwise correlations of the 
# attributes in your dataset.
print("========================================================================================================================")
print("Correlations")
print("========================================================================================================================")

correlations = data.corr(method='pearson')
print(correlations)

#Skew of Univariate Distributions
print("========================================================================================================================")
print("Skew of Univariate Distributions")
print("========================================================================================================================")
skew = data.skew()
print(skew)
