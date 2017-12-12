from matplotlib import pyplot
from pandas import read_csv
import numpy
from pandas.plotting import scatter_matrix

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

#Histograms
print("========================================================================================================================")
print("Histogram Plots")
print("========================================================================================================================")
print("""Histograms group data into bins and provide you a count of the number of observations in each bin. 
From the shape of the bins you can quickly get a feeling for whether an attribute is Gaussian, skewed or even has an exponential distribution. 
It can also help you see possible outliers.""")

print "\n\n"
print("""We can see that perhaps the attributes age, pedi and test may have an exponential distribution. 
We can also see that perhaps the mass and pres and plas attributes may have a Gaussian or nearly Gaussian distribution. 
This is interesting because many machine learning techniques assume a Gaussian univariate distribution on the input variables.
""")

data.hist()
pyplot.show()


#DensityPlots
print("========================================================================================================================")
print("DensityPlots")
print("========================================================================================================================")
print("""Density plots are another way of getting a quick idea of the distribution of each attribute. 
The plots look like an abstracted histogram with a smooth curve drawn through the top of each bin, 
much like your eye tried to do with the histograms.
	""")
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False) 
pyplot.show()


#BoxAndWhiskerPlots
print("========================================================================================================================")
print("BoxAndWhiskerPlots")
print("========================================================================================================================")
print("""Boxplots summarize the distribution of each attribute, drawing a line for the median (middle value) 
and a box around the 25th and 75th percentiles (the middle 50% of the data). 
The whiskers give an idea of the spread of the data and dots outside of the whiskers show candidate 
outlier values (values that are 1.5 times greater than the size of spread of the middle 50% of the data).
""")
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()

#correlationsMatrix
print("========================================================================================================================")
print("correlationsMatrix")
print("========================================================================================================================")
correlations = data.corr()
# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

#ScatterPlotMatrix
print("========================================================================================================================")
print("ScatterPlotMatrix")
print("========================================================================================================================")
scatter_matrix(data)
pyplot.show()
