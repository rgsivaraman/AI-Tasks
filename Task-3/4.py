import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv('Insurance_dataset.csv')
#plotting the Scatter plot to check relationship between Sal and Temp
df.plot.scatter(x='charges',y='age',title='scatterplot of hours and scores percentages');



