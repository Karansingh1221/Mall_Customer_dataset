# Mall_Customer_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#load the dataset
data=pd.read_csv('/content/Mall_Customers.csv')
data.head(10)

count=data.isnull().sum()#checking for null values 
count

data.columns=['Customer id','Gender','Age','Annual Income','Spending Score']#renaming the columns
data.head()

data['Gender']=data['Gender'].map({'Male':0,'Female':1})#converting categorical to numerical data
data.head()#Data preprocessing is completed

data.describe()

#defing the function for plotting histogram
def visualize(column):
  plt.figure(figsize=(10,6))
  sns.histplot(data[column],bins=10,kde=True)
  plt.title(f'{column} Distribution')
  plt.xlabel(column)
  plt.ylabel('Frequency')
  plt.show()


visualize('Age')
visualize('Annual Income')
visualize('Spending Score')
plt.figure(figsize=(10,6))
sns.scatterplot(data=data,x='Annual Income',y='Spending Score',hue='Gender')
plt.title('Annual Income vs Spending Score')
plt.show()

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#feature Selection
features=data[['Age','Annual Income','Spending Score']]

#Scaling the featured data
scaler=StandardScaler()
scaled_features=scaler.fit_transform(features)

#tApplying kmeans clustering
kmeans=KMeans(n_clusters=5,random_state=42)
data['Cluster']=kmeans.fit_predict(scaled_features)

#Evaluating cluster quality
plt.figure(figsize=(10,6))
sns.scatterplot(data=data,x='Annual Income',y='Spending Score',hue='Cluster',palette='viridis')
plt.title('Customer Segments')
plt.show()

