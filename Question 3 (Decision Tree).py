#!/usr/bin/env python
# coding: utf-8

# # Data Mining Final Exam Part 2
# 
#     Name: Aimi Nabilah Hassin
#     Matric No: WQD180105/17198801

# ### Question 3
# 
# In this question, I am using the airbnb dataset from Question 1. The aim for this question is to predict the price for the listing house on airbnb. A predictive model based on Decision Tree algorithm shall be built in order to perform the price prediction.

# #### Importing Required Libraries

# In[5]:


import pandas as pd
import numpy as np


# #### Loading Data from Question 1

# In[6]:


df = pd.read_csv('airbnb.csv')
df.sample(5)


# In[7]:


print(df.describe())


# In[8]:


# Checking for any null value
(df.isnull()).any()


# In[11]:


# Plotting the graph for the price's distribution
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')

pd.plotting.register_matplotlib_converters() 
sns.distplot(a=df['price'])


# In[12]:


df_set = df.loc[df.price < 300] 
sns.distplot(a=df_set['price'])


# **Comment:** From the illustrated distribution graph above, we can observe that the price distribution for airbnb in Manhanttan and Orlando is ranging between USD25-USD300 per night, and  highest distribution falls under USD70.

# #### Feature Selection
# 
# Only important features are selected from the dataset for further processing.

# In[13]:


df_features = ['no_of_reviews', 'rating', 'no_of_guest', 'bedrooms_no']
X = df[df_features] # Features
y = df.price # Target variable

X.sample(5)


# In[14]:


# Finding the correlation between the features and target variable
relevant_parameters = ['price'] + df_features
sns.pairplot(df.loc[df.price < 300][relevant_parameters], hue="rating")


# **Comment:** Based on the features and target variable relationship above, we can notice that the number of reviews on airbnb listing is more popular for the house with few bedrooms. It is understandable that most of the guests rent smaller/fewer rooms. Besides, most of the reviews are also made for the cheaper priced rooms where the reviews can been seen very densed for price below than USD150. It is also noted that most of the reviews have high rating, with 5.0 rating dominates the reviews.

# #### Splitting Data

# In[15]:


# Split dataset into training set and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 70% training and 30% testing

print("Training set: Xtrain:{} ytrain:{}".format(X_train.shape, y_train.shape)) 
print("Test set: Xtest:{} ytest:{}".format(X_test.shape, y_test.shape)) 
print("---") 
print("Full dataset: X:{} y:{}".format(X.shape, y.shape))


# #### Building Decision Tree Model
# Since the target variable data is in continuous form, thus I am using DecisionTreeRegressor for my model.

# In[16]:


from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
airbnb_model = DecisionTreeRegressor(random_state = 42) 
airbnb_model.fit(X_train, y_train)


# In[17]:


DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
           max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           presort=False, random_state=42, splitter='best')


# ![mse.PNG](attachment:mse.PNG)

# **Comment:** Mean Square Error (MSE) is used as the criterion to minimise as for determining locations for future splits.

# In[18]:


# Visualizing decision tree for training set
import graphviz 
from sklearn.tree import export_graphviz

dot_data = tree.export_graphviz(airbnb_model, out_file=None, 
                      feature_names=df_features,  
                      class_names=df.price,  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph


# In[19]:


# Saving the graph in pdf format
graph.render("airbnb_tree1") 


# In[20]:


# The snippets of the prediction model for sample from training set
X_train.head()


# In[21]:


# The obtained prices from the predictions
airbnb_model.predict(X_train.head())


# #### Model Evaluation

# In[22]:


# Evaluating the model by using Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error

# Instructing the model to make predictions for the prices on the test set 
test_predictions = airbnb_model.predict(X_test)

# Calculating the MAE between the actual prices (in y_test) and the predictions made 
test_prediction_errors = mean_absolute_error(y_test, test_predictions)

test_prediction_errors


# ![mae.PNG](attachment:mae.PNG)

# **Comment:** **Mean Absolute Error (MAE)** is used to evaluate our model. Generally, MAE works in minimizing the L1 error using median values at terminal nodes. From the result above, we can say that our model gives an absolute error of approximately USD 54.86 per accomodation when we experiment on the testing data, out of a USD 113.15 mean value  at the initial data exploration. The high MAE might be resulted due to the small dataset or the model is too naive.

# In[23]:


# This function takes both the training and testing sets to compute the MAE for a Decision Tree 

def compute_mae(X_train, y_train, X_test, y_test, max_leaf_nodes): 
  trees_model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 42) 
  trees_model.fit(X_train, y_train) 
  validation_predictions = trees_model.predict(X_test) 
  error = mean_absolute_error(y_test, test_predictions)
  
  return(error)

def get_best_tree_size(X_train, y_train, X_test, y_test, verbose = False):
  # candidates to iterate on finding a better tree depth  
  candidate_max_leaf_nodes = [5, 10, 20, 30, 50, 100, 250, 500]
  
  # initialization 
  minimum_error = None 
  best_tree_size = 5 
  
  # loop to find the minimal error value 
  for max_leaf_nodes in candidate_max_leaf_nodes: 
    current_error = compute_mae(X_train, y_train, X_test, y_test, max_leaf_nodes) 
    verbose and print("(Size: {}, MAE: {})".format(max_leaf_nodes, current_error)) 
    
    if(minimum_error == None or current_error < minimum_error): 
      minimum_error = current_error 
      best_tree_size = max_leaf_nodes 
     
    return(best_tree_size) 
  
best_tree_size = get_best_tree_size(X_train, y_train, X_test, y_test, True) 
best_tree_size


# In[24]:


# Create the Decision Tree model
airbnb_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size, random_state = 42)
airbnb_model.fit(X_train, y_train)

# Generate the predictions for the testing set
test_predictions = airbnb_model.predict(X_test)
test_prediction_errors = mean_absolute_error(y_test, test_predictions)

test_prediction_errors


# **Comment:** After tuning up the maximum number of nodes hyper-parameter, we able to increase our prediction model on average of **~ USD 9.94** (54.863 - 44.925) by reducing the model's errors.

# In[27]:


# Since in previous steps we exclude categorical data, we are going to apply Label Encoding for the categorical data as this particular data may (or may not) improve the model

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df_features_extended = ['city', 'no_of_reviews', 'rating', 'no_of_guest', 'bedrooms_no']
X = df[df_features_extended]

X.sample(5)


# In[28]:


categorical = (X.dtypes == 'object')
categorial_columns = list(categorical[categorical].index)

categorial_columns


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create copies of our data sets to apply the transformations
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

encoder = LabelEncoder()

# for each column we fit the encoder and transform each of the rows
for column in categorial_columns:
    X_train_encoded[column] = encoder.fit_transform(X_train[column])
    X_test_encoded[column] = encoder.transform(X_test[column])

# A sample of our transformed data
X_train_encoded.sample(5)


# In[30]:


# Compute the best tree size
best_tree_size = get_best_tree_size(X_train_encoded, y_train, X_test_encoded, y_test)

# Create the model
airbnb_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size, random_state = 42)
airbnb_model.fit(X_train_encoded, y_train)

# Generate the predictions for the validation set
test_predictions = airbnb_model.predict(X_test_encoded)
test_prediction_errors = mean_absolute_error(y_test, test_predictions)

test_prediction_errors


# **Comment:** By applying label encoding to categorical data, we can boost our predictive model by reducing MAE to approximately **~ USD 29.27**. This result also proves that Label Encoding is the best fit for our categorical data and by inluding the categorical data, we can improve our model significantly as compared to the intial model (without categorical data).

# #### Visualizing Decision Tree

# In[31]:


import graphviz 
from sklearn.tree import export_graphviz

dot_data = tree.export_graphviz(airbnb_model, out_file=None, 
                      feature_names=df_features_extended,  
                      class_names=df.price,  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph


# In[32]:


# Saving the graph in pdf format
graph.render("airbnb_tree2") 

