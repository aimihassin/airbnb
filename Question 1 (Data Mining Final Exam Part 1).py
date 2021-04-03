#!/usr/bin/env python
# coding: utf-8

# # Data Mining Final Exam Part 1
# 
#     Name: Aimi Nabilah Hassin
#     Matric No: WQD180105/17198801

# ## Question 1

# In[1]:


# Import and load all the necessary libraries

import pandas as pd
import numpy as np
import requests
import re
from bs4 import BeautifulSoup


# In[2]:


# Retrieve the data from url

url = "https://github.com/priyasrivast/WebscrapingAirBnbAndHotels/blob/master/dataFiles_cleaner_plotters/airBnb_Mn_Or.csv"

try:
    page = requests.get(url, timeout=5)
    if page.status_code == 200:
        soup = BeautifulSoup(page.content,'html.parser')
        table = soup.find("table", {"class": "js-csv-data csv-data js-file-line-container"})
        data = pd.read_html(str(table))
    else: 
        print(str(page.status_code) + " - Error, page not found.")
except requests.ConnectionError as e:
    print("Connection error")
    print(str(e))
    
df = data[0]
df = df.iloc[:, 1:9]
df.head()


# In[3]:


#Checking the datatypes
df.dtypes


# In[4]:


#Changing the datatypes
df.price = df.price.astype(str)
df.no_of_reviews = df.no_of_reviews.astype(str)


# In[5]:


def remove_special_chars(val):
    new_val = re.findall('[0-9]',val)
    return "".join(new_val)

df.price = df.price.apply(remove_special_chars)
print(df.price.head(5))


# In[6]:


df.no_of_reviews = df.no_of_reviews.apply(remove_special_chars)
print(df.no_of_reviews.head())


# In[7]:


def split_to_list(str_val):
    return str_val.split(",")
    
df['occupancy_lst'] = df.occupancy.apply(split_to_list)
df.occupancy_lst.sample(5)


# In[8]:


len(df)
df.occupancy_lst[0][0]


# In[9]:


df = df.assign(no_of_guest=df.price.mean(), house_size=df.price.mean(), no_of_beds = df.price.mean(), no_of_baths=df.price.mean())


# In[10]:


for i in range(len(df)):
    try:

        df.no_of_guest[i] = df.occupancy_lst[i][0]
        df.house_size[i]  = df.occupancy_lst[i][1]
        df.no_of_beds[i]  = df.occupancy_lst[i][2]
        df.no_of_baths[i]  = df.occupancy_lst[i][3]
        
    except:
        continue


# In[13]:


print(df.dtypes)
df.head()


# In[14]:


df.no_of_guest = df.no_of_guest.apply(remove_special_chars)
print(df.no_of_guest.head())


# In[15]:


print(sum(df.house_size.isna()))

print(sum(df.house_size.isnull()))


# In[16]:


def convert_chars(val):
   ##note that studio is tagged as 0.5 bedroom
    if val == " 'Studio'":
        new_val = '0.5'
    else:
        new_val = re.findall('[0-9]',val)
    
    return "".join(new_val)

df.house_size = df.house_size.astype(str)
df = df.assign(bedrooms_no=df.house_size.apply(convert_chars))

df.sample(5)


# In[17]:


def remove_chars(val):
    new_val = re.findall('[0-9.]+',val)
    return "".join(new_val)

df.no_of_beds = df.no_of_beds.astype(str)
df.no_of_baths = df.no_of_baths.astype(str)

df = df.assign(beds_no=df.no_of_beds.apply(remove_chars))
df = df.assign(bath_no=df.no_of_baths.apply(remove_chars))
df.sample(5)


# In[18]:


df.dtypes


# In[19]:


(df == "none").any() #check for any none values


# In[20]:


(df.isnull()).any()


# In[21]:


(df.isna()).any()


# In[22]:


print(len(df))
df.isna().sum().sum() #to check num of Nans


# In[23]:


df.dropna() #check


# In[24]:


dfClean = df.dropna() #drop all NaN


# In[25]:


print(len(dfClean))
dfClean.isna().sum().sum() #check num of Nans


# In[26]:


#Checking for any empty price and remove it
dfClean.loc[dfClean.price == ''] 


# In[27]:


print(dfClean.shape) # check  shape before dropping 2 rows
dfClean = dfClean.drop([139,494], axis=0)
dfClean.shape # check  shape after dropping 2 rows


# In[28]:


#convert price to float
dfClean.price=dfClean.price.astype(float)
#convert no_of_reviews to float
dfClean.no_of_reviews=dfClean.no_of_reviews.astype(int)
#convert no_of_guest to float
dfClean.no_of_guest=dfClean.no_of_guest.astype(float)
#convert house to float
dfClean.bedrooms_no=dfClean.bedrooms_no.astype(float)
#convert house to float
dfClean.beds_no=dfClean.beds_no.astype(float)
#convert house to float
dfClean.bath_no=dfClean.bath_no.astype(float)

dfClean.dtypes #checking the datatypes


# In[29]:


# Drop un-important columns
dfClean = dfClean.drop(['occupancy_lst','house_size', 'no_of_beds', 'no_of_baths'], axis=1)
dfClean.sample(5)


# 
# **In Question 2, I am required to draw a snowflake schema for this dataset. Hence, I shall split this dataset into a few dataframe tables, so that I able to produce a linkage between these tables.**
# 

# In[30]:


## Adding a 'listing_id' column to the dataframe, which is going to be the primary key for this data

dfClean.insert(0, 'listing_id', range(1000, 1000 + len(dfClean)))
dfClean


## Adding a 'host_id' column

dfClean.insert(1, 'host_id', range(99900, 99900 + len(dfClean)))
dfClean


# In[31]:


# Saving the data into csv
dfClean.to_csv('airbnb.csv')


# In[32]:


## First - create listing_table, which is a fact table (primary one)

listing_table = dfClean[['listing_id', 'host_id', 'price']]
listing_table.sample(5)


# In[33]:


## Second - creating host_table 

host_table = dfClean[['host_id', 'owner_name']]
host_table.rename(columns = {'owner_name':'host_name'}, inplace = True) #renaming the owner_name column
host_table.sample(5)


# In[34]:


## Third - create location table

location_table = dfClean[['listing_id', 'city']]
location_table.sample(5)


# In[35]:


## Fourth - create house table

house_table = dfClean[['listing_id', 'house_title', 'house_type', 'occupancy']]
house_table.sample(5)


# In[38]:


## Fifth - create room table

room_table = dfClean[['house_title', 'no_of_guest', 'bedrooms_no', 'beds_no', 'bath_no']]
room_table.sample(5)


# In[37]:


## Sixth - create review table

review_table = dfClean[['listing_id', 'no_of_reviews', 'rating']]
review_table.sample(5)

