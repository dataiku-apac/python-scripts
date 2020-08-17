
# coding: utf-8

# # Compute similarity between Phone to detect similar behavior
# 
# The goal of this approach is to find phone that might be similar in terms of minutes on the phone during the day and the evening.
# Our intuition is as follow: if two phones are very similar, it is likely that they will behave the same way. 
# As a matter of fact, a phone with a low data usage day and night has a higher probability of churn. 
# 
# We therefore need to compute the similarity score between all the phones. 
# Later we will do a KNN to identify the 5 closest neighbors and if they have an history of churn and we will then flag them as potentially risky.
# 
# Later on, we could also the program they belong too as an extra feature.

# In[2]:


import dataiku as dk 
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns


# ### Getting access to data 
# We are using the dataiku api to directly read the data in database. No need for SQL statement. Sweet :) 

# In[3]:


# Load a DSS dataset as a Pandas dataframe
df = dk.Dataset("customers_prepared").get_dataframe(limit=1000)
df.head()


# ### L2 Normalization of the two axis
#  
# To make sure we can compare things that are comparable, we will use L2 normalizaition on Day_Mins and Eve_Mins. Other technics
# might actually yiel better results. We should compare against standardization for instance. 
# 

# In[4]:


# L2 Normalization of Fico Score and age_card_days
normDayMin = np.sum([np.power(day,2) for day in df.Day_Mins.values])
normEveMin = np.sum([np.power(eve,2) for eve in df.Eve_Mins.values])

df['l2_norm_Day_Mins'] = [day * 1e5 /normDayMin for day in  df.Day_Mins.values] 
df['l2_norm_Eve_Mins'] = [eve * 1e5 /normEveMin for eve in df.Eve_Mins.values] 


# In[19]:


df.head()


# ### Plotting - just for fun :) 
# 
# We can't see much in the graph. However it seems that the biggest consumer phone day and night. This will have to be confirmed with business knowledge from the markeing team. 

# In[5]:


# Plotting just for fun 
df.plot(kind='scatter',x='l2_norm_Day_Mins',y='l2_norm_Eve_Mins')


# ### Cosine Similarity computation 
# 
# Classic computation of the cosine similarity between two phone defined as (Day_mins,Eve_Mins). 
# Doing it through python is super slow by the way. We might ask someone from the data engineering team to compute that
# in SQL directly. The math are not complicated and less prone to errors.

# In[22]:


# Similarity - by the way this is super slow - we might have better results implementing that directly in SQL
sim = []
for i in range(df.shape[0]):
    for j in range(i,df.shape[0]):
        dictSim = {'phone1':df.Phone.values[i],'phone2':df.Phone.values[j]}
        vectorPhone1 = [df.l2_norm_Day_Mins[i],df.l2_norm_Eve_Mins[i]]
        vectorPhone2 = [df.l2_norm_Day_Mins[j],df.l2_norm_Eve_Mins[j]]
        dictSim['sim'] = np.dot(vectorPhone1,vectorPhone2)/ ( np.linalg.norm(vectorPhone1) * np.linalg.norm(vectorPhone2) )
        
        sim.append(dictSim)


# ### End results 
# 
# The math are right - the similarity for the same phon is 1. 
# This is therefore ready to be implemented. 
# This dataset should be used as follow: 
#     for a given phone, identify the K most similar phone. Count the number of churn  already recorded 
#     for the group of similar phone. This give a white label score. The closest to 0 the best the score.

# In[23]:


# End results
df_sim = pd.DataFrame(sim,columns=['phone1','phone2','sim'])
df_sim.head()

