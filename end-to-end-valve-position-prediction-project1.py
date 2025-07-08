#!/usr/bin/env python
# coding: utf-8

# # Predicting valve position on the basis of liquid level
# In this notebook, we will be trying to predict the position, the regulatory valve should assume on the basis of liquid level
# ## 1. Problem definition
# > How well can the valve position be predicted
# ## 2. Data
# > We have two sets of data, the dataset for valve position as well as the liquid level
# * The valve position dataset provides the valve position in terms of percentage (0-100)
# * The liquid level dataset provides liquid level in the tank in mililitres (ml)
# * We are to combine these two separate datasets in order to form our training, validation and test datasets
# ### 2.1. How to get the train, validation and test data sets
# * Firstly we will be combining the valve position and liquid level data from the `Процесс 2_Уровень_008.KIP1.L_S1_2_month.csv` and `Процесс_2_Положение_клпапна_008_KIP1_Pos_KlR7_2_1month.csv` data sets
# * Then we will be using the train_test_split to split the combined data in the ratio of 75:25 to get our training and validation datasets
# * As for the test data, we will use `Процесс_3_Уровень` data set as the test data set
# ## 3. Evaluation and Improvisation
# We will be using RMSLE (Root Mean Squared Log Error) as well as other evaluation methods. 
# The goal of the regression metrics will be to minimize the error.

# In[2]:


# Import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Read the Process_2_liquid_level 
liquid_level_data = pd.read_csv("Процесс 2_Уровень_008.KIP1.L_S1_2_month.csv", low_memory=False)
# Read the Process_2_Valve_Pos dataset
valve_pos_data = pd.read_csv("Процесс_2_Положение_клпапна_008_KIP1_Pos_KlR7_2_1month.csv", low_memory=False)


# In[4]:


liquid_level_data.info()


# In[5]:


len(liquid_level_data.DateTime)


# In[6]:


valve_pos_data.info()


# In[7]:


len(valve_pos_data.DateTime)


# In[8]:


df1 = liquid_level_data.copy()


# In[9]:


df1["DateTime"] = pd.to_datetime(df1["DateTime"], format="%d.%m.%Y %H:%M")


# In[10]:


df1.info()


# In[11]:


df1.DateTime[3]


# In[12]:


df2 = valve_pos_data.copy()
df2["DateTime"] = pd.to_datetime(df2["DateTime"], format = "%d.%m.%Y %H:%M")


# In[13]:


df2.info()


# In[14]:


df1.isna().sum()


# In[15]:


df2.isna().sum()


# In[16]:


# Check for string dtype
for label, content in df2.items():
    if pd.api.types.is_string_dtype(content):
        print("There are string dtypes in:", label)
    if pd.api.types.is_float_dtype(content):
        print("There are floats in:", label)
    
        


# In[17]:


df1.info()


# In[18]:


df1["Vals"] = pd.to_numeric(df1["Vals"],errors="coerce")
df2["Vals"] = pd.to_numeric(df2["Vals"],errors="coerce")


# In[19]:


#Fill up null values in the columns
for label, content in df1.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # make a label for if there was missing content or not
            df1[label+" is missing"] = pd.isnull(content)
            # Fill the missing values with 0
            df1[label] = content.fillna(0)
df1["Fracs"] = df1["Fracs"].astype('int64')
df1["Vals"] = df1["Vals"].astype('int64')            


# In[20]:


df1.info()


# In[21]:


df1.isna().sum()


# In[22]:


df1.head()


# In[23]:



for label, content in df2.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # make a label for if there was missing content or not
            df2[label+" is missing"] = pd.isnull(content)
            # Fill the missing values with 0
            df2[label] = content.fillna(0)
df2["Fracs"] = df2["Fracs"].astype('int64')
df2["Vals"] = df2["Vals"].astype('int64')


# In[24]:


df2.info()

df2.isna().sum()


# In[25]:


df2.head()


# In[26]:


#df1["Values"] = df1["Vals"]+"."+df1["Fracs"].astype(str)
#df2["Values"] = df2["Vals"]+"."+df2["Fracs"].astype(str)


# In[27]:


df1.head(), df2.head()


# In[28]:


df1.info(), df2.info()


# In[29]:


time_stamp_match = list(df1["DateTime"] == df2["DateTime"])


# In[30]:


time_stamp_match.count(True)


# In[ ]:





# In[31]:


df1.info()


# In[32]:


df1["Vals is missing"].value_counts()


# In[33]:


df2["Vals is missing"].value_counts()


# In[34]:


df1["Fracs is missing"].value_counts(), df2["Fracs is missing"].value_counts()


# In[35]:


df1.head()


# In[36]:


df2.head()


# In[37]:


df1["Liquid_levels"] = df1["Vals"].astype(str)+"."+df1["Fracs"].astype(str)


# In[38]:


df2["Valve_positions"] = df2["Vals"].astype(str)+"."+df2["Fracs"].astype(str)


# In[39]:


df1.head()


# In[40]:


df2.head()


# In[41]:


df1["Liquid_levels"] = pd.to_numeric(df1["Liquid_levels"], downcast="float")


# In[42]:


df2["Valve_positions"] = pd.to_numeric(df2["Valve_positions"], downcast="float")


# In[43]:


df1.info(), df2.info()


# In[44]:


df1["Liquid_levels"].isnull().sum()


# In[45]:


df2["Valve_positions"].isnull().sum()


# In[46]:


df1["Liquid_levels"][df1["Liquid_levels"]>1000] = df1["Liquid_levels"].median()


# In[47]:


df2["Valve_positions"][df2["Valve_positions"]>100]


# In[48]:


df2["Valve_positions"][df2["Valve_positions"]>100] = df2["Valve_positions"].median()


# In[49]:


df2["Valve_positions"][df2["Valve_positions"]>100]


# In[50]:


df1["Liquid_levels"].max(), df2["Valve_positions"].max()


# In[51]:


np.argmax(df1["Liquid_levels"]), np.argmax(df2.Valve_positions)


# In[52]:


df1["Valve_positions"] = df2["Valve_positions"]


# In[53]:


df1.info()


# In[54]:


df1["Year"] = df1.DateTime.dt.year
df1["Month"] = df1.DateTime.dt.month
df1["Day"] = df1.DateTime.dt.day
df1["Hour"] = df1.DateTime.dt.hour
df1["Minute"] = df1.DateTime.dt.minute
df1["DayOfWeek"] = df1.DateTime.dt.dayofweek
df1["DayOfYear"] = df1.DateTime.dt.dayofyear


# In[55]:


df1.drop(columns = ["DateTime", "Vals", "Fracs"], inplace = True)


# In[56]:


df1.drop(columns = ["Vals is missing", "Fracs is missing"], inplace = True)


# In[57]:


df1.info()


# ## Modelling
# 
# Preprocessing the data has been completed. Now its time to start the modelling process. 
# In order to model the data, we will use sklearn's `train_test_split()` method to split the data between training and validation sets.

# In[59]:


from sklearn.model_selection import train_test_split


# Now split the dataset between training and validation sets. In this step we will use the liquid level data to predict valve positions
# The splitting will be done in a 75:25 ratio. I.e. 75% of the data will be used to train the model and the rest will be used to validate

# In[61]:


# First split the dataframe between X and y
X = df1.drop("Valve_positions", axis = 1) 
y = df1["Valve_positions"]


# In[62]:


# After creating X and y, its time to split them into training and validation datasets
X_train, X_val, y_train, y_val = train_test_split(X,y,
                                                  test_size=0.25)


# We will use the `RandomForestRegressor` to create our model with `n_jobs` = -1 and `random_state` = 54

# In[64]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_jobs = -1,
                             random_state=54)


# In[65]:


get_ipython().run_cell_magic('time', '', '# Try to fit the model\nmodel.fit(X_train, y_train)')


# In[66]:


model.score(X_train, y_train)


# In[67]:


model.score(X_val, y_val)


# ### Improvisation
# Now that we have gotten a score of about **10%** on the validation dataset, it is obvious that we need to improve the model, in order to do that, lets optimize its parameters

# Improvisation by optimizing the hyperparameters using `RandomizedSearchCV`

# In[70]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import RandomizedSearchCV\nrf_grid = {"n_estimators":np.arange(10, 1000, 10),\n          "max_depth":[None, 3, 5, 10],\n          "min_samples_split": np.arange(2, 40, 2),\n          "min_samples_leaf":np.arange(1, 20, 2),\n          "max_features":[0.5, 1, "sqrt", "auto"],\n          "max_samples": [10000]}\n\n# Instantiate model using these hyperparameters\nrs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,\n                                                   random_state=60),\n                             param_distributions=rf_grid,\n                             n_iter=6,\n                             cv=5,\n                             verbose=True)\n\n\n# Fit the model\nrs_model.fit(X_train, y_train)')


# In[71]:


rs_model.best_params_


# In[72]:


rs_model.score(X_val, y_val)


# In[73]:


# Let's do a function for scoring 
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score
# Create a function to calculate root mean squared log error 
def rmsle(y_true, y_preds):
    """
    This function takes true labels (y_true) and prediction labels (y_preds) as input and computes the root mean squared log error
    """
    return np.sqrt(mean_squared_log_error(y_true, y_preds))
# Another function to show the various scores using the true labels and prediction labels
def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    scores = {"Training R^2 Score":r2_score(y_train, train_preds),
             "Validation R^2 Score": r2_score(y_val, val_preds),
              "Training Mean Absolute Error": mean_absolute_error(y_train, train_preds),
              "Validation Mean Absolute Error": mean_absolute_error(y_val, val_preds), 
              "Traininig RMSLE": rmsle(y_train, train_preds),
              "Validation RMSLE": rmsle(y_val, val_preds)
             }
    return scores


# In[74]:


show_scores(model)


# In[75]:


show_scores(rs_model)


# In[ ]:




