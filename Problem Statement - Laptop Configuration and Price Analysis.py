#!/usr/bin/env python
# coding: utf-8

# ## Context and Objective : 
# 
# We have the data for the laptop seller having different types of laptop with their prices and specification. Dataset contains total 1303 observations.  As  a  data scientist for a online publication  company  which focused on technology, our objective is  to find the insight  from the data provided  ,  and help user or subscriber of the company to get  informative visuals regarding laptop configuration and prices to improve customer experience also  predict the  laptop price with  given specification by the User.

# ## Data Dictionary :
# - Company: Laptop Manufacturer
# - Product: Brand and Model
# - TypeName: Type (Notebook, Ultrabook, Gaming, etc.)
# - Inches: Screen Size
# - ScreenResolution: Screen Resolution
# - Cpu: Central Processing Unit
# - Ram: Laptop RAM
# - Memory: Hard Disk / SSD Memory
# - GPU: Graphics Processing Unit
# - OpSys: Operating System
# - Weight: Laptop Weight
# - Price_euros: Price in euros

# In[1]:


#changing the default dir 
import os as os
dir="E:\GREAT_LEARNING\Supervised Learning_Regression\W2"
os.chdir(dir)
os.getcwd()


# In[2]:


#importing the basic library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_color_codes=True
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#import data for analysis
ldata=pd.read_csv("laptop_price.csv")
np.random.seed(1)
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",200)
ldata.sample(5)


# In[4]:


# checking the information about the data frame

ldata.info()


# In[5]:


#checking the missing vales for  the data frame
ldata.isna().sum()


# In[6]:


#checking the duplicate values for the col in data frame
ldata.duplicated().sum()





#description for the data frame

ldata.describe(include='all')


# In[8]:


#geneating the  unique value counts  for the categorical varibles with seperating cat and num columns
cat_col=[]
num_col=[]
for i in ldata.columns:
    cat_col.append(i)
    if ldata[i].dtypes=="object":
        print(f'unique value counts for colums {i}  :\n{ldata[i].value_counts()}')
        
    else :
        num_col.append(i)
        print(f'{i} :numerical colum')






# In[9]:


#formating some of the col  in the dataframe 
ldata['Product_model']=ldata['Product'].str.split(' ').str[0]
ldata['TypeName']=ldata['TypeName'].str.replace("2 in 1 Convertible","2IN1_Convert")
ldata['Screen_Pc1']=ldata['ScreenResolution'].str.split(" ").str[-1].str.split("x",expand=True)[0]
ldata['Screen_Pc2']=ldata['ScreenResolution'].str.split(" ").str[-1].str.split("x",expand=True)[1]
temp1=ldata['ScreenResolution'].str.split("1").str[0]
temp2=temp1.str.split("2").str[0]
temp3=temp2.str.split("3").str[0]
ldata['Screen_Display']=temp3.str.split("/").str[0].str.replace(" ","_")
ldata['Screen_Display'].value_counts()
ldata['Screen_Display'].replace("",'No Data',inplace=True)


# In[10]:


ldata['Cpu_Speed']=ldata['Cpu'].str.split(" ").str[-1].str.replace("GHz","").astype(float)
temp1=ldata['Cpu'].str.split("1").str[0]
temp2=temp1.str.split("2").str[0]
#rint(temp2.value_counts())
temp3=temp2.str.split("3").str[0]
#rint(temp3.value_counts())
temp4=temp3.str.split(" ").str[1]+"_"+temp3.str.split(" ").str[2]
temp4[temp4.isna()]
temp1=ldata['Cpu'].str.split("1").str[0]
temp2=temp1.str.split("2").str[0]
print(temp2.count())
temp3=temp2.str.split("3").str[0]
print(temp3.count())


ldata['Cpu_Org']=temp3.str.split(" ").str[0]
ldata['Cpu_Series']=temp3.str.split(" ").str[1]+"_"+temp3.str.split(" ").str[2]
ldata['Cpu_Series'].fillna('No_data',inplace=True)
ldata['Ram']=ldata['Ram'].str.split("GB").str[0].astype(int)
ldata['Added_Memory']=ldata['Memory'].str.split(" + ").str[-1].str.split("Flash Storage").str[-1]
ldata['Added_Memory'].replace('',np.nan,regex=True,inplace=True)
ldata['Memory_size']=ldata['Memory'].str.split(" ").str[0]


# In[11]:


#fundtion to convert the TB to GB 
def Storage_to_num(income_val):
 
   if isinstance(income_val, str):  # checks if `income_val` is a string
       multiplier = 1  # handles K vs M salaries
       if income_val.endswith('GB'):
           multiplier = 1
       elif income_val.endswith('TB'):
           multiplier = 1000
       return float(income_val.replace('GB', '').replace('TB', '')) * multiplier
   else:  # this happens when the current income is np.nan
       return np.nan

            


# In[12]:


ldata["Memory_size"].value_counts()


# In[13]:


ldata['Memory_size']=ldata['Memory_size'].apply(Storage_to_num)
ldata['Added_Memory']=ldata['Added_Memory'].str.split(" ").str[0]
ldata['Added_Memory']=ldata['Added_Memory'].apply(Storage_to_num)
#gpu mode and compay 
ldata['Gpu_model']=ldata['Gpu'].str.split(" ").str[1]+'_'+ldata['Gpu'].str.split(" ").str[2]


# In[14]:


ldata.head()


# In[15]:


#checking the weight coloumn for data frame
def wgt_convert(value):
    if isinstance(value,str):
        if value.endswith("kg"):
            return(float(value.replace("kg","")))
        else :
            return np.nan


# In[16]:


ldata["Weight"]=ldata["Weight"].apply(wgt_convert)
#dropping the unnesesory columns
ldata_bkp=ldata
ldata.drop(['laptop_ID', 'Product','ScreenResolution', 'Cpu',  'Memory', 'Gpu'],axis=1,inplace=True)
ldata['Screen_Pc1']=ldata['Screen_Pc1'].astype(int)
ldata['Screen_Pc2']=ldata['Screen_Pc2'].astype(int)
ldata['Added_Memory'].fillna(0,inplace=True,axis=0)
ldata['Gpu_model'].fillna('no data',inplace=True,axis=0)
ldata.drop_duplicates(inplace=True,keep='first')
ldata[['Ram','Added_Memory','Memory_size']]=ldata[['Ram','Added_Memory','Memory_size']].astype(int)


# # EDA

# In[17]:


#plotting the distibution  for the numerical datatypes
cat_col=[]
num_col=[]

for col in ldata.columns:
    if ldata[col].dtypes=="object":
        cat_col.append(col)
    else:
        num_col.append(col)


# In[18]:


for col in num_col:
    bins=50
    f2, (ax1, ax2) = plt.subplots(
        nrows=2, 
        sharex=True, 
        figsize=(10,5),
    )  
    sns.boxplot(
        data=ldata, x=col, ax=ax1, showmeans=True, color="yellow"
    ) 
    sns.histplot(
        data=ldata, x=col, kde=True, ax=ax2, bins=bins, palette="blue"
    ) 
    ax1.axvline(
        ldata[col].mean(), color="red", linestyle="--"
    )  
    ax2.axvline(
        ldata[col].median(), color="green", linestyle="-"
    )  
 
    


# In[19]:


for col in cat_col:
    n=20
    perc=True
    total = len(ldata[col])  # length of the column
    count = ldata[col].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=ldata,
        x=col,
        palette="Paired",
        order=ldata[col].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            ) 
        else:
            label = p.get_height() 

        x = p.get_x() + p.get_width() / 2  
        y = p.get_height()  

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot


# In[20]:



def st_plot(data, predictor, target):
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


# In[21]:


st_plot(ldata, "Company", "TypeName")


# In[22]:


st_plot(ldata, "Company", "Cpu_Org")


# In[23]:


st_plot(ldata, "Company", "OpSys")


# In[24]:


plt.figure(figsize=(15, 5))


sns.boxplot(data=ldata, y="Price_euros", x="Company")
plt.xticks(rotation=45)


plt.show()


# In[25]:


pi_data=ldata.groupby("Company")["Price_euros"].mean()
print(pi_data)
labels=["Acer"          ,
"Apple"       ,
"Asus"         ,
"Chuwi"         ,
"Dell"         ,
"Fujitsu"       ,
"Google"       ,
"HP"          ,
"Huawei"       ,
"LG"           ,
"Lenovo"       ,
"MSI"          ,
"Mediacom"     ,
"Microsoft"    ,
"Razer"        ,
"Samsung"      ,
"Toshiba"     ,
"Vero"          ,
"Xiaomi"       ]
fig = plt.figure(figsize =(10, 7))
plt.pie(pi_data, labels = labels)
 
# show plot
plt.show()


# In[26]:


sns.pairplot(ldata)


# In[27]:


corr=ldata.corr()
sns.heatmap(corr,annot=True)


# In[28]:




from sklearn.linear_model import LinearRegression,Ridge

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor
,BaggingRegressor
,AdaBoostRegressor)
from sklearn.model_selection import train_test_split, StratifiedKFold,cross_val_score
from sklearn import svm
# to check model performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,make_scorer

# to suppress warnings
import warnings

warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


# In[29]:



X=ldata.drop(['Price_euros'],axis=1)
y=ldata['Price_euros']


# In[30]:


from  sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline,make_pipeline
std=StandardScaler(with_mean=True)
onehot=OneHotEncoder()
X[['Inches','Ram','Weight','Screen_Pc1','Screen_Pc2','Cpu_Speed','Added_Memory','Memory_size']]=std.fit_transform(X[['Inches','Ram','Weight','Screen_Pc1','Screen_Pc2','Cpu_Speed','Added_Memory','Memory_size']])
enc_df = pd.DataFrame(onehot.fit_transform(X[['Company', 'TypeName', 'OpSys', 'Product_model', 'Screen_Display', 'Cpu_Org', 'Cpu_Series', 'Gpu_model']]).toarray())


# In[31]:


X.drop(['Company', 'TypeName', 'OpSys', 'Product_model', 'Screen_Display', 'Cpu_Org', 'Cpu_Series', 'Gpu_model'] ,axis=1,inplace=True)


# In[32]:


X=X.join(enc_df)


# In[33]:


y=np.log(y)


# In[34]:


# then we split the temporary set into train and validation
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)


print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)


# In[35]:


#building and testing multipe models
model_tr=[]
model_tr.append(("Linear_Regression",LinearRegression()))
model_tr.append(("ridge",Ridge()))
model_tr.append(("DecisionTreeRegressor",DecisionTreeRegressor(random_state=1)))
model_tr.append(("RandomForestRegressor",RandomForestRegressor(random_state=1)))
model_tr.append(("BaggingRegressor",BaggingRegressor(random_state=1)))
model_tr.append(("AdaBoostRegressor",AdaBoostRegressor(random_state=1)))
model_tr.append(("XGBRegressor",XGBRegressor(random_state=1)))


# In[36]:


# function to compute adjusted R-squared
def adj_r2_score(predictors, targets, predictions):
    r2 = r2_score(targets, predictions)
    n = predictors.shape[0]
    k = predictors.shape[1]
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))


# function to compute MAPE
def mape_score(targets, predictions):
    return np.mean(np.abs(targets - predictions) / targets) * 100


# In[95]:



df=[]
result=pd.DataFrame()
def run_defult_algo(XT,yT,Xv,yV):
    for name,ml  in model_tr:
        dt=ml
        dt.fit(XT,yT)
        predict=dt.predict(Xv)
        #temp=r2_score(y_val,predict)
        r2 = r2_score(yV, predict)  # to compute R-squared
        adjr2 = adj_r2_score(Xv,yV, predict) # to compute adjusted R-squared
        rmse = np.sqrt(mean_squared_error(yV, predict))  # to compute RMSE
        mae = mean_absolute_error(yV, predict)  # to compute MAE
        mape = mape_score(yV, predict)  # to compute MAPE
        # creating a dataframe of metrics
        print(f' Model_Name: {name}():\n rmse: {rmse}\n mae: {mae}\n R2: {r2}\n adjr2: {adjr2}\n mape: {mape}\n ' )
        print('*'*50)
              


# In[97]:


run_defult_algo(X_train,y_train,X_val,y_val)


# In[98]:


from sklearn.model_selection import validation_curve, learning_curve
from mlxtend.plotting import plot_learning_curves


# In[99]:


estimator=RandomForestRegressor()
                                            
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator,  X=X_train, y=y_train, cv=5,return_times=True)
 
# Calculating mean and standard deviation of training score
mean_train_score = np.mean(train_scores, axis = 1)
std_train_score = np.std(train_scores, axis = 1)

# Calculating mean and standard deviation of testing score
mean_test_score = np.mean(test_scores, axis = 1)
std_test_score = np.std(test_scores, axis = 1)
 
# Plot mean accuracy scores for training and testing scores
plt.plot( mean_train_score,
     label = "Training Score", color = 'b')
plt.plot( mean_test_score,
   label = "Cross Validation Score", color = 'g')
 
# Creating the plot
plt.title("Validation Curve with KNN Classifier")
plt.xlabel("Number of Neighbours")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.legend(loc = 'best')
plt.show()


# In[103]:


from sklearn.model_selection import GridSearchCV

param_grid_xgb={'n_estimators':[50,100,150],
            'learning_rate':[0.01,0.1,0.2,0.05],
            'gamma':[0,1,3,5],
            'max_depth':np.arange(1,5,1),
            'reg_lambda':[5,10]}
param_grid_reand={'n_estimators':[50,100,150,200],'max_depth':[1,2,3,4,5]}
scorer = "r2"
GridSearchCV = GridSearchCV(XGBRegressor(), param_grid=param_grid_xgb, scoring='r2', cv=10, n_jobs = -1)
fninal_modle=GridSearchCV.fit(X_train,y_train)
   
print("Best parameters are {} with CV score={}:" .format(GridSearchCV.best_params_,GridSearchCV.best_score_,))


# In[104]:


# define the pipeline
fmodel=XGBRegressor(gamma=0, learning_rate= 0.2, max_depth= 3, n_estimators= 150, reg_lambda= 10)

predict=fninal_modle.predict(X_test)
r2_score(y_test,predict)


# In[105]:


lt=ldata.drop('Price_euros',axis=1)


# In[106]:


temp=pd.DataFrame((sorted(zip(GridSearchCV.best_estimator_.feature_importances_,lt.columns),reverse=True)),columns=("value","col_name")).reindex()
sns.barplot(data=temp,y='col_name',x="value")


# In[ ]:




