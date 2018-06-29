
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from matplotlib.pylab import rcParams
#rcParams['figure.figsize'] = 12, 10
import sklearn
#import numpy as np
#from sklearn.datasets import fetch_mldata
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import train_test_split


# In[4]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


X= np.random.normal(1,1, size=600000).reshape(60000,10)
#X[:,5:10] = X[:,:5]
Y= np.sum(X, axis=1)
X_train, X_test = X[:50000,:5], X[50000:60000,0:5]
Y_train, Y_test = Y[:50000],Y[50000:60000]                                
np.var(Y)

model = LinearRegression(fit_intercept= True)
model.fit(X_train, Y_train)
#model.fit(X[:50000,:],Y_train)
print(model.coef_)
Y_predict = model.predict(X_test)
#Y_predict = model.predict(X[50000:60000,:])
MSE = mean_squared_error(Y_predict, Y_test)


# In[5]:


print(model.coef_)
print(model.intercept_)
print(np.mean(Y_predict))
print(MSE)


# In[11]:


import pandas as pd
A= pd.read_csv('/Users/swagatachakraborty/Desktop/mystery.dat', header = None)
B= pd.DataFrame(A)
A_x = B.iloc[:,:100]
A_y = B.iloc[:,100]

print(A.shape)


# In[13]:



from sklearn.linear_model import Lasso
def lasso_regression(alpha):
    #Fit the model
    lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    lassoreg.fit(A_x, A_y)
    y_pred = lassoreg.predict(A_x)
    
   #Return the result in pre-defined format
    rss = sum((y_pred-A_y)**2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret




# In[14]:


A.head(2)


# In[15]:


#Initialize predictors to all 15 powers of x
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]

col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,101)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]

coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

#Iterate over the 10 alpha values:
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(alpha_lasso[i])


# In[16]:


#coef_matrix_lasso

coef_matrix_lasso.apply(lambda x: sum(x.values==0),axis=1)


# In[17]:


#Initialize predictors to all 15 powers of x
alpha_lasso = np.linspace(0.01,1,10)

col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,101)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]

coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

#Iterate over the 10 alpha values:
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(alpha_lasso[i])


# In[18]:


coef_matrix_lasso.apply(lambda x: sum(x.values==0),axis=1)


# In[19]:


#Initialize predictors to all 15 powers of x
alpha_lasso = np.linspace(0.01,0.1,10)

col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,101)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]

coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

#Iterate over the 10 alpha values:
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(alpha_lasso[i])


# In[20]:


coef_matrix_lasso.apply(lambda x: sum(x.values==0),axis=1)
#coef_matrix_lasso.shape


# In[21]:


#coef_matrix_lasso.iloc[6,]
for i,j in enumerate(coef_matrix_lasso.iloc[5,2:]):
    if j!=0:
       print(i+1, j)


# In[22]:


coef_matrix_lasso.iloc[5,]


# In[23]:


#coef_matrix_lasso.iloc[6,]
for i in range(102):
    if coef_matrix_lasso.iloc[3,i]!=0:
       print(coef_matrix_lasso.iloc[4,i])


# In[24]:


#Initialize predictors to all 15 powers of x
alpha_lasso = np.linspace(0.03,0.04,10)

col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,101)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]

coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

#Iterate over the 10 alpha values:
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(alpha_lasso[i])


# In[25]:


coef_matrix_lasso.apply(lambda x: sum(x.values!=0),axis=1)


# In[27]:


#coef_matrix_lasso.iloc[6,]
for i in range(102):
    if coef_matrix_lasso.iloc[2,i]!=0:
       print(coef_matrix_lasso.iloc[4,i])


# In[51]:


coef_matrix_lasso.iloc[6, 90:]

