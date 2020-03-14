
# coding: utf-8

# # Overview
# Author:yum
# 
# Running has become the new hot topic. Since marathons are the ultimate challenge for runners, we want to create a model that can help train runners, and help them predict their running result using real-time/constantly updating data
# 
# In this part, We will use the neuron network to train our data.
# 
# Our dataset：https://www.kaggle.com/rojour/boston-results

# # Data pretreatment
# First, we should do some pretreatment to our dataset. I will use the prewritten function data_pre to extract the useful data from the dataset and shuffle them.

# In[1]:


import pandas as pd
def time_to_min(str):
    if str is not '-':
        time_segments=str.split(':')
        return float(time_segments[0])*60+float(time_segments[1])+float(time_segments[2])/60;
    else:
        return -1;
def gender_to_numeric(gender):
    if gender=='M':
        return 1;
    else:
        return 0;
def data_pre(file,file_1,file_2):
    path='./'
    df=pd.read_csv(path+file)
    df_1=pd.read_csv(path+file_1)
    df_2=pd.read_csv(path+file_2)
    df=pd.concat([df,df_1,df_2],axis=0,sort=False)
    df=df.drop(df.columns[5:10],axis=1)
    df=df.drop(df.columns[0:3],axis=1)
    df=df.drop(columns='Proj Time')
    df=df.drop(columns=['Overall','Gender','Division'],axis=1)
    df=df[~(df['5K']=='-')&~(df['10K']=='-')&~(df['15K']=='-')&~(df['20K']=='-')&~(df['Half']=='-')&~(df['25K']=='-')&~(df['30K']=='-')&~(df['35K']=='-')&~(df['40K']=='-')&~(df['Official Time']=='-')]
    df['M/F']=df['M/F'].apply(lambda x:gender_to_numeric(x))
    df['5K']=df['5K'].apply(lambda x:time_to_min(x))
    df['10K']=df['10K'].apply(lambda x:time_to_min(x))
    df['15K']=df['15K'].apply(lambda x:time_to_min(x))
    df['20K']=df['20K'].apply(lambda x:time_to_min(x))
    df['Half']=df['Half'].apply(lambda x:time_to_min(x))
    df['25K']=df['25K'].apply(lambda x:time_to_min(x))
    df['30K']=df['30K'].apply(lambda x:time_to_min(x))
    df['35K']=df['35K'].apply(lambda x:time_to_min(x))
    df['40K']=df['40K'].apply(lambda x:time_to_min(x))
    df['Pace']=df['Pace'].apply(lambda x:time_to_min(x))
    df['Official Time']=df['Official Time'].apply(lambda x:time_to_min(x))
    df=df[['Age','M/F','5K','10K','15K','20K','Half','25K','30K','35K','40K','Official Time','Pace']]
    df=df.sample(frac=1).reset_index(drop=True)
    return df


# In[2]:


df=data_pre('marathon_results_2015.csv','marathon_results_2016.csv','marathon_results_2017.csv')
df.head()


# # Data standardization
# Before we throw our data into the DNN, we need to standardize our data.

# In[3]:


data_all=df[['Age','M/F','5K','10K','15K','20K','Half','25K','30K','35K','40K']]
mean=data_all.mean(axis=0)
data_all-=mean
std=data_all.std(axis=0)
data_all/=std
data_all.head()


# # Choose feasible model parameter
# In this part, we want to use
# 
# 'Age','M/F','5K' to predict 'Official time'
# 
# and find some feasible model parameter for lateruse
# 
# That is:
# layer type
# optimizer
# loss
# metrics
# activation function
# regularizer
# batch_size
# shuffle
# 
# Some other parameter may change due to our different prediction such that:
# number of layers
# node of each layers
# epochs

# In[4]:


data=data_all[['Age','M/F','5K']]
target=df['Official Time']


# # Seperate data
# we need separate our input data to trainning set, validation set, test set.
# The reason why we use different validation set and test set is that we want to avoid the problem of information leak.

# In[5]:


data.shape


# In[6]:


data_train=data[15000:70000]
data_val=data[70000:]
data_test=data[0:15000]

target_train=target[15000:70000]
target_val=target[70000:]
target_test=target[0:15000]

import keras.backend as K
def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f


# # Construct Model
# This is one model with activation function "Relu" and without regulization.

# We add R2 to our evaluation method

# In[7]:


import keras.backend as K
import matplotlib.pyplot as plt
def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f


# In[8]:


from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
model = models.Sequential()
model.add(layers.Dense(64,input_shape=(3,), activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=200, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history
history_dict.keys()


# Plot Loss Chart

# In[9]:


val_r2=history_dict['val_r2']
r2=history_dict['r2']
mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
n,m=25,200
plt.plot(epochs[n:m],val_loss[n:m],'r',label='validation Loss')
plt.plot(epochs[n:m],loss[n:m],'b',label='Training Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[10]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# # Add regularization to our model

# In[11]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
model = models.Sequential()
model.add(layers.Dense(64,input_shape=(3,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=200, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history


# Plot loss chart

# In[12]:


import matplotlib.pyplot as plt
val_r2=history_dict['val_r2']
r2=history_dict['r2']
mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
n,m=25,200             
plt.plot(epochs[n:m],val_loss[n:m],'r',label='validation Loss')
plt.plot(epochs[n:m],loss[n:m],'b',label='Training Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[13]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# After adding L2 regularization, our val_mae nearly doesn't change 
# 
# we think this may be caused by we have a very big dataset and just a few parameters, so the problem of overfiting is trivial.
# 
# But, we still decided to add L2 regularization to our final model to mitigate this trivial problem.
# 
# And we can see our model get best performance after around 150 epochs,so we we cut epochs to 150

# # Adjust Epoch

# In[14]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
model = models.Sequential()
model.add(layers.Dense(64,input_shape=(3,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=150, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history


# In[15]:


import matplotlib.pyplot as pltx
val_r2=history_dict['val_r2']
r2=history_dict['r2']
mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
pltx.plot(epochs[30:150],val_loss[30:150],'r',label='validation Loss')
pltx.plot(epochs[30:150],loss[30:150],'b',label='Training Loss')
pltx.title('Training and validation loss')
pltx.xlabel('Epochs')
pltx.ylabel('Loss')
pltx.legend()
pltx.show()


# In[16]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# # Compare Activation function between Relu and Swish
# we use swish as our activation function to compare the permormance with relu.

# In[17]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
from keras import backend as K     
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects

def swish(inputs):
    return (K.sigmoid(inputs) * inputs)
get_custom_objects().update({'swish': Activation(swish)})

model = models.Sequential()
model.add(layers.Dense(64,input_shape=(3,), activation='swish',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(64, activation='swish',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(64, activation='swish',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=100, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history


# In[18]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# The test mae doesn't make a clear difference. So we decided to use the simple Relu activation function instead of the fancy one "swish"

# # The Real Deal
# we get a conclusion of our parameter and we deicded to use them in our all prediction.
# 
# layer type:              Dense
# 
# optimizer:               RMSprop
# 
# loss:                    mse
# 
# metrics:                 mae,r2
# 
# activation function:     relu
# 
# regularizer:             regularizers.l2(0.01)
# 
# batch_size:              5000
# 
# shuffle:                 true

# ## ......'5K' Predict  '10K'

# In[19]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
data=data_all[['Age','M/F','5K']]
target=df['10K']
data_train=data[15000:70000]
data_val=data[70000:]
data_test=data[0:15000]
target_train=target[15000:70000]
target_val=target[70000:]
target_test=target[0:15000]
model = models.Sequential()
model.add(layers.Dense(6,input_shape=(3,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=700, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history


# We plot the line chart of Training and validation loss

# In[20]:


import matplotlib.pyplot as plt
val_r2=history_dict['val_r2']
r2=history_dict['r2']
mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
plt.plot(epochs,val_loss,'r',label='validation Loss')
plt.plot(epochs,loss,'b',label='Training Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# We plot the line chart of Training and validation mae

# In[21]:


plt.plot(epochs[300:],val_mean_absolute_error[300:],'r',label='validation mean_absolute_error')
plt.plot(epochs[300:],mean_absolute_error[300:],'b',label='Training mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.show()


# We plot the line chart of Training and validation r2

# In[22]:


plt.plot(epochs[300:],val_r2[300:],'r',label='validation r2')
plt.plot(epochs[300:],r2[300:],'b',label='Training r2')
plt.title('Training and validation r2')
plt.xlabel('Epochs')
plt.ylabel('mr2')
plt.legend()
plt.show()


# In[23]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# In[24]:


model.predict(data_test)


# ### add MAE and R2 to a list then plot a evaluation chart in the end.

# In[25]:


list_next_mae=[]
list_next_r2=[]
list_ot_mae=[]
list_ot_r2=[]
list_next_distance=[]
list_ot_distance=[]
list_next_mae.append(results[1])
list_next_r2.append(results[2])
list_next_distance.append(10)


# ### An example to save and use the trained model

# Save the model

# In[26]:


model.save('model5k_10k.h5')


# Evaluate the model

# In[27]:


from keras.models import load_model
def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
dependencies = {
     'r2': r2
}
model=load_model('model5k_10k.h5',custom_objects=dependencies)
results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# Predict

# In[28]:


data_test.head()


# In[29]:


model.predict(data_test)


# If we use new data, we need  standardize the original data to (-1,1) first.
# 
# data=(data-all_mean)/all_std(See the 'Standardization' part)
# 
# Here we just input standardized data.

# In[30]:


import numpy as np
p=np.zeros((1,3))
p[0][0]=1.197
p[0][1]=0.912
p[0][2]=-0.156
model.predict(p)


# ## ......'5K' Predict  'Official Time'

# In[31]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
data=data_all[['Age','M/F','5K']]
target=df['Official Time']
data_train=data[15000:70000]
data_val=data[70000:]
data_test=data[0:15000]
target_train=target[15000:70000]
target_val=target[70000:]
target_test=target[0:15000]
model = models.Sequential()
model.add(layers.Dense(64,input_shape=(3,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=150, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history
model.save('model5k_OT.h5')


# In[32]:


val_r2=history_dict['val_r2']
r2=history_dict['r2']
mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
plt.plot(epochs[20:100],val_mean_absolute_error[20:100],'r',label='validation mean_absolute_error')
plt.plot(epochs[20:100],mean_absolute_error[20:100],'b',label='Training mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.show()


# In[33]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# In[34]:


list_ot_mae.append(results[1])
list_ot_r2.append(results[2])
list_ot_distance.append(5)


# ## ......'10K' predict '15K'

# In[35]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
data=data_all[['Age','M/F','5K','10K']]
target=df['15K']
data_train=data[15000:70000]
data_val=data[70000:]
data_test=data[0:15000]
target_train=target[15000:70000]
target_val=target[70000:]
target_test=target[0:15000]
model = models.Sequential()
model.add(layers.Dense(8,input_shape=(4,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=800, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history
model.save('model10k_15k.h5')


# We plot the chart of mean absolute error

# In[36]:


mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
n,m=480,800
plt.plot(epochs[n:m],val_mean_absolute_error[n:m],'r',label='validation mean_absolute_error')
plt.plot(epochs[n:m],mean_absolute_error[n:m],'b',label='Training mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.show()


# In[37]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# In[38]:


list_next_mae.append(results[1])
list_next_r2.append(results[2])
list_next_distance.append(15)


# ## ......'10K' predict 'Official Time'

# In[39]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
data=data_all[['Age','M/F','5K','10K']]
target=df['Official Time']
data_test=data[0:15000]
data_train=data[15000:70000]
data_val=data[70000:]
target_train=target[15000:70000]
target_val=target[70000:]
target_test=target[0:15000]
model = models.Sequential()
model.add(layers.Dense(32,input_shape=(4,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=300, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history
model.save('model10k_OT.h5')


# We plot the chart of mean absolute error

# In[40]:


mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
n,m=100,300
plt.plot(epochs[n:m],val_mean_absolute_error[n:m],'r',label='validation mean_absolute_error')
plt.plot(epochs[n:m],mean_absolute_error[n:m],'b',label='Training mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.show()


# In[41]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# In[42]:


list_ot_mae.append(results[1])
list_ot_r2.append(results[2])
list_ot_distance.append(10)


# ## ......'15K' predict '20K'

# In[43]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
data=data_all[['Age','M/F','5K','10K','15K']]
target=df['20K']
data_train=data[15000:70000]
data_val=data[70000:]
data_test=data[0:15000]
target_train=target[15000:70000]
target_val=target[70000:]
target_test=target[0:15000]
model = models.Sequential()
model.add(layers.Dense(10,input_shape=(5,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=1000, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history
model.save('model15k_20k.h5')


# We plot the chart of mean absolute error

# In[44]:


mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
n,m=500,1000
plt.plot(epochs[n:m],val_mean_absolute_error[n:m],'r',label='validation mean_absolute_error')
plt.plot(epochs[n:m],mean_absolute_error[n:m],'b',label='Training mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.show()


# In[45]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# In[46]:


list_next_mae.append(results[1])
list_next_r2.append(results[2])
list_next_distance.append(20)


# ## ......'15K' predict 'Official Time'

# In[47]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
data=data_all[['Age','M/F','5K','10K','15K']]
target=df['Official Time']
data_train=data[15000:70000]
data_val=data[70000:]
data_test=data[0:15000]
target_train=target[15000:70000]
target_val=target[70000:]
target_test=target[0:15000]
model = models.Sequential()
model.add(layers.Dense(16,input_shape=(5,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(16, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=600, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history
model.save('model15k_OT.h5')


# We plot the chart of mean absolute error

# In[48]:


mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
n,m=200,600
plt.plot(epochs[n:m],val_mean_absolute_error[n:m],'r',label='validation mean_absolute_error')
plt.plot(epochs[n:m],mean_absolute_error[n:m],'b',label='Training mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.show()


# In[49]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# In[50]:


list_ot_mae.append(results[1])
list_ot_r2.append(results[2])
list_ot_distance.append(15)


# ## ......'20K' predict 'Half'

# In[51]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
data=data_all[['Age','M/F','5K','10K','15K','20K']]
target=df['Half']
data_train=data[15000:70000]
data_val=data[70000:]
data_test=data[0:15000]
target_train=target[15000:70000]
target_val=target[70000:]
target_test=target[0:15000]
model = models.Sequential()
model.add(layers.Dense(12,input_shape=(6,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=1000, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history
model.save('model20k_Half.h5')


# We plot the chart of mean absolute error

# In[52]:


mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
n,m=500,1000
plt.plot(epochs[n:m],val_mean_absolute_error[n:m],'r',label='validation mean_absolute_error')
plt.plot(epochs[n:m],mean_absolute_error[n:m],'b',label='Training mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.show()


# In[53]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# In[54]:


list_next_mae.append(results[1])
list_next_r2.append(results[2])
list_next_distance.append(21)


# ## ......'20K' predict 'Official Time'

# In[55]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
data=data_all[['Age','M/F','5K','10K','15K','20K']]
target=df['Official Time']
data_train=data[15000:70000]
data_val=data[70000:]
data_test=data[0:15000]
target_train=target[15000:70000]
target_val=target[70000:]
target_test=target[0:15000]
model = models.Sequential()
model.add(layers.Dense(12,input_shape=(6,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=1200, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history
model.save('model20k_OT.h5')


# We plot the chart of mean absolute error

# In[56]:


mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
n,m=600,1200
plt.plot(epochs[n:m],val_mean_absolute_error[n:m],'r',label='validation mean_absolute_error')
plt.plot(epochs[n:m],mean_absolute_error[n:m],'b',label='Training mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.show()


# In[57]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# In[58]:


list_ot_mae.append(results[1])
list_ot_r2.append(results[2])
list_ot_distance.append(20)


# ## ......'Half' predict '25K'

# In[59]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
data=data_all[['Age','M/F','5K','10K','15K','20K','Half']]
target=df['25K']
data_train=data[15000:70000]
data_val=data[70000:]
data_test=data[0:15000]
target_train=target[15000:70000]
target_val=target[70000:]
target_test=target[0:15000]
model = models.Sequential()
model.add(layers.Dense(14,input_shape=(7,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=1000, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history
model.save('modelHalf_25K.h5')


# We plot the chart of mean absolute error

# In[60]:


mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
n,m=400,1000
plt.plot(epochs[n:m],val_mean_absolute_error[n:m],'r',label='validation mean_absolute_error')
plt.plot(epochs[n:m],mean_absolute_error[n:m],'b',label='Training mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.show()


# In[61]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# In[62]:


list_next_mae.append(results[1])
list_next_r2.append(results[2])
list_next_distance.append(25)


# ## ......'Half' predict 'Official Time'

# In[63]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
data=data_all[['Age','M/F','5K','10K','15K','20K','Half']]
target=df['Official Time']
data_train=data[15000:70000]
data_val=data[70000:]
data_test=data[0:15000]
target_train=target[15000:70000]
target_val=target[70000:]
target_test=target[0:15000]
model = models.Sequential()
model.add(layers.Dense(16,input_shape=(7,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(16, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=2000, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history
model.save('modelHalf_OT.h5')


# We plot the chart of mean absolute error

# In[64]:


mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
n,m=500,2000
plt.plot(epochs[n:m],val_mean_absolute_error[n:m],'r',label='validation mean_absolute_error')
plt.plot(epochs[n:m],mean_absolute_error[n:m],'b',label='Training mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.show()


# In[65]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# In[66]:


list_ot_mae.append(results[1])
list_ot_r2.append(results[2])
list_ot_distance.append(21)


# ## ......'25K' predict '30K'

# In[67]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
data=data_all[['Age','M/F','5K','10K','15K','20K','Half','25K']]
target=df['30K']
data_train=data[15000:70000]
data_val=data[70000:]
data_test=data[0:15000]
target_train=target[15000:70000]
target_val=target[70000:]
target_test=target[0:15000]
model = models.Sequential()
model.add(layers.Dense(16,input_shape=(8,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=1000, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history
model.save('model25K_30K.h5')


# We plot the chart of mean absolute error

# In[68]:


mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
n,m=500,1000
plt.plot(epochs[n:m],val_mean_absolute_error[n:m],'r',label='validation mean_absolute_error')
plt.plot(epochs[n:m],mean_absolute_error[n:m],'b',label='Training mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.show()


# In[69]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# In[70]:


list_next_mae.append(results[1])
list_next_r2.append(results[2])
list_next_distance.append(30)


# ## ......'25K' predict 'Official Time'

# In[71]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
data=data_all[['Age','M/F','5K','10K','15K','20K','Half','25K']]
target=df['Official Time']
data_train=data[15000:70000]
data_val=data[70000:]
data_test=data[0:15000]
target_train=target[15000:70000]
target_val=target[70000:]
target_test=target[0:15000]
model = models.Sequential()
model.add(layers.Dense(16,input_shape=(8,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(16, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=1200, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history
model.save('model25K_OT.h5')


# We plot the chart of mean absolute error

# In[72]:


mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
n,m=800,1200
plt.plot(epochs[n:m],val_mean_absolute_error[n:m],'r',label='validation mean_absolute_error')
plt.plot(epochs[n:m],mean_absolute_error[n:m],'b',label='Training mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.show()


# In[73]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# In[74]:


list_ot_mae.append(results[1])
list_ot_r2.append(results[2])
list_ot_distance.append(25)


# ## ......'30K' predict '35K'

# In[75]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
data=data_all[['Age','M/F','5K','10K','15K','20K','Half','25K','30K']]
target=df['35K']
data_train=data[15000:70000]
data_val=data[70000:]
data_test=data[0:15000]
target_train=target[15000:70000]
target_val=target[70000:]
target_test=target[0:15000]
model = models.Sequential()
model.add(layers.Dense(32,input_shape=(9,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=1000, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history
model.save('model30k_35k.h5')


# We plot the chart of mean absolute error

# In[76]:


mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
n,m=500,1000
plt.plot(epochs[n:m],val_mean_absolute_error[n:m],'r',label='validation mean_absolute_error')
plt.plot(epochs[n:m],mean_absolute_error[n:m],'b',label='Training mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.show()


# In[77]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# In[78]:


list_next_mae.append(results[1])
list_next_r2.append(results[2])
list_next_distance.append(35)


# ## ......'30K' predict 'Official Time'

# In[79]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
data=data_all[['Age','M/F','5K','10K','15K','20K','Half','25K','30K']]
target=df['Official Time']
data_train=data[15000:70000]
data_val=data[70000:]
data_test=data[0:15000]
target_train=target[15000:70000]
target_val=target[70000:]
target_test=target[0:15000]
model = models.Sequential()
model.add(layers.Dense(64,input_shape=(9,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=700, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history
model.save('model30k_OT.h5')


# We plot the chart of mean absolute error

# In[80]:


mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
n,m=400,700
plt.plot(epochs[n:m],val_mean_absolute_error[n:m],'r',label='validation mean_absolute_error')
plt.plot(epochs[n:m],mean_absolute_error[n:m],'b',label='Training mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.show()


# In[81]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# In[82]:


list_ot_mae.append(results[1])
list_ot_r2.append(results[2])
list_ot_distance.append(30)


# ## ......'35K' predict '40K'

# In[83]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
data=data_all[['Age','M/F','5K','10K','15K','20K','Half','25K','30K','35K']]
target=df['40K']
data_train=data[15000:70000]
data_val=data[70000:]
data_test=data[0:15000]
target_train=target[15000:70000]
target_val=target[70000:]
target_test=target[0:15000]
model = models.Sequential()
model.add(layers.Dense(32,input_shape=(10,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=1000, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history
model.save('model35k_40K.h5')


# We plot the chart of mean absolute error

# In[84]:


mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
n,m=500,1000
plt.plot(epochs[n:m],val_mean_absolute_error[n:m],'r',label='validation mean_absolute_error')
plt.plot(epochs[n:m],mean_absolute_error[n:m],'b',label='Training mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.show()


# In[85]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# In[86]:


list_next_mae.append(results[1])
list_next_r2.append(results[2])
list_next_distance.append(40)


# ## ......'35K' predict 'Official Time'

# In[87]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
data=data_all[['Age','M/F','5K','10K','15K','20K','Half','25K','30K','35K']]
target=df['40K']
data_train=data[15000:70000]
data_val=data[70000:]
data_test=data[0:15000]
target_train=target[15000:70000]
target_val=target[70000:]
target_test=target[0:15000]
model = models.Sequential()
model.add(layers.Dense(16,input_shape=(10,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=1000, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history
model.save('model35k_OT.h5')


# We plot the chart of mean absolute error

# In[88]:


mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
n,m=600,1000
plt.plot(epochs[n:m],val_mean_absolute_error[n:m],'r',label='validation mean_absolute_error')
plt.plot(epochs[n:m],mean_absolute_error[n:m],'b',label='Training mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.show()


# In[89]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# In[90]:


list_ot_mae.append(results[1])
list_ot_r2.append(results[2])
list_ot_distance.append(35)


# ## ......'40K' predict 'Official Time'

# In[91]:


def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
data=data_all[['Age','M/F','5K','10K','15K','20K','Half','25K','30K','35K','40K']]
target=df['40K']
data_train=data[15000:70000]
data_val=data[70000:]
data_test=data[0:15000]
target_train=target[15000:70000]
target_val=target[70000:]
target_test=target[0:15000]
model = models.Sequential()
model.add(layers.Dense(16,input_shape=(11,), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae',r2])
history=model.fit(data_train, target_train,epochs=1000, batch_size=5000,shuffle='true',validation_data=(data_val,target_val),verbose=0)
history_dict=history.history
model.save('model40K_OT.h5')


# We plot the chart of mean absolute error

# In[92]:


mean_absolute_error=history_dict['mean_absolute_error']
loss=history_dict['loss']
val_mean_absolute_error=history_dict['val_mean_absolute_error']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)
n,m=600,1000
plt.plot(epochs[n:m],val_mean_absolute_error[n:m],'r',label='validation mean_absolute_error')
plt.plot(epochs[n:m],mean_absolute_error[n:m],'b',label='Training mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.show()


# In[93]:


results=model.evaluate(data_test,target_test,verbose=0)
print('Test Loss:  ',results[0])
print('Test mae:   ',results[1])
print('Test r2:    ',results[2])


# In[94]:


list_next_mae.append(results[1])
list_next_r2.append(results[2])
list_next_distance.append(42)


# In[95]:


list_ot_mae.append(results[1])
list_ot_r2.append(results[2])
list_ot_distance.append(40)


# # Plot the Evaluation Chart

# ## Partial Prediction Chart

# ### Partial MAE Chart

# In[162]:


#plt.figure(figsize=(12,8))
plt.plot(list_next_distance,list_next_mae,'r',label='R2',mfc='r',ms=5,marker='o')
plt.title('Mean Absolute Error of partial Prediction')
plt.xlabel('Distance / KM')
plt.ylabel('MAE / Minutes')
for x, y in zip(list_next_distance, list_next_mae):
        plt.text(x, y+0.03, '%.2f'%y, ha='center', va='bottom', fontsize=10.5)
plt.savefig('fig'+str(int(time.time()))+'.jpg')
plt.legend()
plt.show()


# ### Partial R2 Chart

# In[142]:


import time
plt.plot(list_next_distance,list_next_r2,'b',label='R2',mfc='b',ms=5,marker='o')
plt.title('R2 of partial Prediction')
plt.xlabel('Distance / KM')
plt.ylabel('R2')
plt.legend()
plt.show()


# ## Official Time Prediction Chart

# ### Official Time MAE Chart

# In[163]:


plt.plot(list_ot_distance,list_ot_mae,'r',label='MAE',mfc='r',ms=5,marker='o')
plt.title('Mean Absolute Error of Official Time Prediction')
plt.xlabel('Distance')
plt.ylabel('MAE / Minutes')
for x, y in zip(list_ot_distance, list_ot_mae):
        plt.text(x, y+0.5, '%.2f'%y, ha='center', va='bottom', fontsize=10.5)
plt.savefig('fig'+str(int(time.time()))+'.jpg')
plt.legend()
plt.show()


# ### Official Time R2 Chart

# In[161]:


plt.plot(list_ot_distance,list_ot_r2,'b',label='R2',mfc='b',ms=5,marker='o')
plt.title('R2 of Official Time Prediction')
plt.xlabel('Distance / KM')
plt.ylabel('R2')
for x, y in zip(list_ot_distance, list_ot_r2):
        plt.text(x, y+0.005, '%.2f'%y, ha='center', va='bottom', fontsize=11)
plt.legend()
plt.show()

