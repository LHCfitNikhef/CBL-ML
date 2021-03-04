#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 23:42:50 2020

@author: isabel
"""
#OLD

print('Importing packages...')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from matplotlib import cm
import os
import csv
import warnings
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
import tensorflow.compat.v1 as tf
from copy import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(100)

warnings.filterwarnings('ignore')
print('done')

print('Importing datafiles...')
from Load_data import *
from Functions import  *
print('done')


#%%

wl1 = 50
wl2 = 100

df_dx = pd.DataFrame()

all_files = [file14, file15, file16, file17, file19, file20, file21, file22, file23]

for i,j in enumerate([14,15,16,17,19,20,21,22,23]):
    df_dx['x%(j)s' % {"j": j}]  =  all_files[i]['x_shifted']
    df_dx['y%(j)s' % {"j": j}]  =  smooth(all_files[i]['y_norm'], wl1)
    df_dx['derivative y%(j)s' %{"j": j}] = np.divide(df_dx['y%(j)s'% {"j": j}].diff(), \
                                                     df_dx['x%(j)s'% {"j": j}].diff())
    df_dx['smooth derivative y%(j)s' %{"j": j}] = smooth(df_dx['derivative y%(j)s' %{"j": j}], wl2)


#%%

li = []

for i in ([14,15,16,19,20,21]):
    crossing = df_dx[(df_dx['derivative y%(i)s' %{"i": i}] > 0) & (df_dx['x%(i)s'% {"i": i}] > 1)]['x%(i)s'% {"i": i}].min()
    li.append(crossing)
    
dE1 = min(li)
dE1_min = np.round(dE1, 3)
print("The value of dE1 is", dE1_min)


#%%

nrows, ncols = 2,1
gs = matplotlib.gridspec.GridSpec(nrows,ncols)
plt.figure(figsize=(ncols*7,nrows*4.5))

cm_subsection = np.linspace(0,1,24) 
colors = [cm.viridis(x) for x in cm_subsection]

hfont = rc('font',**{'family':'sans-serif','sans-serif':['Sans Serif']})
          
for i in range(2):
    ax = plt.subplot(gs[i])
    #ax.set_xlim([0,9])
    #ax.tick_params(which='major',direction='in',length=7)
    #ax.tick_params(which='minor',length=8)
    plt.axhline(y=0, color='black', linewidth=1, alpha=.8)
    #plt.axvline(x=0, color='darkgray', linestyle='--', linewidth = 1)
    #plt.axvline(x=dE1, color='darkgray', linestyle='--', linewidth = 1, label='$\Delta$E1' %{'s': dE1})
    
    for j in ([17,22,23]):
        if i == 0:
            p2 = ax.plot(df_dx['x%(i)s'% {"i": j}],df_dx['derivative y%(i)s' %{"i": j}], color=colors[j], label='%(i)s' %{"i": j})
            
        
    for j in ([14,15,16,19,20,21]):
        k = j-3
        
        if i == 0:
            p1 = ax.plot(df_dx['x%(i)s'% {"i": j}],df_dx['derivative y%(i)s' %{"i": j}], color=colors[-k], label='%(i)s' %{"i": j})
            ax.set_ylim([-.002, .001])
            ax.set_xlim([0, 6])
            ax.set_ylabel('dI/dE',fontsize=18)
            ax.set_yticks([-0.002, -0.001, 0, 0.001])
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.legend(loc=2, fontsize=16)
        
    for j in ([17,22,23]):   
        if i == 1: 
            ax.axhline(y=1, linestyle='-', color='gray')
            p1 = ax.plot(df_dx['x%(i)s'% {"i": j}], \
                         np.divide(df_dx['derivative y14'],df_dx['derivative y%(i)s'%{"i": j}]), 'k--', label='Ratio sp4/sp'%{"i":j})
            
            ax.axvline(x=1.65, linestyle='--')
            ax.set_ylim([-1, 2])
            ax.set_xlim([.5,3.5])   
            ax.set_ylabel('R = dI/dE(sample) / dI/dE(vac)', fontsize=18)
            ax.set_xlabel('$\Delta$E (eV)', fontsize=218)
            ax.legend()  
    
    if i == 0:
        ax.tick_params(labelbottom=True)
        ax.tick_params(which='major', length= 10, labelsize=18)
        ax.tick_params(which='minor', length= 10, labelsize=10)
    if i == 1:
        ax.set_xlabel('Energy loss (eV)', fontsize=24)
        ax.tick_params(length= 10, labelsize=18)
        ax.tick_params(which='major', length= 10, labelsize=18)
        ax.tick_params(which='minor', length= 10, labelsize=10)
    
plt.tight_layout()
#plt.savefig("Derivatives.pdf")
plt.show()



#%%

df_vacmean = pd.DataFrame()
nbins = 150
df_vacuum = df_vacuum[(df_vacuum['x_shifted'] < 20) & (df_vacuum['x_shifted'] > -.5)]
df_vacmean['x'] = np.linspace(df_vacuum['x_shifted'].min(),df_vacuum['x_shifted'].max(), nbins)
df_vacmean['y'], df_vacmean['sigma'] = binned_statistics(df_vacuum['x_shifted'], (df_vacuum['y']), nbins)[0:2]
df_vacmean['ratio'] = np.divide(df_vacmean['y'], df_vacmean['sigma'])

dE2 = df_vacmean['x'][df_vacmean['ratio'] < 1].min()
dE2 = np.round(dE2)
print("The value for dE_II is", (dE2))


#%%

nrows, ncols = 1,1
gs = matplotlib.gridspec.GridSpec(nrows,ncols)
plt.figure(figsize=(ncols*5,nrows*3.5))

cm_subsection = np.linspace(0,1,24) 
colors = [cm.viridis(x) for x in cm_subsection]

hfont = rc('font',**{'family':'sans-serif','sans-serif':['Sans Serif']})

ax = plt.subplot(gs[0])
ax.set_title('Intensity to sigma ratio', fontsize=16)
ax.set_xlim([-1,15])
ax.set_xlabel('Energy loss (eV)', fontsize=14)
ax.tick_params(which='major',direction='in',length=7, labelsize=14)
ax.tick_params(which='minor',length=8)
p1 = ax.plot(df_vacmean['x'],smooth(np.divide(df_vacmean['y'], df_vacmean['sigma']), 10), color=colors[0])
ax.axhline(y=1, linestyle='-')
ax.axvline(x=dE2, linestyle='dotted', linewidth='2')
plt.show()

print('The values of dE1 and dE2:', np.round(dE1,2), "eV and", dE2, "eV")

#%%

df_window = df[(df['x_shifted'] < dE1) & (df['x_shifted'] > -.5)]
df_window_vacuum = df_vacuum[(df_vacuum['x_shifted'] <= dE1) & (df_vacuum['x_shifted'] > -.5)]

df_mean, df_vacmean = pd.DataFrame(), pd.DataFrame()
nbins = 30

test1 = df_window['x_shifted'].min()
test2 = df_window['x_shifted'].max()

df_mean['x'] = np.linspace(df_window['x_shifted'].min(),df_window['x_shifted'].max(), nbins)
df_mean['y'], df_mean['sigma'] = binned_statistics(df_window['x_shifted'], np.log(df_window['y']), nbins)[0:2]

df_vacmean['x'] = np.linspace(df_window_vacuum['x_shifted'].min(),df_window_vacuum['x_shifted'].max(), nbins)
df_vacmean['y'], df_vacmean['sigma'] = binned_statistics(df_window_vacuum['x_shifted'], np.log(df_window_vacuum['y']), nbins)[0:2]

print("Training data points for DeltaE > DeltaE_I have been removed.")
print("Experimental mean and sigma are calculated.")



#%%


min_x = dE2
max_x = 16
N_pseudo = 20

df_pseudo = pd.DataFrame({'x':np.linspace(min_x, max_x, N_pseudo),'y': .5 * np.ones(N_pseudo), \
                    'sigma': .08 * np.ones(N_pseudo)})
df_full = pd.concat([df_mean, df_pseudo])

print('Pseudo data points added for Delta E > DeltaE_II')
print('Training data set "df_full" has been created')

df_full.describe()


#%%

def make_model(inputs, n_outputs):
    hidden_layer_1 = tf.layers.dense(inputs, 10, activation=tf.nn.sigmoid)
    hidden_layer_2 = tf.layers.dense(hidden_layer_1, 15, activation=tf.nn.sigmoid)
    hidden_layer_3 = tf.layers.dense(hidden_layer_2, 5, activation=tf.nn.relu)
    output = tf.layers.dense(hidden_layer_3, n_outputs, name='outputs', reuse=tf.AUTO_REUSE)
    return output

print("NN is initialized.")


#%%

#tf.compat.v1.get_default_graph
#tf.compat.v1.disable_eager_execution()

tf.get_default_graph
tf.disable_eager_execution()


x = tf.placeholder("float", [None, 1], name="x")
y = tf.placeholder("float", [None, 1], name="y")
sigma = tf.placeholder("float", [None, 1], name="sigma")

predictions = make_model(x,1)

df_train_full = df_full
df_train_full = df_train_full.drop_duplicates(subset = ['x']) # Only keep one copy per x-value

N_full = len(df_train_full['x'])

full_x = np.copy(df_train_full['x']).reshape(N_full,1)
full_y = np.copy(df_train_full['y']).reshape(N_full,1)
full_sigma = np.copy(df_train_full['sigma']).reshape(N_full,1)

N_pred = 3000
pred_min = -.5
pred_max = 20
predict_x = np.linspace(pred_min,pred_max,N_pred).reshape(N_pred,1)

print("Dataset is split into train subset (80%) and validation subset (20%)")




#%%

Nrep = 1000

full_y_reps = np.zeros(shape=(N_full, Nrep))
i=0
while i < Nrep:
        full_rep = np.random.normal(0, full_sigma)
        full_y_reps[:,i] = (full_y + full_rep).reshape(N_full)
        i+=1 
        
std_reps = np.std(full_y_reps, axis=1)
mean_reps = np.mean(full_y_reps, axis=1)

print('MC pseudo data has been created for 1000 replicas')


N_train = int(.8 * N_full)
N_test = int(.2 * N_full)

#%%

import time
from datetime import datetime
now = datetime.now()

def function_train():
    
    chi_array = []
    
    cost = tf.reduce_mean(tf.square((y-predictions)/sigma), name="cost_function")
    eta = 6e-3
    optimizer = tf.train.RMSPropOptimizer(learning_rate=eta, decay=0.9, momentum=0.0, epsilon=1e-10).minimize(cost)
    saver = tf.train.Saver(max_to_keep=1000)
    
    print("Start training on", '%04d'%(N_train), "and validating on",'%0.4d'%(N_test), "samples")
    
    Nrep = 100

    for i in range(0,25):
        
        full_y = full_y_reps[:, i].reshape(N_full,1)
        
        train_x, test_x, train_y, test_y, train_sigma, test_sigma = \
            train_test_split(full_x, full_y, full_sigma, test_size=.2)
    
        print(len(train_x))
        train_x, test_x = train_x.reshape(N_train,1), test_x.reshape(N_test,1)
        train_y, test_y = train_y.reshape(N_train,1), test_y.reshape(N_test,1)
        train_sigma, test_sigma = train_sigma.reshape(N_train,1), test_sigma.reshape(N_test,1)
        
        
        ### Train and validate
        prev_test_cost = 0
        prev_epoch = 0
        avg_cost = 0

        array_train = []
        array_test = []

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            
            training_epochs  = 1000
            display_step = 500

            for epoch in range(training_epochs):

                _, c = sess.run([optimizer, cost], 
                                feed_dict={
                                    x: train_x,
                                    y: train_y,
                                    sigma: train_sigma
                                })

                avg_cost = c
                
                # test_cost = cost.eval({x: test_x, y: test_y, sigma: test_sigma})


                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "| Training cost=", "{:.9f}".format(avg_cost), "| Validation cost=", "{:.9f}".format(test_cost))
                    array_train.append(avg_cost)
                    array_test.append(test_cost)
                    saver.save(sess, 'Models/All_models/my-model.ckpt', global_step=epoch , write_meta_graph=False) 

                    
                elif test_cost < prev_test_cost:
                    prev_test_cost = test_cost
                    prev_epoch = epoch

            best_iteration = np.argmin(array_test) 
            best_epoch = best_iteration * display_step
            best_model = 'Models/All_models/my-model.ckpt-%(s)s' % {'s': best_epoch}

            print("Optimization %(i)s Finished! Best model after epoch %(s)s" % {'i': i, 's': best_epoch})
            


            dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
            d_string = now.strftime("%d.%m.%Y")
            t_string = now.strftime("%H:%M:%S")
            
            saver.restore(sess, best_model)
            saver.save(sess, 'Models/Best_models/%(s)s/best_model_%(i)s' % {'s': d_string, 'i': i})


            predictions_values = sess.run(predictions, 
                                feed_dict={
                                    x: train_x,
                                    y: train_y 
                                }) 


            extrapolation = sess.run(predictions,
                                feed_dict={
                                    x: predict_x
                                })
            

        sess.close()
        

        nownow = datetime.now()
        print("time elapsed", nownow-now)

        a = np.array(train_x).reshape(N_train,)
        b = np.array(train_y).reshape(N_train,)
        c = np.array(predictions_values).reshape(N_train,)
        
        d = array_train
        e = array_test
       
        k = np.array(predict_x).reshape(N_pred,)
        l = np.array(extrapolation).reshape(N_pred,)
        
        path_to_data = 'Data/Results/%(date)s/'% {"date": d_string} 
        
        np.savetxt(path_to_data + 'Predictions_%(k)s.csv' % {"k": i}, list(zip(a,b,c)),  delimiter=',', fmt='%f')
        np.savetxt(path_to_data + 'Cost_%(k)s.csv' % {"k": i}, list(zip(d,e)),  delimiter=',',fmt='%f')
        np.savetxt(path_to_data + 'Extrapolation_%(k)s.csv' % {"k":i}, list(zip(k, l)),  delimiter=',', fmt='%f')
    
    
    plt.figure(figsize=(10,5))
    plt.plot(train_x, train_y, '.', label='train')
    plt.plot(test_x, test_y, '.', label='test')
    plt.axvline(x=dE1, color='lightgray')
    plt.axvline(x=dE2, color='lightgray')
    plt.title('Visualization of training data', fontsize=15)
    plt.ylabel('Log intensity', fontsize=14)
    plt.xlabel('Energy loss (eV)', fontsize=14)
    plt.legend(fontsize=14)
    plt.show()
    
 #%%
 
function_train()
 
#%%



 #%%
 
d_string = '09.11.2020'
good_files = np.ones(100)
prediction_file = pd.DataFrame()

predict_x = np.linspace(-0.5, 20, 1000).reshape(1000,1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(0,25):
        if good_files[i] == 1:
            best_model = 'Models/Best_models/%(s)s/best_model_%(i)s'% {'s': d_string, 'i': i}
            #saver = tf.compat.v1.train.Saver(max_to_keep=1000)
            saver = tf.train.Saver(max_to_keep=1000)
            saver.restore(sess, best_model)

            extrapolation = sess.run(predictions,
                                    feed_dict={
                                    x: predict_x
                                    })

            prediction_file['prediction_%(i)s' % {"j": j, "i": i}] = extrapolation.reshape(1000,)
            

#%%

for i in range(0,25):
    plt.plot(predict_x, np.exp(prediction_file.iloc[:,i]))
    plt.xlim([1, 5])
    plt.ylim([0, 10000])














