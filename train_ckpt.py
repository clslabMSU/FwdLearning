# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 21:17:40 2017

@author: thy1995
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:51:43 2017

@author: thy1995
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:29:34 2017

@author: thy1995
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:03:29 2017

@author: thy1995
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:22:02 2017

@author: thy1995
"""

import tensorflow as tf
import numpy as np
import csv
import os
from fileOP import  writeRows
import random

import matplotlib.pyplot as plt

from datetime import datetime

plt.rcParams["figure.figsize"] = [16, 9]


def makeFolder(addr):
    if not os.path.exists(addr):
        os.makedirs(addr)

start_time = str(datetime.now())

FILE_PATH  = "results_with_operator_16_100m_250e.csv"
loadfolder = "C:\\savefile\\partial\\15_9_cont\\"
savefolder = "C:/savefile/partial/15_9_contcont/"

LSTM = "nfg" #nfg, cifg, full lstm
clipped = True
new_shuffle = True
new_loss = True
UNROLL =  40
scaled = False
noise = False
cell_count = 1 
hidden_layer_size = 24
input_size = 4
target_size = 12
num_epoch = 20000
lr = 0.01
epoch_threshold = 1000
map_count = 100
min_map = 128
max_map = 250

interval = 10

target_map  = max_map
holdOut = [15,9]
holdOutname = "_".join(str(x) for x in holdOut)


loadckpt = 2631

makeFolder(savefolder)
tf.reset_default_graph()



def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def shuffle(data, label):
    global target_map
    new_data = []
    new_label = []
    order = np.random.choice(range(map_count), map_count, replace = False )
    if new_shuffle:
        target_map = random.choice([min_map, max_map])
    else:
        target_map = max_map
    for i in order:
        add = np.random.normal(0, 0.01) if noise else 0
        temp_data = [data[i][j] + add for j in range(target_map)]
        temp_label = [label[i][j] + add  for j in range(target_map)]
        new_data.extend(temp_data)
        new_label.extend(temp_label)
    return new_data , new_label   

with tf.device("/cpu:0"):
    data_peak = np.recfromcsv(FILE_PATH, delimiter = ',') # peak through data to see number of rows and cols
    num_cols = len(data_peak[0])
    num_rows = len(data_peak)
    data  = np.zeros([num_rows+1, num_cols]) # num_cols - 1 means skip label col
 
    with open(FILE_PATH) as csvfile:
        row_index = 0
        reader= csv.reader(csvfile)
        for row in reader:
            for cols_index in range(num_cols):
                data[row_index][cols_index]= row[cols_index]
                if scaled:
                    if data[row_index][cols_index] == 0:
                        data[row_index][cols_index] == 0.1
                    elif data[row_index][cols_index] == 1:
                        data[row_index][cols_index] == 0.9
            row_index+=1
            
    task_train =  [i for i in range(len(data[-1])) if data[-1][i] not in holdOut]
    task_test =  [i for i in range(len(data[-1])) if data[-1][i] in holdOut]
    testFunction = [data[-1][i] for i in range(len(data[-1])) if data[-1][i] in holdOut]
    
    INPUT = data[0:2]
    LABEL = data[-2]
    testLABEL = LABEL[:][task_test]
    INPUT = np.transpose(INPUT)
    testINPUT = INPUT[:]
    LABEL = LABEL[task_train]
    LABEL = np.transpose([LABEL])
    
    INPUT = INPUT[task_train]
    
    testINPUT = testINPUT[task_test]
    testINPUT = testINPUT.T.tolist()
    
    testINPUT.append(testLABEL)
    testINPUT.append(testFunction)
    
    writeRows(savefolder + "holdout" + holdOutname + '.csv' , testINPUT)
    
    
    map_count = int(len(INPUT) / max_map)
    
    input = list(chunks(INPUT, max_map))
    label = list(chunks(LABEL, max_map))
#    rnn = LSTM_layer(cell_count, input_size, hidden_layer_size, target_size)
#    outputs = rnn.output()
#    W2 = tf.Variable(tf.random_uniform([1,cell_count *target_size,6], minval = 0, maxval = 1), name = "Weight2")
#    W3 = tf.Variable(tf.random_uniform([ 1, 6,1], minval = 0, maxval = 1), name = "Weight3")
#    B2 = tf.Variable(tf.zeros([6]), name = "Bias2")
#    B3 = tf.Variable(tf.zeros([1]), name = "Bias3")
#
#    y3 = tf.tanh(tf.matmul(outputs, W2)  + B2, name = "output1")
#    y4 = tf.sigmoid(tf.matmul(y3, W3) + B3, name = "output2")
#    loss = tf.reduce_sum(tf.square(tf.subtract(y4, y_)), name = "squared_l")
#    loss_ = tf.reduce_sum(tf.subtract(y_, y4), name = "unsigned_l")
#    rms = tf.train.RMSPropOptimizer(learning_rate = lr).minimize(loss, name = "update_ops")
#    optimizer = rms
#    sess = tf.Session()
#    init = tf.global_variables_initializer()
#    sess.run(init)
    with tf.Session() as sess:
        
        temp_debug = np.array([], ndmin = 3)
        #INPUT
        temp_loss = 0
        last_output = 0
        best_loss = float("INFINITY")
        all_best_loss = []
        all_loss = []
        
        tf_saver = tf.train.import_meta_graph(loadfolder + "lr0.01rmsunroll" + str(UNROLL) + ".meta")
        tf_saver.restore(sess,  loadfolder + "best/model.ckpt-" + str(loadckpt))
        print("OKAY")
         
        saver0 = tf.train.Saver(max_to_keep = None)
        saver0.export_meta_graph(savefolder + "lr" + str(lr) + 'rms' + 'unroll' + str(UNROLL) + '.meta')
        
    
        cell_output =  []
        
        for epoch_i in range(num_epoch):
            print("Epoch ", epoch_i)
            counter = 0
            batch = np.array([]).reshape(0,UNROLL,input_size)
            
            loss_l = []
            INPUT, LABEL = shuffle(input, label)
            for index in range(len(INPUT)):
    
                temp_ = INPUT[counter]
                temp_ = np.insert(temp_,2,[temp_loss, last_output]).reshape(input_size)
                temp_label =  LABEL[counter]
                temp_label = temp_label.reshape(1,1)
                temp_debug = np.append(temp_debug, np.array(temp_, ndmin = 3), axis  = (2 if (not(counter) and not (epoch_i)) else 1 ))
                if counter > 0 or epoch_i > 0:
                    temp_debug = np.delete(temp_debug, 0 , axis = 1)
                
                while len(temp_debug[0]) < UNROLL:
                    temp_debug = np.insert(temp_debug, 0, np.ones([1,1, input_size]), axis = 1)
                if new_loss:
                    interval_step = index % target_map
                    if interval_step > interval:
                        batch = np.append(batch, temp_debug, axis =0)
                else:
                    batch = np.append(batch, temp_debug, axis =0)
                last_output, temp_loss  =sess.run(["output2:0","unsigned_l:0"], feed_dict = {"inputs:0" : temp_debug,"label:0" : temp_label})
    
                loss_l.append(np.abs(temp_loss) )
                counter = counter + 1
    
            m = np.mean(loss_l)
            s = np.std(loss_l)
            all_loss.append(m)
            print("####################################################")
            print("Mean:", m)
            print("Std:", s)
            
            p1 = plt.plot(range(len(loss_l)), loss_l)
            plt.legend(handles = p1)
            plt.show()
            if  m < best_loss:
                best_loss  = m
                makeFolder(savefolder+"best\\")
                saver0.save(sess, savefolder + "best/model.ckpt", global_step = epoch_i + loadckpt, write_meta_graph= False)
                
                text_file = open(savefolder + "Epoch" + str(epoch_i + loadckpt)+".txt", "w")
                text_file.write(str(best_loss))
                text_file.close()
                all_best_loss.append(best_loss)
            if new_loss:
                LABEL = [ LABEL[i] for i in range(len(LABEL)) if i% target_map > interval]
            sess.run("update_ops",  feed_dict = {"inputs:0" : batch, "label:0" : LABEL})
    
        saver0.save(sess, savefolder + "best/model.ckpt",  global_step = epoch_i + loadckpt + 1 , write_meta_graph= False)
    
        writeRows(savefolder + "best_error.csv", np.transpose([all_best_loss]))
        writeRows(savefolder + "all_error.csv", np.transpose([all_loss]))
    
        print(start_time)
        #end_time  = str(datetime.now())
        #print(end_time)

 