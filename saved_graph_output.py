# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:48:08 2017

@author: thy1995
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:42:40 2017

@author: thy1995
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 09:51:57 2017

@author: thy1995
"""

import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from fileOP import writeRows
from os.path import isfile, join
from os import listdir
tf.reset_default_graph()
plt.rcParams["figure.figsize"] = [16, 9]
FILE_PATH  = "\\\\EGR-1L11QD2\\CLS_lab\\codeTest\\fwl_project\\test data for 16 functions\\processed_boolean_16_100m_32e.csv"
folder = "D:\\CLS_lab\\codeTest\\fwl_project\\savefile\\newloss\\16coupled_clipped\\"
record  = folder  + "best\\"

onlyfiles = [(record + f) for f in listdir(record) if isfile(join(record, f))]
checkpoints =  [a[0:-6] for a in onlyfiles if a[-6:] == ".index"]

UNROLL = 40

data_peak = np.recfromcsv(FILE_PATH, delimiter = ',') # peak through data to see number of rows and cols

num_cols = len(data_peak[0])
num_rows = len(data_peak)
data  = np.zeros([num_rows+1, num_cols]) # num_cols - 1 means skip label col
interval = 10

    
    
with open(FILE_PATH) as csvfile:
    row_index = 0
    reader= csv.reader(csvfile)
    for row in reader:
        for cols_index in range(num_cols):
            data[row_index][cols_index]= row[cols_index]
        row_index+=1

#data = sklearn.utils.shuffle(data) shuffle data?

#data = np.transpose(data)
INPUT = data[0:2]
LABEL = data[-1:]
LABEL = np.transpose(LABEL)



INPUT = np.transpose(INPUT)

loss_ckpt = []

best_ckpt = " "
best =  float("INFINITY")
with tf.Session() as sess:
  tf_saver = tf.train.import_meta_graph(folder + "lr0.01rmsunroll40.meta")
  #tf.global_variables_initializer()
  for ckpt in checkpoints:
      loss = []
      temp_loss = 0
      last_output = 0
      counter = 0
      temp_debug = np.array([], ndmin = 3)
      tf_saver.restore(sess,  ckpt)  
      for _ in range(len(INPUT)):
          temp_ =  INPUT[counter]
          temp_ = np.insert(temp_,2,[temp_loss, last_output]).reshape(4)
          temp_label =  LABEL[counter]
          temp_label = temp_label.reshape(1,1)
          temp_debug = np.append(temp_debug, np.array(temp_, ndmin = 3), axis  = (2 if not(counter) else 1 ))
          if counter > UNROLL - 1:
              temp_debug = np.delete(temp_debug, 0 , axis = 1)
          last_output ,temp_loss = sess.run(["output2:0","unsigned_l:0"], feed_dict = {"inputs:0" : temp_debug, "label:0" : temp_label })
          loss.append(np.abs(temp_loss))
          counter = counter + 1
      p1 = plt.plot(range(len(loss)), loss)
      plt.legend(handles = p1)
      plt.show()
      current_l = np.mean(loss)
      new_loss = np.mean([loss[i] for i in range(len(loss)) if i % 128 > interval])
      print(current_l)
      print("New loss", new_loss)
      loss_ckpt.append([ckpt.rsplit("\\")[-1].split("-")[-1], current_l, new_loss])
      if current_l < best:
          best_ckpt = ckpt
          best = current_l
        
        #writeRows(folder + "error128_1201.csv", np.transpose([loss]))

print(best_ckpt)
print(best)
writeRows(folder + "128.csv", loss_ckpt)

