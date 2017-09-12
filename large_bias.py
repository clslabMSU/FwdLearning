# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:51:26 2017

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
from fileOP import  writeRows
import tensorflow as tf
import numpy as np
import csv
import os


import matplotlib.pyplot as plt

from datetime import datetime

def makeFolder(addr):
    if not os.path.exists(addr):
        os.makedirs(addr)
        
start_time = str(datetime.now())



plt.rcParams["figure.figsize"] = [16, 9]
FILE_PATH  = "D:/CLS_lab/codeTest/fwl_project/training_data_100m_256e/processed_boolean_16.csv"
savefolder = "D:\\CLS_lab\\codeTest\\fwl_project\\savefile\\quad_funshuffle\\1stacked_48\\"



scaled = True
map_count = 100
min_map = 128
max_map = 200

UNROLL =  16
cell_count = 1 
hidden_layer_size = 48

input_size = 4
target_size = 24
num_epoch = 2000
lr = 0.01
epoch_threshold = 400

epoch_chkpt = [10, 100, 1000, 2000]
with tf.device("/cpu:0"):
    _inputs = tf.placeholder(tf.float32,shape=[None, None, input_size],
                              name='inputs')
    y_ = tf.placeholder(tf.float32, [None, 1], name = "label")
    
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
    
    def shuffle(data, label):
        new_data = []
        new_label = []
        order = np.random.choice(range(map_count), map_count, replace = False )
        for i in order:
            #Shuffle 3 times
#            r = np.random.randint(min_map, max_map, 1)[0]
#            r_permute = np.random.choice(range(max_map),r, replace = False )
#            temp_data =[data[i][j] for j in r_permute]
#            temp_label = [label[i][j] for j in r_permute]
            
            temp_data = [data[i][j] for j in range(max_map)]
            temp_label = [label[i][j] for j in range(max_map)]
            new_data.extend(temp_data)
            new_label.extend(temp_label)
        return new_data , new_label   
    
    class LSTM_cell(object):
    
        """
        LSTM cell object which takes 3 arguments for initialization.
        input_size = Input Vector size
        hidden_layer_size = Hidden layer size
        target_size = Output vector size
    
        """
    
        def __init__(self, input_size, hidden_layer_size, target_size):
    
            # Initialization of given values
            self.input_size = input_size
            self.hidden_layer_size = hidden_layer_size
            self.target_size = target_size
    
            # Weights and Bias for input and hidden tensor
            self.Wi = tf.Variable(tf.zeros(
                [self.input_size, self.hidden_layer_size]), name = "Wi")
            self.Ui = tf.Variable(tf.zeros(
                [self.hidden_layer_size, self.hidden_layer_size]),name ="Ui")
            self.bi = tf.Variable(tf.zeros([self.hidden_layer_size]),name = "bi")
    
            self.Wf = tf.Variable(tf.zeros(
                [self.input_size, self.hidden_layer_size]), name = "Wf")
            self.Uf = tf.Variable(tf.zeros(
                [self.hidden_layer_size, self.hidden_layer_size]), name = "Uf")
            self.bf = tf.Variable(tf.fill([self.hidden_layer_size], 2.0), name = "bf")
    
            self.Wog = tf.Variable(tf.zeros(
                [self.input_size, self.hidden_layer_size]), name = "Wog")
            self.Uog = tf.Variable(tf.zeros(
                [self.hidden_layer_size, self.hidden_layer_size]), name = "Uog")
            self.bog = tf.Variable(tf.zeros([self.hidden_layer_size]), name = "bc")
    
            self.Wc = tf.Variable(tf.zeros(
                [self.input_size, self.hidden_layer_size]), name = "Wc")
            self.Uc = tf.Variable(tf.zeros(
                [self.hidden_layer_size, self.hidden_layer_size]), name = "Uc")
            self.bc = tf.Variable(tf.zeros([self.hidden_layer_size]), name = "bc")
    
            # Weights for output layers
            self.Wo = tf.Variable(tf.truncated_normal(
                [self.hidden_layer_size, self.target_size], mean=0, stddev=.01), name = "Wo")
            self.bo = tf.Variable(tf.truncated_normal(
                [self.target_size], mean=0, stddev=.01), name = "bo")
            
            
    
            # Placeholder for input vector with shape[batch, seq, embeddings]
            #self._inputs = tf.placeholder(tf.float32,
                                          #shape=[None, None, self.input_size],
                                          #name='inputs')
    
            # Processing inputs to work with scan function
            self.processed_input = process_batch_input_for_RNN(_inputs)
    
    
            self.initial_hidden = _inputs[:, 0, :]
            self.initial_hidden = tf.matmul(
                self.initial_hidden, tf.zeros([input_size, hidden_layer_size]))
    
            self.initial_hidden = tf.stack(
                [self.initial_hidden, self.initial_hidden])
        # Function for LSTM cell.
    
        def Lstm(self, previous_hidden_memory_tuple, x):
            """
            This function takes previous hidden state and memory
             tuple with input and
            outputs current hidden state.
            """
            previous_hidden_state, c_prev = tf.unstack(previous_hidden_memory_tuple)
    
            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            ) #were sigmoid
    
             #Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            ) # were sigmoid
    
            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            ) #were sigmoid
    
            # New Memory Cell, or block input
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )
    
            # Final Memory cell
            c = f * c_prev + i * c_
    
            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)
    
            return tf.stack([current_hidden_state, c])
    
        # Function for getting all hidden state.
        def get_states(self):
            """
            Iterates through time/ sequence to get all hidden state
            """
    
            # Getting all hidden state throuh time
            all_hidden_states = tf.scan(self.Lstm,
                                        self.processed_input,
                                        initializer=self.initial_hidden,
                                        name='states')
            all_hidden_states = all_hidden_states[:, 0, :, :]
    
            return all_hidden_states
    
        # Function to get output from a hidden layer
        def get_output(self, hidden_state):
            """
            This function takes hidden state and returns output
            """
            output = tf.sigmoid(tf.matmul(hidden_state, self.Wo) + self.bo)
    
            return output
    
        # Function for getting all output layers
        def get_outputs(self):
            """
            Iterating through hidden states to get outputs for all timestamp
            """
            all_hidden_states = self.get_states()
    
            all_outputs = tf.map_fn(self.get_output, all_hidden_states)
    
            return all_outputs
    
    
    # Function to convert batch input data to use scan ops of tensorflow.
    def process_batch_input_for_RNN(batch_input):
        """
        Process tensor of size [5,3,2] to [3,5,2]
        """
        batch_input_ = tf.transpose(batch_input, perm=[2, 0, 1])
        X = tf.transpose(batch_input_) #in case we do batch processing
    
        return X
    
    def one_hot(m):
        if m[0][0] == 0:
            return [[1,0]]
        else:
            return [[0,1]]
    
    class LSTM_layer(object):
        def __init__(self, cell_count ,input_size, hidden_layer_size, target_size):
            self.input_size = input_size
            self.hidden_layer_size = hidden_layer_size
            self.target_size = target_size
            self.cell_count = cell_count
            self.LSTM_list = []
            for i in range(self.cell_count):
                self.LSTM_list.append(LSTM_cell(self.input_size, self.hidden_layer_size, self.target_size))
        
        def output(self):
            output = []
            for  i in range(self.cell_count):
                output.append(self.LSTM_list[i].get_outputs()[-1])
            return tf.reshape(tf.transpose(output, perm = [1,0,2]), [1,-1, target_size * cell_count])
        def output_debug(self):
            output = []
            for  i in range(self.cell_count):
                output.append(self.LSTM_list[i].get_outputs()[-1])
            return output
    
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
            row_index+=1
    
    #data = sklearn.utils.shuffle(data) shuffle data?
    
    #data = np.transpose(data)
    INPUT = data[0:2]
    LABEL = data[-1:]
    LABEL = np.transpose(LABEL)
    
    
    
    INPUT = np.transpose(INPUT)
    
    input = list(chunks(INPUT, max_map))
    label = list(chunks(LABEL, max_map))
    
    
    rnn = LSTM_layer(cell_count, input_size, hidden_layer_size, target_size)
    outputs = rnn.output()
    
    W2 = tf.Variable(tf.random_uniform([1,cell_count *target_size,6], minval = 0, maxval = 1), name = "Weight2")
    W3 = tf.Variable(tf.random_uniform([ 1, 6,1], minval = 0, maxval = 1), name = "Weight3")
    B2 = tf.Variable(tf.zeros([6]), name = "Bias2")
    B3 = tf.Variable(tf.zeros([1]), name = "Bias3")
    
    y3 = tf.tanh(tf.matmul(outputs, W2)  + B2, name = "output1")
    y4 = tf.sigmoid(tf.matmul(y3, W3) + B3, name = "output2")
    loss = tf.reduce_sum(tf.square(tf.subtract(y4, y_)), name = "squared_l")
    loss_ = tf.reduce_sum(tf.subtract(y_, y4), name = "unsigned_l")
    
    
    #sgd = tf.train.GradientDescentOptimizer(learning_rate  = 0.175).minimize(loss)
    #adam = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)
    rms = tf.train.RMSPropOptimizer(learning_rate = lr).minimize(loss)
    #ada = tf.train.AdagradOptimizer(learning_rate = 0.02).minimize(loss)
    #momentum = tf.train.MomentumOptimizer(learning_rate = 0.01, momentum = 0.001, use_nesterov= True).minimize(loss)
    #adadelta =  tf.train.AdadeltaOptimizer(learning_rate=0.01).minimize(loss)
    
    
    
    optimizer = rms
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    
    temp_debug = np.array([], ndmin = 3)
    #INPUT
    temp_loss = 0
    last_output = 0
    best_loss = float("INFINITY")
    
    all_best_loss = []
    all_loss = []
    
    saver0 = tf.train.Saver(max_to_keep = None)
    saver0.export_meta_graph(savefolder + "lr" + str(lr) + 'rms' + 'unroll' + str(UNROLL) + '.meta')
    
    cell_output =  []
    
    for epoch_i in range(num_epoch):
        print("Epoch ", epoch_i)
        counter = 0
        batch = np.array([]).reshape(0,UNROLL,input_size)
        
        loss_l = []
        INPUT, LABEL = shuffle(input, label)
        if epoch_i + 1 in epoch_chkpt:
            c_out  = np.array([],ndmin = 3).reshape(1,0,cell_count * target_size)
        for _ in range(len(INPUT)):
            # For error
            temp_ = INPUT[counter]
            #INPUT
            temp_ = np.insert(temp_,2,[temp_loss, last_output]).reshape(input_size)
            #temp.append(temp_)
            #print(np.shape(temp))
            temp_label =  LABEL[counter]
            temp_label = temp_label.reshape(1,1)
            temp_debug = np.append(temp_debug, np.array(temp_, ndmin = 3), axis  = (2 if (not(counter) and not (epoch_i)) else 1 ))
            #For batch
            
            if counter > 0 or epoch_i > 0:
                temp_debug = np.delete(temp_debug, 0 , axis = 1)
            
            while len(temp_debug[0]) < UNROLL:
                temp_debug = np.insert(temp_debug, 0, np.ones([1,1, input_size]), axis = 1)
            
            batch = np.append(batch, temp_debug, axis =0)
            last_output, temp_loss  =sess.run([y4,loss_], feed_dict = {_inputs : temp_debug, y_ : temp_label})
            
            if epoch_i + 1 in epoch_chkpt:
                c_out = np.append(c_out,sess.run(outputs, feed_dict = {_inputs : temp_debug, y_ : temp_label}), axis = 1 )
            
            
            loss_l.append(np.abs(temp_loss) )
            counter = counter + 1
        
        if epoch_i + 1 in epoch_chkpt:
            cell_output.append(c_out[0])
        
        m = np.mean(loss_l)
        s = np.std(loss_l)
        
        all_loss.append(m)
        
        print("####################################################")
        print("Mean:", m)
        print("Std:", s)
        
        p1 = plt.plot(range(len(loss_l)), loss_l)
        plt.legend(handles = p1)
        plt.show()
        
        if epoch_i > epoch_threshold and m < best_loss:
            best_loss  = m
            makeFolder(savefolder+"best\\")
            saver0.save(sess, savefolder + "best/model.ckpt", global_step = epoch_i, write_meta_graph= False)
            
            text_file = open(savefolder + "Epoch" + str(epoch_i)+".txt", "w")
            text_file.write(str(best_loss))
            text_file.close()
            all_best_loss.append(best_loss)
            
        
        #print(sess.run(y4,  feed_dict = {_inputs : batch, y_ : LABEL}))
        sess.run(optimizer,  feed_dict = {_inputs : batch, y_ : LABEL})
    
    
    
    
    
    
    
    makeFolder(savefolder + "final\\")
    saver0.save(sess, savefolder + "best/model.ckpt",  global_step = epoch_i +1 , write_meta_graph= False)
    
    
    
    
    writeRows(savefolder + "best_error.csv", np.transpose([all_best_loss]))
    writeRows(savefolder + "all_error.csv", np.transpose([all_loss]))
    
    
    for i in range(len(epoch_chkpt)):
        writeRows(savefolder + "cell_outputstr" + str((epoch_chkpt[i])) + ".csv"  , cell_output[i] )
    
    print(start_time)
    end_time  = str(datetime.now())
    print(end_time)
    
    #########################
    #
    #
    #FILE_PATH  = "D:\\CLS_lab\\codeTest\\fwl_project\\new_data_0_1_range_20000\\processed_boolean_14.csv"
    #folder ="D:\\CLS_lab\\codeTest\\fwl_project\\savefile\\sigmoidlast_cell6_hidden6_hidden2_rms0.01_unroll16\\"
    #
    #UNROLL = 16
    #
    #data_peak = np.recfromcsv(FILE_PATH, delimiter = ',') # peak through data to see number of rows and cols
    #
    #num_cols = len(data_peak[0])
    #num_rows = len(data_peak)
    #data  = np.zeros([num_rows+1, num_cols]) # num_cols - 1 means skip label col
    #
    #    
    #    
    #with open(FILE_PATH) as csvfile:
    #    row_index = 0
    #    reader= csv.reader(csvfile)
    #    for row in reader:
    #        for cols_index in range(num_cols):
    #            data[row_index][cols_index]= row[cols_index]
    #        row_index+=1
    #
    ##data = sklearn.utils.shuffle(data) shuffle data?
    #
    ##data = np.transpose(data)
    #INPUT = data[0:2]
    #LABEL = data[-1:]
    #LABEL = np.transpose(LABEL)
    #
    #
    #
    #INPUT = np.transpose(INPUT)
    #
    #loss = []
    #temp_loss = 0
    #counter = 0
    #temp_debug = np.array([], ndmin = 3)
    #
    #for _ in range(len(INPUT)):
    #    temp_ =  INPUT[counter]
    #    temp_ = np.insert(temp_,2,temp_loss).reshape(3)
    #    temp_label =  LABEL[counter]
    #    temp_label = temp_label.reshape(1,1)
    #    temp_debug = np.append(temp_debug, np.array(temp_, ndmin = 3), axis  = (2 if not(counter) else 1 ))
    #    if counter > UNROLL - 1:
    #        temp_debug = np.delete(temp_debug, 0 , axis = 1)
    #    temp_loss = sess.run(loss_, feed_dict = {_inputs : temp_debug, y_ : temp_label })
    #    c = sess.run(outputs, feed_dict = {_inputs : temp_debug})
    #    if counter % 1000 == 0:
    #        print(c)
    #    loss.append(np.abs(temp_loss))
    #    counter = counter + 1
    #
    #p1 = plt.plot(range(len(loss)), loss)
    #plt.legend(handles = p1)
    #plt.show()
