# FwdLearning
## Tensorflow implementation of Fixed-Weight Meta-Learning

### Experiment Settings
Run fwd_main.py with your prefered experiment settings.

Arguments:

* FILE_PATH: The csv file that contains the input to the network. The files are attached in the package.

* savefolder: Folder to save the check point file. For faster training time, we only save the model when it reaches a new best loss.

* LSTM: the architecture. Choice of nfg, cifg, lstm.

* Clipped: Clip or no clipp the memory cell.

* new_shuffle: experimentation with vary the length of each episode.

* new_loss: The use of meta loss or normal loss as the objective function to optimize.

* unroll: unroll parameter

* scaled: experiment with scaling the input to a specific range.

* noise: experiment with gaussian noise addition to the input.

* cell_count: number of LSTM connected vertically. Current configuration is 1. It's possible to make a n-ensemble of LSTM but observation is that this worsens performance. 

* hidden_layer_size: hidden layer size in the LSTM.

* input size: 4 - x1, x2, error(t-1), predict(t-1).

* target_size: size of the output layer of LSTM.

* number_epoch: number of epoch for training.

* lr: learning rate for the optimizer.

* epoch_threhold: not save the model unless passed this threshold.

* max_count: number of task episode in an epoch.

* min_map max_map: number of timestep in an episode. Recommended to make min_map == max_map since varying mapping length does not improve performance.

* interval: number of episodes. 

### Input file:
* row 1,2: x1 x2 at timestep t.

* row 3: label at timestep t-1.

* row 4 functional mapping.

We have 5 file corresponding 5 episode length used in the paper: 32 64 128 200 250.

### Save folder structure:

* Savefolder

  * meta graph file (.meta file)
  
  * Epoch400.txt
  
  * Epoch401.txt
  
  * ...
  
  * Checkpoint folder (named best)

To save storage space, we only save the meta graph once and then save the weight of the network in Checkpoint folder. The saved text file (Epoch400.txt) is the mean error for that specific checkpoint. 

### Load checkpoint test on a dataset

Run test_ckpt.py to load checkpoints

Arguments

* FILE_PATH: The csv file that contains the input to the network. The files are attached in the package.

* savefolder: Folder to save the check point file. This folder name is the same name as savefolder set in fwd_main.py

* UNROLL : the UNROLL parameter should match with that in fwd_main.py

* ckpt: specifies the check point number in the checkpoint folder (error if load a non existing checkpoint)

test_ckpt.py write out 2 files for further evaluation. The first file is plot.png that plot 10 the erorr of first 10 task episodes. error.csv is the error file across timesteps. 
