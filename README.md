# Stanford CS348n Homework 1

## Dependencies

This codebase uses Python 3.6 and PyTorch 1.5.1.

Please install the python environments

    pip install -r requirements.txt


## Problem 1

Please use `prob1.ipynb` as the starting point.


## Problem 2 and 3

First, go to the folder `cd` and follow the README there to compile the GPU implementation of Chamfer distance.

Then, download [the data](http://download.cs.stanford.edu/orion/cs348n/chair_dataset.zip) and unzip it under the home directory.

For problem 2, implement the encoder `PointNet` and decoder `FCDecoder` in `model.py` and run

    python train_ae.py

For problem 3, implement the variational computation in `Sampler` in `model.py` and run

    python train_vae.py

Check the file `randgen.py` for the code snippet for random shape generation.

If you implement things correctly, you should be able to see promising reconstruction results in 5-10 min of training. 
Training for the whole 1000 epoches roughly will take 30 min on a decent GPU.
Train it longer to get better results though until the validation curve goes plateau.

### Codebase Structure and Experiment Log

There are four main files

  * `model.py`: defines the NN (this is the only file you need to modify!);
  * `data.py`: contains the dataset and dataloader code;
  * `train_ae.py` and `train_vae.py`: the main trainer code;
  * `train_utils.py`: contain utilify functions.

Each experiment run will output a directory with name `exp-ae` for AE and `exp-vae` for VAE, and it contains

  * `ckpts`: store the model checkpoint every 1000 steps of training;
  * `train` and `val`: store the tensorboard information, used for training and validation curve visualization;
  * `val_visu`: contains the validation results for one batch every 10 epoches of training;
  * `conf.pth`: stores all parameters for the training;
  * `data.py` and `train_ae.py`: backups the training and data code.
  * `train_log.txt`: stores the console output for the training.

## Questions

Please ask TA or post on Piazza for any question.
Please do not use the Github Issue for Q&As for the class.

