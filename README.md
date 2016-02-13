# ssl-lasagne

INSTALL:

> 1. INSTALL cuda (https://developer.nvidia.com/cuda-downloads (better use run file (local))) and cudnn (https://developer.nvidia.com/cudnn, more info see https://groups.google.com/forum/#!msg/caffe-users/nlnMFI0Mh7M/8Y4z1VCcBr4J). Be careful when installing cuda it might break the system (i.e. can't get into login screen, use recovery mode to remove the driver come with cuda and it should work fine after). Better make backups before installing.

> 2. INSTALL theano: http://deeplearning.net/software/theano/install.html

> 3. INSTALL lasagne: https://github.com/Lasagne/Lasagne (installation instructions in README.md)

> 4. CLONE this git repository into a directory

> 5. DOWNLOAD mnist.pk.gz (http://deeplearning.net/data/mnist/mnist.pkl.gz) and unpack mnist.pkl into project directory

> 6. OPEN and RUN semisupervised_lasagne.py

IN FILE SEMISUPERVISED_LASAGNE.PY:

EDIT PARAMETERS:

> MOVE to "SET PARMETERS" section

Here you can set the number of LISTA stacks and learning ratio for loss components

1. Set number of LISTA stacks

Example:

1 stacks: dimensions = [[1500, 3, 200]]. The structure has 1 stack with the dictionary size for LISTA structure is 1500,
 number of LISTA shrinkage layers is 3 and dimension of output after projection matrix is 200. This is only the first 
 encoding half of the structure, the decoding half dimensions is derived automatically

2 stacks: dimensions = [[1500,3,500],[1000,3,300]]. 2 stacks with the number in each list has the similar meaning to the
1 stack case. 2nd stack will take output after projection matrix of 1st stack as input. Again, the decoding half 
dimensions is derived automatically

2. Set learning ratio for loss components:

3 components are loss for unsupervised/reconstruction path, loss for supervised/classification path and regularizer loss
(i.e. total sum of weight matrices values) in respective order
Example: lr = (1.0, 1, 1e-4)

RUN:
> 1. RUN the file

> 2. CHOOSE A MODE: "TEST" or "TRAIN"

> 3. IF "TRAIN" IS CHOSEN AND THERE IS A PREVIOUS MODEL FILE FROM LAST RUN (check variable LAST_MODEL_PATH): 
3 options: "CONTINUE" (train using last model as init value for parameters), "OVERRIDE" (train completely new model),
"END" (quit)
