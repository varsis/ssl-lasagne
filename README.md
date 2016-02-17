# ssl-lasagne

### Requirements:
* Python
* PIP or easy_install (For automated install)

### Optional Installs (GPU Support):
[cuda]: <https://developer.nvidia.com/cuda-downloads>
[cudNN]: <https://developer.nvidia.com/cudnn>
* [cuda] - (better use run file (local))
* [cuDNN] - (more info see https://groups.google.com/forum/#!msg/caffe-users/nlnMFI0Mh7M/8Y4z1VCcBr4J). Be careful when installing cuda it might break the system (i.e. can't get into login screen, use recovery mode to remove the driver come with cuda and it should work fine after). Better make backups before installing.

### Automated Installation:
```sh 
$ git clone https://github.com/minhnhat93/ssl-lasagne.git ssl-lasagne
$ cd ssl-lasagne
```
> Using a virtual enviroment may not work, use with caution!

> setup.py uses 'pip' if you do not have this installed, check requirements.txt for the required packages.

```sh
$ python setup.py
```
Follow the prompts to download mnist.pkl, and install the requirements using pip

### Manual Installation

1. INSTALL theano "bleeding-edge" version: http://deeplearning.net/software/theano/install.html

2. INSTALL lasagne: https://github.com/Lasagne/Lasagne (installation instructions in README.md)

3. CLONE this git repository into a directory

4. DOWNLOAD mnist.pk.gz (http://deeplearning.net/data/mnist/mnist.pkl.gz) and unpack mnist.pkl into project directory

5. OPEN and RUN semisupervised_lasagne.py

### Parameters for Semisupervised Learning

Set the paramters in `semisupervised_lasagne.py`, find the section that contains the following code:
```
#-----------------------SET PARAMETERS-------------------------#
# Set the dimension here, 1 list = 1 stack, 2 list = 2 stacks, etc...
dimensions = [[1500, 3, 200]]  # example of 1 stack
#dimensions = [[1500,3,500],[1000,3,300]] # example of 2 stacks
# Set learning ratio for unsupervised, supervised and weights regularization
lr = (1.0, 1, 1e-4)
```
> Here you can set the number of LISTA stacks and learning ratio for loss components by adjusting each variable accordingly

### Set number of LISTA stacks:
1. stacks: ```dimensions = [[1500, 3, 200]]```. The structure has 1 stack with the dictionary size for LISTA structure is 1500,
 number of LISTA shrinkage layers is 3 and dimension of output after projection matrix is 200. This is only the first 
 encoding half of the structure, the decoding half dimensions is derived automatically
2. stacks: ```dimensions = [[1500,3,500],[1000,3,300]]```. 2 stacks with the number in each list has the similar meaning to the
1 stack case. 2nd stack will take output after projection matrix of 1st stack as input. Again, the decoding half 
dimensions is derived automatically

### Set learning ratio for loss components:

1. components are loss for unsupervised/reconstruction path, loss for supervised/classification path and regularizer loss
(i.e. total sum of weight matrices values) in respective order
Example: ```lr = (1.0, 1, 1e-4)```

### Running the Program

> You may wish to use the GPU, either setup .theanorc or use the alternate command

```sh
$ python semisupervised_lasagne.py
```

Alternate command (Your gpu device number may vary):
```sh
$ THEANO_FLAGS='device=gpu0' python semisupervised_lasagne.py
```

* CHOOSE A MODE: `"TEST"` or `"TRAIN"`

* IF `"TRAIN"` IS CHOSEN AND THERE IS A PREVIOUS MODEL FILE FROM LAST RUN (check variable LAST_MODEL_PATH): 
3 options: "CONTINUE" (train using last model as init value for parameters), "OVERRIDE" (train completely new model),
"END" (quit)


