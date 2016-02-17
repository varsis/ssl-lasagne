#!/usr/bin/env python

import os, urllib, gzip
from subprocess import call

# Current Path of the file
path = os.path.dirname(os.path.realpath(__file__))

# Make sure we are executing where this script resides
os.chdir(path)

# Get files, want to check if we have mnist data base
files = os.listdir(path)

# Download and unzip file
if 'mnist.pkl' not in files:
    mnist = urllib.URLopener()
    mnist.retrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
    call(['gunzip','mnist.pkl.gz'])

def getInput():
    while True:
        answer = raw_input()
        answer.lower()

        if answer == 'y' or answer == 'yes':
            # Setup env
            return True
        elif answer == 'n' or answer == 'no':
            return False

print "(EXPERIMENTAL) Do you want to setup a virtual enviroment? (y/n) "
useenv = getInput()

# Put env in env
if useenv:
    call(['virtualenv','env'])
    call(['source','env/bin/activate'])
    # Next Install requirments for package
    call(['pip','install','-r','requirements.txt'])

    print "To enter the virtual enviroment next time use: 'source env/bin/activate'"

else:
    # Install in user folder
    print "Do you want to install the requirments globally? Otherwise at user level (y/n)"
    if getInput():
        call(['pip','install','-r','requirements.txt'])
    else:
        call(['pip','install','--user','-r','requirements.txt'])

print "Installation complete!"
print "To use GPU with Theano use 'THEANO_FLAGS='device=gpu0' python semisupervised_lasagne.py'"
print "OR see Theano documentation for using .theanorc"
