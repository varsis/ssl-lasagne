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

print "Do you want to setup a virtual enviroment? (y/n)"
invalid = True
useenv = False
while invalid:
    answer = raw_input()
    answer.lower()

    if answer == 'y' or answer == 'yes':
        # Setup env
        invalid = False
        useenv = True
    elif answer == 'n' or answer == 'no':
        invalid = False

# Put env in env
if useenv:
    call(['virtualenv','env'])
    call(['source','env/bin/activate'])

# Next Install requirments for package
call(['pip','install','-r','requirements.txt'])

print "To enter the virtual enviroment next time use: 'source env/bin/activate'"





