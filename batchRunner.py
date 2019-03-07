#!/usr/bin/python
import os, sys
import subprocess as sp
import time

files = ['pic1.jpg', 'pic2.jpg', 'pic3.jpg', 'pic4.jpg']


if __name__ == "__main__":

    for file in files:

        # argv = ['python','runner.py','-f',file]
        argv = ['python','runnerGreyScale.py','-f',file]

        sp.Popen(argv)
