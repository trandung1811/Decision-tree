#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 02:50:24 2020

@author: dungtran
"""
My python files are programed to answer part 1, part 2 and part 3 (hw2_par1.py and hw2_par2.py). 
The data set used for part to is "train.csv" the first 650 rows is used for training and the rest is for testing. 
To run the program: 
 - Run directly from IDE (I code in Spider of Anacoda)
 - Run from terminal: Python3 hw2_part1.py (for Mac OS)
                      python3 hw2_part2.py (for Mac OS)
 
In part 1, as my program require recursion to check all the node of the tree, so I printed out the tree as follow: 

 root:  level 

   --> ['None']:  True
   --> ['Junior']-->['phd']-->['None']: True, ['yes'] :  False, ['no'] :  True, 
   --> ['Mid'] :  True, 
   --> ['Senior']-->['tweets']-->['None']: False, ['yes'] :  True, ['no'] :  False,
   
 The tree is the same with the output in the requirement, but in different form: 
       
in part 2, I add a "Improvement Propose" in the report as to present my ideas to improve the algorithm