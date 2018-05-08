# IntentionPrediction
### This repo is used to predict the intentions of human drivers as the approach intersections. 
### It is the product of numerous smaller projects, and as such may not be the most interpretable. 
### As I have time I will continue to clean it up.

## Project History 
This work started as a class project for CS 229 at Stanford University in Spring of 2016. 
I worked with alin719 at that time, and that is when much of the data processing was done, although the overall goal was much different.

That summer I did an REU (research experience as an undergraduate) with the Stanford Intelligent Systems Laboratory (SISL) under the mentorship of Mykel Kochenderfer and continued the work, pivotting away from the original goal and focusing on the high level intention prediction.

In the Fall of that year (2016) I further continued the project through the class project for CS 221.
It was here that I made much of the method of experimentation and analysis better, through the changes to cross validation, etc. 

That winter and spring I worked with Tim Wheeler to perform significant revisions to the paper and generally clarify everything, continuing under the supervision of Mykel Kochenderfer at SISL.
We had the honor of presenting the results at the IEEE Intelligent Vehicles Symposium (IVS) 2017, and this code is what was used in that paper:
    Generalizable Intention Prediction of Human Drivers at Intersections (https://ieeexplore.ieee.org/document/7995948/).

## Code Description
This repository consists of a number of parts: Data Processing - Model Execution - Evaluation. Some files are specific to only one part, while others are used throughout.
We use the NGSIM dataset, specifically the data for the urban roads Lankershim Blvd. and Peachtree St. 

I put a lot of effort into making it easy to run stuff, because from previous experience on this project it can get really messy really quick. 

However, its still not trivial, and you should familiarize yourself program.py and BayesNet.jl to make sure I didn't screw anything up :P

This is the repository for the work I did this past summer working on predicting human driver intentions on the approach to intersections from up to 600 feet away. 

This repository is not self sufficient because the raw data files are not included due to size limitations, but can be obtained for free from the NGSIM site.

Additionally, there is a lot of redundant and unused code from the 229 project, since I began by cloning that and have not gone through and removed it yet. 

## Pipeline
The file "program.py" is where nearly everything is executed from. 
I will admit to have made this file before discovering the beauty of python argparse, so apologies in advance that I did everything by hand...

Once the data is downloaded and the associated constant in constants.py is changed (or you put it in res/Lankershim, res/Peachtree), these command "should" work to run basically what I ran

      - The data needed is just the raw trajectory files for NGSIM Lankershim and Peachtree
      
python program.py c  
    - to combine the two datasets, if not already (included in data.zip are already combined)

python program.py a  
    - to augment the raw trajectory files

python program.py f  
    - creates the features that will be used as Train/Test data

python program.py t  [...] 
    - additional arguments specified in program.py. This trains and tests the specified models. 

julia BayesNet.jl       
    - uses the same features created, but trains and tests the discrete bayesian network

python program.py e [...] 
    - to evaluate the outputs

python analysis.py  
    - to analyze the models

julia analysis.jl      
    - to analyze the BayesNet


## Files
A breakdown of files and purposes (if not listed is either deprecated or unused or boring):

BayesNet.jl: Fits and saves output for bayesian network run on the specified test (change in code)

program.py: The "runfile" for SVM, DNN, LSTMs run on the specified test (change in code). Also where calls to functions that process data take place, this is what generates results.

validation.py: At one point did small tests for LSTM and others, but now mainly to call evaluation functions (like plotting, scoring)


In lib:

  constants: constants... (and some functions because bad style I know... srry)
  
  data2_class: improved class for data structure, main purpose is a dictionary of frames and vehicle ids to data, and specific functions. the functions are part of the class because they are very specific and there is only one class instance at any time anyway
  
  data_class: mostly deprecated, used for old stuff still, may be needed if remaking data from raw
  
  data_util: various utilities for helping with data manipulation at all stages
  
  driver_util: some more utilities that have a more specific purpose to this project
  
  eval_util: various functions and methods to evaluate the results. No result generation except the baseline
  
  frame_utile: mostly deprecated at this point, don't even remember it.
  
  goal_processing: goal-specific data processing
  
  goal_util: goal-specific utilities
  
  learn_util: mostly deprecated/from the 229 project not the research project
  
  merger_methods: not applicable to research project - is methods to help identify and quantify merging vehicles
  
  signals_util: utilities in my attempt to use signals as features, not used in final
  
  util.py: unmodified since class project, uninteresting
  
  vehicleclass.py: specific functions for the vehicle class (like getLaneID)
 
 
 In bin:
 
   almost entirely useless now, but setup still helps on start, and visualize can sometimes be usefule.
 
## TODO
(This is mostly for me, but also if you want to work on it!)
    1. Go through and verify the setup process again. I am sure things have been deprecated and probably don't work as intended
    2. Clean up code. There is definitely a lot of redundant and messy code, that I should probably clean up at some point.
    3. Extend with other useful things

