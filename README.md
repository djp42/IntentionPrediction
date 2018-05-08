UPDATE 12/15 -- For the purposes of the CS221 project

This repository consists of a number of parts: Data Processing - Model Execution - Evaluation. Some files are specific to only one part, while others are used throughout. 

PIPELINE TO RUN STUFF

The file "program.py" is where nearly everything is executed from. 

Once the data is downloaded and the associated constant in constants.py is changed (or you put it in res/Lankershim, res/Peachtree), these command "should" work to run basically what I ran

      - The data needed is just the raw trajectory files for NGSIM Lankershim and Peachtree
      
python[3] program.py c  ---- to combine the two datasets, if not already (included in data.zip are already combined)

python[3] program.py a  ---- to augment the raw trajectory files

python[3] program.py f  ---- creates the features that will be used as Train/Test data

python[3] program.py t  [...] ---- additional arguments specified in program.py. This trains and tests the specified models. 

julia BayesNet.jl       ---- uses the same features created, but trains and tests the discrete bayesian network

python[3] program.py e [...] ---- to evaluate the outputs

python[3] analysis.py  ---- to analyze the models

julia analysis.jl      ---- to analyze the BayesNet


I put a lot of effort into making it easy to run stuff, because from previous experience on this project it can get really messy really quick. 

However, its still not trivial, and you should familiarize yourself program.py and BayesNet.jl to make sure I didn't screw anything up :P

Below is a blurb that basically says this code is poorly written, but I have tried to clean it up at least a little. 
Below the blurb is a breakdown of the files, which I have not gone through to change, but can't bring myself to delete. Most of it stands, with the major changes happening in program.py, lib/data_util.py, and the creation of lib/score_util.py because the other one was a mess.

Also the creation of analysis.[py/jl]

The new system runs the same models as before, but I fixed a number of bugs (>.>) and it now actually makes sense in that it tests based on intersection. This is very important for significance to the research community and stuff. 


======================================================================
This is the repository for the work I did this past summer working on predicting human driver intentions on the approach to intersections from up to 600 feet away. 

This repository is not self sufficient because the raw data files are not included due to size limitations, but can be obtained for free from the NGSIM site.

Disclaimer: this code is not the cleanest I can make because its purpose is "research code" to get things to work and produce meaningful results. Also some of the commit messages are good, but sometimes I slacked on them, especially towards the beginning. Sorry in advance.

Additionally, there is a lot of redundant and unused code from the 229 project, since I began by cloning that and have not gone through and removed it yet. 



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
  
