| Testing | Coverage |
| :-----: | :------: | 
| [![Build Status](https://travis-ci.org/djp42/IntentionPrediction.svg?branch=SpringCleaning2019)](https://travis-ci.org/djp42/IntentionPrediction) | [![Coverage Status](https://coveralls.io/repos/github/djp42/IntentionPrediction/badge.svg?branch=SpringCleaning2019)](https://coveralls.io/github/djp42/IntentionPrediction?branch=SpringCleaning2019) |

# IntentionPrediction
This repo is used to predict the intentions of human drivers as the approach intersections. 
It is the product of numerous smaller projects, and as such may not be the most interpretable. 
As I (djp42) have time I will continue to clean it up.

## Project History 
This work started as a class project for CS 229 at Stanford University in Spring of 2016. 
I worked with [alin719](https://github.com/alin719) at that time, and that is when much of the data processing was done, although the overall goal was much different.

That summer I did an REU (research experience as an undergraduate) with the Stanford Intelligent Systems Laboratory (SISL) under the mentorship of Mykel Kochenderfer and continued the work, pivotting away from the original goal and focusing on the high level intention prediction.

In the Fall of that year (2016) I further continued the project through the class project for CS 221.
It was here that I made much of the method of experimentation and analysis better, through the changes to cross validation, etc. 

That winter and spring I worked with [Tim Wheeler](http://timallanwheeler.com/index.html) to perform significant revisions to the paper and generally clarify everything, continuing under the supervision of Mykel Kochenderfer at SISL.
We had the honor of presenting the results at the IEEE Intelligent Vehicles Symposium (IVS) 2017, and this code is what was used in that paper:
- Generalizable Intention Prediction of Human Drivers at Intersections - [https://ieeexplore.ieee.org/document/7995948/](https://ieeexplore.ieee.org/document/7995948/).

## Code Description
This repository consists of a number of parts: Data Processing - Model Execution - Evaluation. Some files are specific to only one part, while others are used throughout.
We use the NGSIM dataset, specifically the data for the urban roads Lankershim Blvd. and Peachtree St. 

I put a lot of effort into making it easy to run stuff, because from previous experience on this project it can get really messy really quick. 

However, its still not trivial, and you should familiarize yourself program.py and BayesNet.jl to make sure I didn't screw anything up :P

The project addresses the challenge of predicting human driver intentions on the approach to intersections from up to 600 feet away. 

See this pre-release [https://github.com/djp42/IntentionPrediction/releases/tag/v0.1](https://github.com/djp42/IntentionPrediction/releases/tag/v0.1) for the required data. This is further explained in the setup instructions.

**See [utils/argument_utils.py](utils/argument_utils.py) for the command line arguments to pass into `program`, or run `python program.py -h`**

## Install
There actually aren't that many dependencies for the python portion. There are just a few required packages for julia as well. See below.
- Python3.5
- `pip install -U -r requirements.txt`
- `pip install -U -r requirements_test.txt`

### Julia Install
1. Install the `julia` programming language using your system default or direct download from [https://julialang.org/downloads/](https://julialang.org/downloads/)
2. Add the `julia` requirements. Open julia interpreter and use the package manager:
    ```
    julia
        using Pkg
        Pkg.add("BayesNets")
        Pkg.add("Discretizers")
        Pkg.add("JLD2")
        Pkg.add("DelimitedFiles")
    )
    ```

## Setup
1. Download data and set paths.
    - `./build.sh`
        - downloads data from [https://github.com/djp42/IntentionPrediction/releases/download/v0.1/data.tar.gz](https://github.com/djp42/IntentionPrediction/releases/download/v0.1/data.tar.gz) and extracts to [res/](res/)
        - sets `INTENTPRED_PATH` to the current directory if it has not been set before.
2. (optional) Process data.
    - `python program.py combine`
    - `python program.py augment --filenames trajectories-lankershim.txt trajectories-peachtree.txt`
    - Optional because you can use the included augmented data that is already processed.
    - There are 3 stages of the data:
        1. *raw* - the raw trajectories from the original NGSIM format: `trajectories-[start]-[end].txt`.
        2. *combined* - we combine the raw trajectories of subsequent time periods to create a single set of data: `trajectories-[roadname].txt`
        3. *augmented* - Adding information such as velocity, acceleration, to the data: `AUGv2_trajectories-[roadname].txt`
    - These steps just about 10 minutes.
3. Create feature-ized data.
    - `python program.py featurize --featurize_type i --test_nums [test_nums]`
        - `[test_nums]` are the "000", "001", etc., which indicate which features to use.
    - featurize_type `s` is basically deprecated. option `i` means we save features by intersection, which is more efficient and useful.
    - This can take a while for the robust feature sets including neighbors and history, up to over a half hour for test features 111.



## Execution
The file "program.py" is where nearly everything is executed from. 
I will admit to have made this file before discovering the beauty of python argparse, so apologies in advance that I did everything by hand...

After the setup from the previous section, we can produce results.

* Train models and produce results. 
    - `program.py train --models [models] --test_nums [test_nums] --test_intersections [test_intersections]`
        - additional arguments specified in program.py. This trains and tests the specified models. 
    - As you may expect, this can take a long time for the neural nets. It took me around 6 hours.
* Evaluate the results from previous step.
    - `program.py evaluate --models [models] --test_nums [test_nums] --test_intersections [test_intersections] [... other flags]`
        - additional arguments specified in program.py. 
* Train and test the discrete bayesnet
    - `julia BayesNet.jl`       
    - uses the same features created, but trains and tests the discrete bayesian network.
* Analyze (deprecated)
    - `python analysis.py`  
        - to analyze the models.
        - Currently unsupported on all cases
    - `julia analysis.jl`   
       - to analyze the BayesNet


## Files
A breakdown of files and purposes (if not listed is either deprecated or unused or boring):

BayesNet.jl: Fits and saves output for bayesian network run on the specified test (change in code)

program.py: The "runfile" for SVM, DNN, LSTMs run on the specified test (change in code). Also where calls to functions that process data take place, this is what generates results.

validation.py: [deprecated] At one point did small tests for LSTM and others, but now mainly to call evaluation functions (like plotting, scoring)


In [utils/](utils/):
* constants: constants... (and some functions because bad style I know... srry)
* data2_class: improved class for data structure, main purpose is a dictionary of frames and vehicle ids to data, and specific functions. the functions are part of the class because they are very specific and there is only one class instance at any time anyway
* data_class: mostly deprecated, used for old stuff still, may be needed if remaking data from raw
* data_util: various utilities for helping with data manipulation at all stages
* driver_util: some more utilities that have a more specific purpose to this project
* eval_util: [deprecated] various functions and methods to evaluate the results. No result generation except thebaseline
* frame_util: Used for occasional utilties like path names, frame handling, etc. Should be revisted and cleaned up.
* signals_util: [deprecated] utilities in my attempt to use signals as features, not used in final
* util.py: [deprecated] unmodified since class project, uninteresting
* vehicleclass.py: specific functions for the vehicle class (like getLaneID)


## TODO
* Add more unittests
* Better object oriented programming throughout.
* Further clean files, removing what is unnecessary to the intention prediction. 
* `INTENTPRED_PATH` use throughout.
* Better filepaths.
* Make sure nothing is importing unnecessary modules.
* Upgrade to newer version of `tensorflow`
* Ensure saving / loading of all models works.
* Check use of `DistInd` in `score_utils.py` for confusion matrix (it does not have a test num so it may be incorrect usage for test 0...?).
* Clean up code. There is definitely a lot of redundant and messy code, that I should probably clean up at some point.
* Add comments
* Extend with other useful things

