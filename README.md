# Nonlinear Time Series Prediction, ARIMA vs. Neural Networks  

Repository for WS 20/21 seminar thesis. Uploaded for demonstrative purposes.  

This thesis aims to illustrate the concept of nonlinearity in time series and to compare the forecasting capabilities of neural networks and traditional arima models for nonlinear time series.


## Directory structure  

In general it is recommended to keep the given folder structure. All scripts have their path as working directory and reference paths relative to that. No absolute paths must be given.

## How to reproduce the results:  

The results were obtained by running Python 3.7.4 and using all packages in their respective versions stated in /code/environment/requirements.txt

Note: the results might change when different versions of Python or the packages are used. At the time of creation of this repository, the package used pmdarima is incopatible with python
versions later than 3.7

The results can be reproduced easily by setting up a virtual environment in /code/environment


### How to set up a virtual environment in the command line:  

1. navigate to /code/environment
2. set up the environment, seminar_env is an arbitrary environment name
```
python -m venv seminar_env
```
3. activate the environment (Windows)
```
seminar_env\Scripts\activate.bat
```
(Linux/Mac)
```
source seminar_env/bin/activate
```
4. install all required packages
```
pip install -r requirements.txt
```
python scripts can now be run within the command line using the virtual environment, to make the virtual environment available in an IDE, 
additional steps must be taken depending on which IDE is used.

## Python scripts
### nnplots.py  

Code to create plots of activation functions and neural network structure. Plots are stored in manuscript/src/latexGraphics.  

Code for neural network structure adapted from https://gist.github.com/anbrjohn/7116fa0b59248375cd0c0371d6107a59. 

### empiricalanalysis.py  

Note: the script uses multiprocessing for GARCH model optimization. Multiprocessing can be disabled in the 'control panel'

Code used for the empirical analysis in chapter 5. Outputs are :
* manuscript/src/latexChapters/Variables.tex, which defines results from code
to use in latex. 
* manuscript/src/latexTables/ArimaResults.tex, output of ARIMA model
* manuscript/src/latexTables/BDSResults.tex, output of BDS Test

### benchmark.py  

Note: the script uses multiprocessing for MLP model optimization. Multiprocessing can be disabled in the 'control panel'. The script takes quite
a long time to run. ~120 minutes with 36 cores and 64 GB RAM.

Code used for forecast comparison in chapter 6. Outputs are:
* .pkl files in /code/results

### getresults.py  

Code used to retrieve results from benchmark.py script. Outputs are:
* plots for forecast results figures
* manuscript/src/latexTables/Specifications.tex
* manuscript/src/latexTables/MSE.tex

## license & copyright  

&copy; Till Ehrich, if not stated otherwise
