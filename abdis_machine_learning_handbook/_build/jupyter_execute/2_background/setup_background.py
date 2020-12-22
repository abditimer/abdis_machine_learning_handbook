#!/usr/bin/env python
# coding: utf-8

# # Setup Background
# In this section, we will cover:
# - different types of python environments
# - how to set up your environment to build models
# 
# ## Intro
# Setting up environments for python can be confusing:
# 
# <img src='https://imgs.xkcd.com/comics/python_environment_2x.png' alt='meme of python dependencies all over the place' width='500'>
# 
# In this notebook, we will aim to help summarise the different approaches, and help you get on your feet quickly!

# ## Environments
# Environments allow you to isolate away from your machine, hence, giving you a way to build isolated project with different package dependencies. 
# 
# ### Conda
# - Part of the Anaconda Distribution, you can use conda to create virtual environments and install packages
# - built for data science and other data tasks
# - it is a package and environment manager
# - You can create a new environment per project
# - create environment using `conda create <name>`
# - activate and enter it using: `source activate <name>` (for windows, just use activate)
# - use `conda list` to view all packages in your virtual environment
# - use `conda install <python packages>` to install new py packages
# - to update conda, run: `conda upgrade conda` and `conda upgrade --all`
# - to update a conda package, run: `conda update <package name>` or run `conda update --all`
# - Environments:
#     - create new environment with : `conda create -n <name>`
#     - activate conda environment with : `conda activate <env name>`
#     - deactivate it with : `conda deactivate`
#     - export environment using : `conda env export > environment.yaml`
#     - list all environments : `conda env list`
# 
# 
# ### MiniConda
# - Like conda, but smaller
# - smaller distribution that only contains python and conda 
# - no preinstalled packages
# 
# ### Pip
# - default python package manager 
# - can download all packages from PyPi distribution
# - to check if you have pip, use the typical --version command with pip: `pip --version`
# - export all current packages from an environment using `pip freeze > requirements.txt`

# In[ ]:




