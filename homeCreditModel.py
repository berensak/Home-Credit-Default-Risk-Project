# HOME CREDIT DEFAULT RISK PROJECT

"""
'''
In this project we try to predict home credit default risk for clients.
In this script we focus on modeling.

Dataset: https://www.kaggle.com/c/home-credit-default-risk/overview

Steps to follow for modeling:
    -

'''
"""

# Load dependencies
import pandas as pd
import fastparquet
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import pickle

from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.simplefilter(action="ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Load dataset
df = pd.read_parquet("data_preprocessed.parquet")
df.head()

