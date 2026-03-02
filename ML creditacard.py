# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve

import tensorflow as tf
from tensorflow import keras

# Import dataset

df = pd.read_csv("creditcard.csv")
df.head()