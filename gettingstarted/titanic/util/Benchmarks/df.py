__author__ = 'Abner'

import pandas as pd
import numpy as np

df = pd.read_csv('../Data/train.csv', header=0)

# print df.tail(3)
# print type(df)
print df['sex']