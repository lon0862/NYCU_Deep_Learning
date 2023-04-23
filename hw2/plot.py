import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('mean.csv')
plt.plot(df)
plt.xlabel('1k episode')
plt.ylabel('score')
plt.show()