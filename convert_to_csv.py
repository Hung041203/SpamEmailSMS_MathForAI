import pandas as pd

data = pd.read_csv("SMSSpamCollection", sep='\t', header=None)

data.columns = ['label','message']

data.to_csv("spam.csv", index=False)