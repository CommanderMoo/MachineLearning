import pandas as pd
from sklearn import tree
import numpy as np


# file defined
df = pd.read_csv(r"creatingGen.csv")
print("Import Successful...")
print("\n\t ** Read the File **\n\n")
print(df)

# first we enter data (tranning data)
dogFeatures = df[["age", "weight"]]
dogLabels = df[["label_health"]]
# dogFeatures = [[,],[,],[,],[,],[,],[,]]
# dogLabels = [1, 1, 1, 1, 0, 0, 0]

"""
select only the columns I want to use as features from the data frame
"""

clf = tree.DecisionTreeClassifier()
clf = clf.fit(dogFeatures, dogLabels)