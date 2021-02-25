"""
this is where all the magic will happens
---
I am comparing the weigths of old v young bulldogs
1 = young // 0 = old
--
features = age // weight
--
Special thanks to Micheal for helping me fix my csv
"""

import pandas as pd
from sklearn import tree

# this is for the user to see
def display_prediction(result):
    if result == 0:
        print("\n This is a young pup")
    else:
        print("\n This is a old pup")

# file defined
df = pd.read_csv(r"PitBull.csv")
print("Import Successful...\n\n")
print("\n\t *** Time for user input ***")

# first we enter data (tranning data)
Features = df[["age", "weight"]]
Labels = df[["index"]]

"""
select only the columns I want to use as features from the data frame
"""
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Features, Labels)
# -----------------------------
# hard code
print("\n0 = Old, 1 = Young")
print("This dog is 2 years old. ")
# print("Is the dog young or small?")
result = input("Is this dog Young or Small?")
# prediction
result = clf.predict([[2, 3]])
# showing
display_prediction(result)

# hard code
print("\n0 = Old, 1 = Young")
print("This dog is 13 lbs. ")
# print("Is the dog young or small?")
result = input("Is this dog Young or Small?")
# prediction
result = clf.predict([[7, 13]])
# showing
display_prediction(result)

# hard code
print("\n0 = Old, 1 = Young")
print("This dog is 4 years old. ")
# print("Is the dog young or small?")
result = input("Is this dog Young or Small?")
# prediction
result = clf.predict([[4, 4]])
# showing
display_prediction(result)

# hard code
print("\n0 = Old, 1 = Young")
print("This dog is 15 lbs")
# print("Is the dog young or small?")
user = input("Is this dog Young or Small?")
# prediction
result = clf.predict([[4, 15]])
# showing
display_prediction(result)
# -----------------------------
# demo area -------------------
print("*" * 50)
print("\t *** Data training turning on ***")
print("*" * 50)

# get user information
print("\n\t *** Is this a young or old dog? ***")

# get user information
Features = input("How old is your dog?")
Labels = input("How much does your dog weigh in pounds?")
result = clf.predict([[Features, Labels]])
display_prediction(result)
# -----------------------------
