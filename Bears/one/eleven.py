'''
 pip install pandas
 pip install sklearn
 pip install openpyxl
'''

# lets do our imports of libs
# from math import e
import pandas as pd
from sklearn import tree
# import openpyxl
import xlrd


def display_prediction_result(numeric_result):
    if numeric_result[0] == 0:
        print(" You have a cat")
    else:
        print(" You have a dog")


# define the file to import

# book = xlrd.open_workbook("MLCDTCatDog.xlsx")
# read the excel file into the data frame
df = pd.read_excel(r"MLCDTCatDog.csv")

print("This has something to do with your import")

# select only the columns I want to use
features = df[["feature_height_in", "feature_weight_lbs"]]
labels = df[["label_cat_or_dog"]]

"""
select only the columns I want to use as features from the data frame
"""

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# -----------------------------------------------------------------
# hard code--
print("\n0 = Cat, 1 = Dog ")
print("This is the cat example ")
print("features_height_in = 9.1, features_weight_lbs = 7.9")
# prediction area
numeric_result = clf.predict([[9.1, 7.9]])
# show prediction area
display_prediction_result(numeric_result)

# hard code--
print("\n0 = Cat, 1 = Dog ")
print("This is the cat example ")
print("features_height_in = 9.45, features_weight_lbs = 8.7")
# prediction area
numeric_result = clf.predict([[9.45, 8.7]])
# show prediction area
display_prediction_result(numeric_result)

# hard code--
print("\n0 = Cat, 1 = Dog ")
print("This is the dog example ")
print("features_height_in = 30, features_weight_lbs = 35")
# prediction area
numeric_result = clf.predict([[30, 35]])
# show prediction area
display_prediction_result(numeric_result)

# hard code--
print("\n0 = Cat, 1 = Dog ")
print("This is the dog example ")
print("features_height_in = 44, features_weight_lbs = 80")
# prediction area
numeric_result = clf.predict([[44, 80]])
# show prediction area
display_prediction_result(numeric_result)
print("\n")
# -----------------------------------------------------------------

print("*" * 50)
print("\t *** Data training turning on ***")
print("*" * 50)

print("\n\t *** Time for user input ***")
# get user information
pet_height_in = input("What is your pets height in inches?")
pet_weight_lbs = input("What is your pets weight in pounds?")
numeric_result = clf.predict([[pet_height_in, pet_weight_lbs]])
display_prediction_result(numeric_result)

print("\n\n\t *** End of Simulation ***")
print("*" * 50)



'''
def prediction(result):
    if result[0] == 0:
        if result == 1:
            print("hi")
    else:
        print("no")

'''