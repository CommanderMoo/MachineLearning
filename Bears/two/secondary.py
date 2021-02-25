"""
RPJ
this is for assignment two
this is also late 2-10-21
"""

from sklearn import tree
import pandas as pd

df = pd.read_csv(r"allPandas.csv")


# let the user know what our app is about
print("our app will take radar data from WWII, and then tell you")
print("if it is a bomber ot a fighter")
# training data
# features wing span, fuselage length in order

feature = df[["height", "weight", "length"]]
# labels bomber=1, Fighter=0
labels = df[["name"]]

# this is the training
clf = tree.DecisionTreeClassifier()
clf = clf.fit(feature, labels)

# alt click = multi type
# use our Ai Ml app to predict what type of plane it will be
print("\n")
print("Welcome back Hunter! Once again we are on a wild hunt for the wild creatures of today.")
print(" We shall hunt pandas this day but I cant tell which are giants and which are red.")
print(" Be my aid this day and we shall explore the wild around.")
print(" 0 is a Red Pandas and 1 is a Giant Pandas!\n")
# H W L
print("-" * 50)
print("This is a 138lb bear. This is a Giant "+(clf.predict([[92, 138, 152]])))
print("This is a 156ft white spotted beast. This is a Giant "+clf.predict([[94, 138, 156]]))
print("-" * 50)
print("This is a 132lb totally not blackbear but instead a Giant "+clf.predict([[96, 132, 154]]))
print("This is a 136lb ball of cuddling bamboo. This is a Giant "+clf.predict([[90, 136, 156]]))
print("This is a Giant "+clf.predict([[98, 130, 158]])+". One of the most known bears on the planet.")

print("-" * 50)
print("-" * 50)

print("This is a 62lb bear. This is a Red "+clf.predict([[5, 62, 5]]))
print("This is a 7ft tall red blur beast. This is a Red "+clf.predict([[7, 32, 7]]))
print("-" * 50)
print("This is a 58lb totally dangerous Red "+clf.predict([[9, 58, 11]]))
print("This is a 11ft, totally dangerous Red "+clf.predict([[9, 58, 11]]))
print("This is a Red "+clf.predict([[5, 54, 5]])+". A species that is actually new to me.")
print("-" * 50)

# pandas two
