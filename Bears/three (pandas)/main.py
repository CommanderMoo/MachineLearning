"""
RPJ~
this is for assignment one
this is also late 2-10-21
"""

from sklearn import tree

# app information
print("\nOur app will take panda data, and then tell you")
print(" if it is a Panda or a Red Panda.")

# training data
# features
# height//weight//length
feature = [[96, 136, 154], [98, 130, 158], [92, 134, 152], [92, 138, 156], [92, 130, 150],
           [7, 38, 9], [7, 30, 3], [5, 7, 3], [7, 28, 5], [5, 58, 11]]

# labels
# panda=1
# reds=0
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# this is the training
clf = tree.DecisionTreeClassifier()
clf = clf.fit(feature, labels)

# use our Ai Ml app to predict what type of panda it will be
# print("\n")
print("Today we will be going through the jungle to locate some wild animals.")
print("\nThe animals we will be tracking today will be: ")
print(" (0) is a red pandas")
print(" or ")
print(" (1) is a pandas ")
print("Try to guess the data with the computer. Happy Hunting.\n\n")

print("----------")
print("This is some tall grass, it almost looks like wood.")
print("\n Panda \n")
print(clf.predict([[97, 130, 156]]))
print("----------")
print("The forest is thick, watch out for mushrooms and bugs.")
print("\n Panda \n")
print(clf.predict([[97, 130, 156]]))
print("----------")
print("This is some nice grass, not to big not too small.")
print("\n Red panda \n")
print(clf.predict([[5, 58, 11]]))
print("----------")
print("Imagine eating a pandacake!!")
print("\n Panda \n")
print(clf.predict([[97, 130, 156]]))
print("----------")
print("We are not big and we enjoy fruits and grubs.")
print("\n Red panda \n")
print(clf.predict([[5, 58, 11]]))
print("----------")
print("I dont know kung fu but my belly is still big!")
print("\n Panda \n")
print(clf.predict([[97, 130, 156]]))
print("----------")
print("Watch your plants, I will eat small flowers and grasses aswell.")
print("\n Red panda \n")
print(clf.predict([[5, 58, 11]]))
print("----------")
print("You'll need more then you'd expect to fill me up.")
print("\n Panda \n")
print(clf.predict([[97, 130, 156]]))
print("----------")
print("Insects and grubs will fill me up.")
print("\n Red panda \n")
print(clf.predict([[5, 30, 3]]))
print("----------")
print("I am small, red, and most surely a panda.")
print("\n Red panda \n")
print(clf.predict([[5, 7, 3]]))
print("----------")
print("References: Chinahighlights.com")


# this is for pandas one