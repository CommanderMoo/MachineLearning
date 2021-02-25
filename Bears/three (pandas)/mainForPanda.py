
from sklearn import tree

print("\n This program will list data from WII and declare whether an imcoming object is a fighter plane or a bomber.")
# first we enter data (tranning data)
# features - length m wingspan m height m propellers # & weight kg
# m for meters kg for kilos
print("If a plane is a fighter it will house a [1] or if it is a bomber it will house a [2].")

# US Bombers
B_seventeen = [[22.66], [31.62], [5.82], [3], [16391]]
B_twentynine = [[30.18], [43.05], [8.46], [4], [33739]]
B_Xfifthteen = [[26.70], [45.43], [7.87], [3], [17141]]

# US Fighters
P_fortyE = [[9.665], [11.3677], [3.25], [3], [2686]]
# Warhawk
P_fiftyoneD = [[9.83], [11.28], [4.08], [4], [3465]]
# Mustand
P_fortysevenD_forty = [[11.0173], [12.429], [4.472], [4], [4536]]
# Thunderbolt

# my red -------------------------
r_1 = [[7], [44], [7], "RedPanda"]
#
r_2 = [[9], [36], [10], "RedPanda"]
#
r_3 = [[9], [58], [11], "RedPanda"]
# --------------------------------

# my snow ------------------------
s_1 = [[96], [136], [154], "panda"]
#
s_2 = [[90], [138], [150], "panda"]
#
s_3 = [[94], [134], [158], "panda"]
# --------------------------------

# names
names = [B_seventeen, B_twentynine, B_Xfifthteen, P_fortyE, P_fiftyoneD, P_fortysevenD_forty]

# this is where im comparing wingspan to propellers
features = [[31.62, 3], [43.05, 4], [45.43, 3], [11.3677, 3], [11.28, 4], [11.0173, 3]]
# labels - bombers - 1 / fighters - 2
labels = [1, 1, 1, 2, 2, 2]

# do this
pandas = [[[94], [134], [158], "panda"], [[90], [138], [150], "panda"], [[96], [136], [154], "panda"],
          [[7], [44], [7], "RedPanda"], [9], [36], [], "RedPanda"], [[9], [58], [11], "RedPanda"]

bears = [1, 1, 1, 2, 2, 2]
# clf = clf.fit(features, labels)

# 

# tranning
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# use out ai ml app to predict the plane
# this is where im comparing wingspans to propellers
# features = [[31.62, 3], [43.05, 4], [45.43, 3], [11.3677, 3], [11.28, 4], [11.0173, 3]]
print("\n")
print("Keep in mind bombers are [1] and fighters are [2]! The plane listed is...")
print("\n")

# i want to over estimate the wingspan sizes and randomize the propeller with 3 propellers

print("--------------------")
print(P_fiftyoneD)
print("Above is the stats for such a plane. \nLength/ wingspan/ Height/ Propellers/ Kilograms")
print(clf.predict([[11, 3]]))
print("A fighter! This plane would be something similar to a P-51D.")
print("Prediction numbers I used 11 and 3 here.")
print("\n")
print("--------------------")
# one fighter

print("--------------------")
print(B_twentynine)
print("Above is the stats for such a plane. \nLength/ wingspan/ Height/ Propellers/ Kilograms")
print(clf.predict([[30, 4]]))
print("This should be a bomber! It looks like a B-29.")
print("Prediction numbers I used 30 and 4 here.")
print("\n")
print("--------------------")
# two bomber ----------------

print("--------------------")
print(P_fortyE)
print("Above is the stats for such a plane. \nLength/ wingspan/ Height/ Propellers/ Kilograms")
print(clf.predict([[10, 3]]))
print("A fighter! This plane would be something similar to a P-40E.")
print("Prediction numbers I used 10 and 3 here.")
print("\n")
print("--------------------")
# three fighter

print("--------------------")
print(B_seventeen)
print("Above is the stats for such a plane. \nLength/ wingspan/ Height/ Propellers/ Kilograms")
print(clf.predict([[30, 3]]))
print("A bomber! This plane would be something similar to a B-17.")
print("Prediction numbers I used 30 and 3 here.")
print("\n")
print("--------------------")
# four bomber ---------------

print("--------------------")
print(P_fortysevenD_forty)
print("Above is the stats for such a plane. \nLength/ wingspan/ Height/ Propellers/ Kilograms")
print(clf.predict([[13, 4]]))
print("A fighter! This plane would be something similar to a P-47D-40.")
print("Prediction numbers I used 13 and 4 here.")
print("\n")
print("--------------------")
# five fighter

print("--------------------")
print(B_Xfifthteen)
print("Above is the stats for such a plane. \nLength/ wingspan/ Height/ Propellers/ Kilograms")
print(clf.predict([[40, 3]]))
print("This should be a bomber! It looks like a B-X15.")
print("Prediction numbers I used 40 and 3 here.")
print("--------------------")
# six bomber


