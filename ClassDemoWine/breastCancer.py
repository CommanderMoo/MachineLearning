from sklearn.datasets import load_breast_cancer
from sklearn import tree
import numpy as np


# breastCancerData
bCD = load_breast_cancer()
print(bCD)

print("\n\t*** Target Names ***")
for tn in bCD.target_names:
    print(tn)

print("\n\t *** Feature Names ***")
for fn in bCD.feature_names:
    print(fn)

bCDTest = [10, 255, 568]

tranning_target = np.delete(bCD.target, bCDTest)
tranning_data = np.delete(bCD.data, bCDTest, axis=0)

test_target = bCD.target[bCDTest]
test_data = bCD.data[bCDTest]

dt_classifier = tree.DecisionTreeClassifier()

dt_classifier.fit(tranning_data, tranning_target)

print("\n\t*** Test Result ***")
print("Test Target actual labels are ", test_target)
print("Predictions by our DT labels are ", dt_classifier.predict(test_data))

# this is working 5:33 1/27/21