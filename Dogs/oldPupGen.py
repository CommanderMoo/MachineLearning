'''
in this file i will create my own dataset.
----
this dataset is going to be designed to give me
the possible sizes of a bulldog going from a to
---
be determined
age // weight
'''

import random
import csv

# things for creating lines of arrays
driver = 50
timeStep = 0
# things for creating a new excel file
ideaOne = "OlderDogs"
format = ".csv"

# my data and sources
minAge = 4
maxAge = minAge + 4
# minHeight = 12
# maxHeight = minHeight + 4
minWeight = 5
maxWeight = minWeight + 20

field = ["Age", "Weight", "index"]
outputList = ["age", "weight"]
# tempList = []

if __name__ == "__main__":

    while driver > timeStep:
        age = random.randrange(minAge, maxAge, 2)
        weight = random.randrange(minWeight, maxWeight, 2)

        # print variables
        tempList = [age, weight, "old"]
        outputList.append(tempList)
        print(tempList)

        outputList.append(tempList)
        timeStep += 1
        print(timeStep)

    with open(ideaOne + format, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(field)
        write.writerows(outputList)