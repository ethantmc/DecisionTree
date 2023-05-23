import json
import numpy as np
def main(fname):
    root = []
    with open("../data/"+fname, 'r') as f:
        v = json.load(f)
        root = v
    test = np.loadtxt('../data/test.txt')
    listOfPersons = []
    i = 0
    ##create the testing data objects
    while(i < len(test[0])): ##while I < the number of columns
        person = {'RISK':test[0][i], 'AGE':test[1][i], 'CRED_HIS':test[2][i], 'INCOME':test[3][i], 'RACE':test[4][i], 'HEALTH':test[5][i]} ##risk (label), age, credit history, income, race, health is the order of attributes
        i += 1
        listOfPersons.append(person)
    testSize = len(test[0])
    treeAccuracy = 0
    for i in listOfPersons:
        treeAccuracy += getSingleAccuracy(root,listOfPersons.pop())
    #print("treeAccuracy: ")
    #print(treeAccuracy/testSize)
    return treeAccuracy/testSize
def getSingleAccuracy(tree,person):
    node = tree
    while type(node) == list:
        v = person[node[0]]
        if type(v) == np.float64:
            v = str(int(v))
        node = node[1][v]
    if(node == person['RISK']):
        return 1
    else:
        return 0
