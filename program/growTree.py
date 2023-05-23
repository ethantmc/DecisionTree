import numpy as np
import json
import math

def main():
    with open('../data/dataDesc.txt') as f:
        domain = json.load(f)
    ##load the data
    train = np.loadtxt('../data/train.txt')
    #person = [[["RISK", train[0][0]], ["AGE", train[1][0]], ["CRED_HIS", train[2][0]], ["INCOME", train[3][0]], ["RACE", train[4][0]], ["HEALTH", train[5][0]]]]
    listOfPersons = []
    i = 0
    while(i < len(train[0])):
        person = [train[0][i], train[1][i], train[2][i], train[3][i], train[4][i], train[5][i]] ##risk (label), age, credit history, income, race, health is the order of attributes
        i += 1
        listOfPersons.append(person)
    
    ##calculate set entropy
    riskNo = 0
    riskYes = 0
    for i in listOfPersons: 
        if(i[0] == 2):
            riskYes += 1
        elif(i[0] == 1):
            riskNo += 1
    ##save the fractions because python's math module didn't like it
    lowRiskFraction = (riskNo/(riskYes+riskNo))
    highRiskFraction = (riskYes/(riskYes+riskNo))
    setEntropy = (-lowRiskFraction*(math.log(lowRiskFraction,2)))-(highRiskFraction*(math.log(highRiskFraction,2)))

    ##select the root node- attribute with the most gain
    i2 = 1 ##start at 1 as 0 is class label and should not be used.
    bestGainSoFar = 0
    bestGainAttributeIndex = 0
    while(i2 < len(listOfPersons[0])): ##iterate for each attribute
        current = calculateGain(listOfPersons,i2,setEntropy,domain[i2][1])
        if(current > bestGainSoFar):
            bestGainSoFar = current
            bestGainAttributeIndex = i2
        i2+=1 
    root = [bestGainAttributeIndex,{}] ##construct tree

    for i in domain[bestGainAttributeIndex][1]: ##create the keys for the root node, and their lists
        root[1][i] = None##i is the key in data description, root[1] is the dictionary.
    for i in domain[root[0]][1]: ##For every key in data description's dictionary, i = numerical value of the key. Complicated line, but basically uses the class label to see what the dictionary looks like, and i is keys in the dictionary, which just so happen to be numbers.
        key = i
        root[1][i] = calculateReducedList(listOfPersons,key,root[0])##attach reduced lists for each child node
    for i in root[1]: ##recursive function call to create all children from these child nodes
        root[1][i]=pickBestAtNode(root[1][i],domain)
    ##so turns the specifications say integer encoding is fine, but that doesn't actually mean for the class labels, just the attribute labels. And now i've built the whole program.
    ##so relabel root[0]
    root[0] = domain[root[0]][0]
    treeFilePath = '../data/decisionTree.txt'
    with open(treeFilePath, 'w') as f:
        json.dump(root, f)
    return treeFilePath
def calculateReducedList(data,attributeValue,attributeIndex):
    reducedList = []
    for i in data:
        if(i[attributeIndex] == attributeValue):
            reducedList.append(i)
    return reducedList

def pickBestAtNode(reducedListOfPersons,domain):
    ##do all calculations related to gain using the list of values at a given node, and then generate the child nodes.
    i2 = 1 ##start at 1 as 0 is class label RISK

    bestGainSoFar = 0 ##same loop as in main, but it's recursive from here.
    bestGainAttributeIndex = 0
    setEntropy = calculateSetEntropyAtNode(reducedListOfPersons)
    if(setEntropy == 0):
        ##this, combined with the check that we'd be sorting based on Risk, catches all instances of incorrect expansion. Could it be cleaner? yes. Would that be a multi day refactor? also yes.
        riskYes = 0
        riskNo = 0
        for i in reducedListOfPersons: 
            if(i[0] == 2):
                riskYes += 1
            elif(i[0] == 1):
                riskNo += 1
        if(riskYes <= riskNo):##if high risk is less or as common as low risk, classify instances as low risk
            return 1
        elif(riskNo < riskYes): ##otherwise, instances are classified as high risk.
            return 2
    else:    
        while(i2 < len(reducedListOfPersons[0])): ##iterate for each attribute
            current = calculateGain(reducedListOfPersons,i2,setEntropy,domain[i2][1])
            if(current > bestGainSoFar):
                bestGainSoFar = current
                bestGainAttributeIndex = i2
            i2+=1 
        if(bestGainAttributeIndex == 0):
                riskYes = 0
                riskNo = 0
                for i in reducedListOfPersons: 
                    if(i[0] == 2):
                        riskYes += 1
                    elif(i[0] == 1):
                        riskNo += 1
                if(riskYes <= riskNo):##if high risk is less or as common as low risk, classify instances as low risk
                    return 1
                elif(riskNo < riskYes): ##otherwise, instances are classified as high risk.
                    return 2
                return "pickBestAtNode impossible error. Are you using a quantum computer?" # :)
        node = [bestGainAttributeIndex,{}]
        # altNode = [domain[bestGainAttributeIndex][0],{}]
        # print(altNode) #fixing it this way is even worse, back to repairing class labels via recursion..
        # print("altNode ^") ##I cleaned up other dev statements, but thought I'd leave this one in case you're interested.
        #create child level
        for i in domain[bestGainAttributeIndex][1]:
            node[1][i] = None
        for i in domain[node[0]][1]: ##For every key in data description's dictionary, i = numerical value of the key. Complicated line, but basically uses the class label to see what the dictionary looks like, and i is keys in the dictionary, which just so happen to be numbers.
            key = i ##Weird error was fixed by this assignment. Possibly redundant, now.
            node[1][i] = calculateReducedList(reducedListOfPersons,key,node[0])##attach reduced lists for each child node
        for i in node[1]: ##recursive function call
            node[1][i]=pickBestAtNode(node[1][i],domain)
        node[0] = domain[node[0]][0] ##convert from numeric to string for class label. It took me ages to find a way to patch this - I was trying to repair this problem with another recursive call before. It was painful.
    return node
def calculateSetEntropyAtNode(reducedListOfPersons):

    if(len(reducedListOfPersons) == 0): ##catch empty list to avoid math error
        return 0

    riskNo = 0
    riskYes = 0
    for i in reducedListOfPersons: 
        if(i[0] == 2):
            riskYes += 1
        elif(i[0] == 1):
            riskNo += 1
    if(riskNo+riskYes == 0): ##second chance to abort before math in case the list isn't empty, but is mangled/nulled out
        return 0 ##despite looking almost identical to the other one of this, it can't be replaced with a function due to this one not being in a loop.

    lowRiskFraction = (riskNo/(riskYes+riskNo))
    highRiskFraction = (riskYes/(riskYes+riskNo))
    if(lowRiskFraction == 0 or highRiskFraction == 0):
        return 0

    setEntropy = (-lowRiskFraction*(math.log(lowRiskFraction,2)))-(highRiskFraction*(math.log(highRiskFraction,2))) ##calculate entropy, finally.
    return setEntropy
def calculateGain(data,attribute,setEntropy,domain):
    ##data is the remaining set of Persons at this node
    ##attribute is the index of the attribute to calculate for
    ##entropy is set entropy for this node, which reduces calculations a bit.
    temp=0

    x = 0
    listOfSubsetEntropies = []

    while(x < len(domain)): ##iterate for the number of values in this attribute's domain - domain is passed in from the Data Desc with only the list of values, not the string.
        riskNo = 0
        riskYes = 0
        for i in data: ##tally the class label occurences for data. Remember, only those who match this far down the decision tree exist inside this function call.
            if(i[attribute] == domain[x]): ##Only tally if the attribute we're checking matches the sub-attribute we're currently tallying for.
                if(i[0] == 2):
                    riskYes += 1
                elif(i[0] == 1):
                    riskNo += 1
        if(riskNo+riskYes == 0):
            x+=1
            continue ##if the total is zero, skip this value in the domain
        #per-sub attribute occurences entropy calc (still in a loop for the domain of values)
        lowRiskFraction = (riskNo/(riskYes+riskNo))
        highRiskFraction = (riskYes/(riskYes+riskNo))
        tempTotal = riskYes+riskNo
        if(lowRiskFraction == 0 or highRiskFraction == 0):
            tempEntropy = 0
        else:
            tempEntropy = (-lowRiskFraction*(math.log(lowRiskFraction,2)))-(highRiskFraction*(math.log(highRiskFraction,2)))
        listOfSubsetEntropies.append([tempTotal,tempEntropy]) ## to be called  for calc as (subsetEntropies[i][0]/len(data))*subsetEntropies[i][1] + subsetEntropies[i+1][0].... etc
        x+=1
    categoryEntropy = calculateSubsetEntropy(listOfSubsetEntropies,len(data))
    gain = setEntropy-categoryEntropy
    return gain
def calculateSubsetEntropy(listOfSubsetEntropies,setSize):
    categoryEntropy = 0
    for i in listOfSubsetEntropies:
        categoryEntropy += ((i[0]/setSize)*i[1])
    return categoryEntropy
if __name__== '__main__':
    main()