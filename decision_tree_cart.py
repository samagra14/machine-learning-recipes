from __future__ import print_function

training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

header = ["color", "diameter", "label"]


def isNumeric(value):
    return isinstance(value,int) or isinstance(value,float)

def classCounts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        else:
            counts[label]+=1
    return counts

class Question:
    def __init__(self,column,value):
        self.column = column
        self.value = value

    def match(self,example):
        val = example[self.column]
        if isNumeric(val):
            return val >= self.value
        else:
            return val == self.value
    def __repr__(self):
        condition = "=="
        if isNumeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


def partition(rows,question):
    trueRows, falseRows = [],[]
    for row in rows:
        if question.match(row):
            trueRows.append(row)
        else:
            falseRows.append(row)
    return trueRows,falseRows

def gini(rows):
    counts = classCounts(rows)
    impurity = 1
    for label in counts:
        probabilityLabel = counts[label]/float(len(rows))
        impurity -=probabilityLabel**2
    return impurity

def informationGain(left,right,currentUncertainty):
    p = float(len(left))/(len(left)+len(right))
    return currentUncertainty-p*gini(left)-(1-p)*gini(right)

def findBestSplit(rows):
    numberOfFeatures = len(rows[0])-1
    bestGain = 0
    bestQuestion = None
    currentUncertainty = gini(rows)
    for col in range(numberOfFeatures):
            values = set([row[col] for row in rows])
            for value in values:
                question = Question(col,value)
                trueRows , falseRows = partition(rows,question)
                if len(trueRows) == 0 or len(falseRows) == 0:
                    continue
                gain = informationGain(trueRows,falseRows,currentUncertainty)
                if gain >= bestGain:
                    bestGain = gain
                    bestQuestion = question
    return bestGain, bestQuestion

class Leaf:
    def __init__(self,rows):
        self.predictions = classCounts(rows)

class DecisionNode:
    def __init__(self, question,trueBranch, falseBranch):
        self.question = question
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch

def buildTree(rows):
    gain, question = findBestSplit(rows)
    if gain == 0:
        return Leaf(rows)
    trueRows, falseRows = partition(rows, question)
    trueBranch = buildTree(trueRows)
    falseBranch = buildTree(falseRows)
    return DecisionNode(question,trueBranch,falseBranch)

def printTree(node, spacing = " "):
    if isinstance(node,Leaf):
        print(spacing + 'Predict: ', node.predictions)
        return
    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    printTree(node.trueBranch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    printTree(node.falseBranch, spacing + "  ")

myTree = buildTree(training_data)
printTree(myTree)
