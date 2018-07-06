import numpy as np
import random
import matplotlib.pyplot as plt

#Empty lists for training and testing data
listA, listB, listC, listD = [], [], [], []

class Neural_Network(object):
    def __init__(self):        
        #Define number of neurons per layer
        self.inputLayerSize = 2
        self.hiddenLayer1Size = 5
        self.hiddenLayer2Size = 5
        self.outputLayerSize = 1

        self.learningRate = .7
        
        #Weight matricies
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayer1Size)
        self.W2 = np.random.randn(self.hiddenLayer1Size,self.hiddenLayer2Size)
        self.W3 = np.random.randn(self.hiddenLayer2Size,self.outputLayerSize)

        
    def forward(self, X):
        #Propagate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        self.a3 = self.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.W3)
        yHat = self.sigmoid(self.z4)
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
       #Compute derivative with respect to W, W2 and W3 for a given X and y:
        self.yHat = self.forward(X)

        delta4 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z4))
        dJdW3 = np.dot(self.a3.T, delta4)
        
        delta3 = np.dot(delta4, self.W3.T) * self.sigmoidPrime(self.z3)
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2, dJdW3

    def train(self, iterations, X, Y):
        for i in range(iterations):
            dJdW1, dJdW2, dJdW3 = self.costFunctionPrime(X, Y)
            self.W1 -= self.learningRate * dJdW1
            self.W2 -= self.learningRate * dJdW2
            self.W3 -= self.learningRate * dJdW3

def getAverage(x, y):
    return (x + y) / 2

def generateData():
    #Create random decimal pairs for training input,
    #average those numbers for the output.
    for i in range(200):
        a = round(random.random(), 3)
        b = round(random.random(), 3)
        listA.append([a, b])
        listB.append([getAverage(a, b)])

        c = round(random.random(), 3)
        d = round(random.random(), 3)
        listC.append([c, d])
        listD.append([getAverage(c, d)])

def runNetworkAndPrintData(network, numpyInputs, numpyOutputs, stdInputs, stdOutputs):
    cost = network.costFunction(numpyInputs, numpyOutputs)
    yHat = list(network.yHat)
    yHatList = []

    #
    for hat in yHat:
        for miniHat in hat:
            yHatList.append([miniHat])
    
    
    for i in range(len(stdOutputs)):
        if i % 50 == 0:
            updatedCost = list(network.costFunction(numpyInputs, numpyOutputs))
            print('Sample:', i + 50, '\nInputs:', stdInputs[i][0], ',', stdInputs[i][1] , '\nGuess: ', round(yHatList[i][0], 3),
                  '\nCorrect Answer: ', round(stdOutputs[i][0], 2), '\n% Error:',
                  round(abs(yHatList[i][0] - stdOutputs[i][0]) / stdOutputs[i][0], 2), '\n\n')

def averageError(network, y):
    yAxis = []
    yHat = list(network.yHat)
    for a, b in zip(yHat, y):
        for c, d in zip(a, b):
            yAxis.append(abs(c - d) / b)

    return yAxis
            
            

def graph(yAxis):
    xAxis = [x for x in range(200)]
    plt.plot(xAxis, yAxis, 'bo')
    plt.axis([0, 200, 0, 1])
    plt.ylabel('Percent Error (%)')
    plt.xlabel('Training Iteration')
    plt.show()
    
    

def main():
    
    generateData()
    
    #Convert data sets to NumPy Arrays
    trainingSetInput = np.array(listA)
    trainingSetOutput = np.array(listB)
    testingSetInput = np.array(listC)
    testingSetOutput = np.array(listD)

    NN = Neural_Network()

    print('Forward propagation of training data before training:\n')
    runNetworkAndPrintData(NN, trainingSetInput, trainingSetOutput, listA, listB)

    #Get cost function of untrained model
    cost = NN.costFunction(trainingSetInput, trainingSetOutput)
    print('\nCost function before training: ', cost, '\n')

    yAxis = averageError(NN, listD)
    graph(yAxis)
    
    print('\nTraining...\n')

    NN.train(10000, trainingSetInput, trainingSetOutput)

    yAxis = averageError(NN, listD)
    graph(yAxis)

    print('Forward propagation of training data after training:\n')
    runNetworkAndPrintData(NN, trainingSetInput, trainingSetOutput, listA, listB)

    #Get new cost function of trained model.
    cost = NN.costFunction(trainingSetInput, trainingSetOutput)
    print('Cost function after training:', cost, '\n')

    print('Forward propagation of unseen testing data:\n')
    runNetworkAndPrintData(NN, testingSetInput, testingSetOutput, listC, listD)
    
    #Get cost function on new data set.
    newCost = NN.costFunction(testingSetInput, testingSetOutput)
    print('Cost function of new data: ', newCost)

    yAxis = averageError(NN, listD)
    graph(yAxis)



if __name__ == '__main__':
    main()
    




