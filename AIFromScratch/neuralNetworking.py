import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

class Network:
    def __init__(self, layers_, activations):
        self.layers = []
        self.inputs = []
        self.outputs = []
        self.biases = []
        self.activations = activations
        for index, layer in enumerate(layers_):
            if index > 0:
                self.layers.append(np.random.randn(layers_[index-1], layer))
                self.biases.append(np.random.randn(1, layer))

    def saveWeights(self, filename):
        with open(filename, "w") as f:
            writeData = ""
            for layer in self.layers:
                for neuron in layer:
                    for weight in neuron:
                        if weight != neuron[len(neuron)-1]:
                            writeData += str(weight)+" "
                        else:
                            writeData += str(weight)
                    writeData += ","
                writeData += "\n"
            writeData = writeData.replace(",\n", "\n")
            f.write(writeData)

    def loadWeightsFromFile(self, filename):
        with open(filename, "r") as f:
            filedata = f.read()
            filedata = filedata.split("\n")
            for index, layer in enumerate(filedata):
                filedata[index] = layer.split(",")
                for index2, neuron in enumerate(filedata[index]):
                    filedata[index][index2] = neuron.split()
                    for index3, weight in enumerate(filedata[index][index2]):
                        filedata[index][index2][index3] = float(filedata[index][index2][index3])
            del filedata[-1]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def ReLU(self, x):
        return np.maximum(x, 0)

    def meanSquaredError(self, output, expected):
        return np.average(np.dot(np.power(np.subtract(expected, output), 2), 0.5))

    def der_meanSquaredError(self, output, expected):
        return np.expand_dims(np.subtract(expected, output), axis=0)

    def feedforward(self, input):
        layerBreak = input
        self.inputs = []
        self.outputs = []
        for index, layer in enumerate(self.layers):
            layerWBias = np.append(layer, self.biases[index], axis=0)
            self.inputs.append(np.expand_dims(np.array(layerBreak), axis=0))
            if self.activations[index] == "Sigmoid": #This network supports two activation functions
                layerBreak = self.sigmoid(np.dot(np.append(layerBreak, 1), layerWBias)) # dot product each layer input with network layer weights
            else:
                layerBreak = self.ReLU(np.dot(np.append(layerBreak, 1), layerWBias))
        self.outputs.append(np.expand_dims(np.array(layerBreak), axis=0)) # adds to network current outputs for backpropagation
        return layerBreak

    def backprop(self, output, expected, learningrate=1):
        self.errors = []
        self.deltas = []
        error = self.der_meanSquaredError(output, expected) # finding the gradients using derivative mean squared error
        for index, layer in enumerate(reversed(self.layers)):
            self.errors.append(error)
            error = np.dot(self.errors[-1], layer.T) # the error is dot producted back through the network
        self.deltas = []
        for index, layer in enumerate(self.layers): # finding the adjustment deltas using layer input and error dot-producted
            self.deltas.append(np.round(np.dot(np.dot(self.errors[-index-1].T, np.expand_dims(np.append(self.inputs[index], 1),axis=0)),learningrate),decimals=5))
        for index, layer in enumerate(self.layers): # updating the network with new deltas
            bias_update = (self.deltas[index].T)[-1]
            weight_update = self.deltas[index].T[:-1]
            self.layers[index] = np.add(self.layers[index], np.dot(weight_update, learningrate))
            self.biases[index] = np.add(self.biases[index], np.dot(bias_update, learningrate))
        return self.meanSquaredError(output, expected)

#training the network for convergence
nn = Network([2, 6, 2], activations=["Sigmoid", "Sigmoid", "Sigmoid"])  # defines the Network object - the architecture
                                                                        # and activations can be assigned as shown
# epoch training function
for i in range(100):
    expected = [0, 1] # the value I want the network to output
    input = [0.5, 0.5] # the input into the network each time
    output = nn.feedforward(input) # feedforward the network
    error = nn.backprop(output, expected) # backpropagation returns the network mse error
    print(output)

'''
Example Output:
[0.45238554 0.7235998 ]
[0.13243048 0.89986564]
[0.07694355 0.9381569 ]
[0.05382301 0.95536962]
[0.04119867 0.96515747]
[0.03327361 0.97146656]
[0.02784998 0.97586736]
[0.0239128  0.97910938]
[0.02092874 0.98159526]
[0.01859156 0.98356072]
[0.01671331 0.98515305]
[0.01517207 0.98646894]
[0.01388544 0.98757377]
[0.01279556 0.98851452]
[0.01186092 0.98932518]
[0.01105089 0.99003048]
[0.01034213 0.99064984]
[0.00971725 0.9911978 ]
[0.00916218 0.99168616]
[0.0086659 0.9921239]
[0.00821964 0.99251855]
[0.00781626 0.99287618]
[0.00745    0.99320155]
[0.0071159  0.99349896]
[0.00680988 0.99377182]
[0.0065288 0.994023 ]
[0.00626953 0.994255  ]
[0.0060298  0.99446992]
[0.0058074  0.99466949]
[0.00560052 0.99485543]
[0.00540771 0.99502891]
[0.00522754 0.99519127]
[0.00505877 0.9953435 ]
[0.00490039 0.99548651]
[0.00475154 0.99562108]
[0.00461133 0.99574798]
[0.00447903 0.99586779]
[0.00435402 0.99598112]
[0.00423572 0.9960885 ]
[0.00412356 0.99619034]
[0.00401714 0.99628706]
[0.003916   0.99637908]
[0.00381977 0.99646668]
[0.00372812 0.9965502 ]
[0.00364063 0.99662991]
[0.00355715 0.99670604]
[0.00347733 0.9967788 ]
[0.00340101 0.99684851]
[0.00332792 0.99691527]
[0.00325788 0.99697931]
[0.00319068 0.99704078]
[0.00312622 0.99709983]
[0.00306423 0.99715655]
[0.00300467 0.99721112]
[0.00294732 0.99726368]
[0.00289212 0.99731434]
[0.00283894 0.99736314]
[0.00278763 0.99741024]
[0.00273815 0.99745566]
[0.00269033 0.99749955]
[0.00264417 0.99754196]
[0.00259954 0.99758299]
[0.0025564  0.99762266]
[0.00251464 0.99766107]
[0.00247423 0.99769825]
[0.00243507 0.99773429]
[0.00239708 0.9977692 ]
[0.00236029 0.99780307]
[0.0023246  0.99783595]
[0.00228999 0.99786786]
[0.00225637 0.99789885]
[0.00222369 0.99792898]
[0.00219196 0.99795824]
[0.0021611  0.99798669]
[0.00213108 0.99801434]
[0.0021019  0.99804128]
[0.00207351 0.9980675 ]
[0.00204589 0.99809302]
[0.00201894 0.99811788]
[0.00199271 0.99814212]
[0.00196714 0.99816576]
[0.00194223 0.99818877]
[0.00191794 0.99821122]
[0.00189421 0.99823314]
[0.00187111 0.99825453]
[0.00184854 0.99827543]
[0.00182648 0.99829583]
[0.00180494 0.99831575]
[0.00178389 0.99833521]
[0.00176333 0.99835421]
[0.00174325 0.99837279]
[0.00172362 0.99839098]
[0.00170443 0.99840877]
[0.00168566 0.99842615]
[0.00166732 0.99844315]
[0.00164932 0.99845982]
[0.00163172 0.9984761 ]
[0.00161451 0.99849207]
[0.00159764 0.9985077 ]
[0.00158111 0.99852302] - as you can see the network converges to the expected output of [0, 1]

Process finished with exit code 0

'''