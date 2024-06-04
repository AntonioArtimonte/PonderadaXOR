import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NN:
    def __init__(self, inputs):
        self.inputs = inputs
        self.l = len(self.inputs)
        self.li = len(self.inputs[0])

        # Inicializando os pesos aleatoriamente para a camada de entrada e a camada escondida
        self.wi = np.random.random((self.li, self.l))  # Pesos da camada de entrada para a camada escondida
        self.wh = np.random.random((self.l, 1))        # Pesos da camada escondida para a camada de saída

    def feedforward(self, inp):
        self.hidden = sigmoid(np.dot(inp, self.wi))  # Ativação da camada escondida
        self.output = sigmoid(np.dot(self.hidden, self.wh))  # Ativação da camada de saída
        return self.output

    def backpropagation(self, inputs, outputs, learning_rate):
        output = self.feedforward(inputs)
        # Cálculo do erro (Função de custo: erro quadrático médio)
        error = outputs - output
        d_output = error * sigmoid_derivative(output)  # Derivada do erro na saída

        error_hidden = d_output.dot(self.wh.T)
        d_hidden = error_hidden * sigmoid_derivative(self.hidden)  # Derivada do erro na camada escondida

        # Atualização dos pesos com gradiente descendente
        self.wh += learning_rate * self.hidden.T.dot(d_output)
        self.wi += learning_rate * inputs.T.dot(d_hidden)

    def train(self, inputs, outputs, learning_rate, epochs):
        for epoch in range(epochs):
            self.backpropagation(inputs, outputs, learning_rate)

    def accuracy(self, inputs, outputs):
        predictions = self.feedforward(inputs)
        predictions = (predictions > 0.5).astype(int)
        return np.mean(predictions == outputs) * 100

# Dados de entrada e saída para o problema XOR
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

n = NN(inputs)
n.train(inputs, outputs, 0.1, 10000)
print("Previsões após o treinamento:")
print(n.feedforward(inputs))
print("Acurácia após o treinamento:")
print(n.accuracy(inputs, outputs),"%")
