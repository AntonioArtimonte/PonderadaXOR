import torch as pt
from torch.nn.functional import mse_loss

class XORPy:
    def __init__(self):
        # Define a semente para reprodução
        pt.manual_seed(33)
        # Define a arquitetura da rede
        self.model = pt.nn.Sequential(
            pt.nn.Linear(2, 5),
            pt.nn.ReLU(),
            pt.nn.Linear(5, 1)
        )
        # Define o otimizador
        self.optimizer = pt.optim.Adam(self.model.parameters(), lr=0.03)

    # Função para treinar o modelo
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            # Forward pass
            y_est = self.model(X)
  
            # Calcula a perda
            loss = mse_loss(y_est, y)

            # Propagação do erro
            loss.backward()

            # Atualiza os pesos
            self.optimizer.step()

            # Esvazia o gradiente
            self.optimizer.zero_grad()

            # Calcula a acurácia a cada 10 épocas
            if epoch % 10 == 0:
                acc = self.accuracy(y_est, y)
                print(f'Epoch [{epoch}/{epochs}] - Perda: {loss.item() * 100:.4f}%, Acurácia: {acc * 100:.2f}%')

    # Função para calcular a acurácia
    def accuracy(self, y_pred, y_true):
        predicted = y_pred >= 0.5
        correct = (predicted.float() == y_true).float()
        acc = correct.sum() / len(correct)
        return acc.item()

    # Função para testar o modelo
    def test(self, X, y):
        with pt.no_grad():
            y_test = self.model(X)
            acc = self.accuracy(y_test, y)
            print(f'\nValores finais:')
            print(y_test)
            print(f'Acurácia final: {acc * 100:.2f}%')
            return y_test, acc

# Dados
X = pt.tensor([[0, 0],
               [0, 1],
               [1, 0],
               [1, 1]], dtype=pt.float32)

y = pt.tensor([0, 1, 1, 0], dtype=pt.float32).reshape(X.shape[0], 1)

# Instância da classe
xor_model = XORPy()

# Treinamento
EPOCHS = 100
xor_model.train(X, y, EPOCHS)

# Teste
xor_model.test(X, y)
