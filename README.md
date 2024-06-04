# MLP para Porta XOR

Este repositório contém a implementação de um Perceptron Multicamadas (MLP) capaz de resolver o problema da porta lógica XOR.

## Requisitos

- Python 3.11
- NumPy
- PyTorch

## Instalação

Clone o repositório e instale os requisitos:

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
pip install numpy torch
```

## Uso

### MLP com numpy apenas

Para treinar e testar o MLP com PyTorch, execute o script `MLP.py`

```bash
python3 MLP.py
```

### MLP com o Pytorch

Para treinar e testar o MLP com PyTorch, execute o script `MLPytorch.py`

```bash
python3 MLPytorch.py
```

## Explicação do Código

### MLP com numpy

O código `MLP.py` implementa um MLP com uma camada escondida usando NumPy. A rede é treinada usando gradiente descendente e backpropagation. A implementação inclui uma métrica de acurácia para avaliar o desempenho do modelo.

### MLP com Pytorch

O código `MLPytorch.py` implementa um MLP com uma camada escondida usando PyTorch. A rede é treinada usando gradiente descendente e backpropagation. A implementação também inclui uma métrica de acurácia para avaliar o desempenho do modelo.

## Vídeo demonstrativo

É possível ver o funcionamento de ambos os códigos no vídeo clicando [aqui](./PonderadaXOR.mp4)

## Estrutura do repositório

- `MLP.py`: Implementação do MLP usando NumPy
- `MLPytorch.py`: Implementação do MLP usando PyTorch
- `README.md`: Este arquivo README com instruções de instalação e uso
