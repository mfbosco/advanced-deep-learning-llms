# 01 - Sentiment Analysis (Análise de Sentimentos)

Projeto de análise de sentimentos no dataset IMDB utilizando a abordagem Bag of Words (BoW) e redes neurais com PyTorch.

## 📋 Descrição

Este projeto implementa um classificador de sentimentos binário (positivo/negativo) para reviews de filmes do dataset IMDB. A implementação utiliza uma abordagem clássica de Bag of Words combinada com uma rede neural MLP (Multi-Layer Perceptron).

## 🎯 Objetivos

- Implementar pipeline completo de processamento de texto
- Construir representação Bag of Words eficiente
- Treinar modelo de classificação binária
- Avaliar desempenho em dados de teste
- Otimizar processamento para uso em GPU


## 📊 Dataset

- **Fonte**: Stanford IMDB Dataset (via Hugging Face)
- **Tamanho**: 25,000 reviews para treino + 25,000 para teste
- **Classes**: Binária (0 = negativo, 1 = positivo)
- **Formato**: Texto livre (reviews em inglês)

## 🛠️ Implementação

### 1. Pré-processamento e Tokenização

```python
def pre_process(text):
    return re.sub(r'[^\w\s]', '', text).lower().split()
```

**Características:**
- Remoção de pontuações
- Conversão para minúsculas
- Tokenização por espaços
- Vocabulário limitado às 20,000 palavras mais frequentes

### 2. Representação Bag of Words

```python
class IMDBDataset(Dataset):
    def __init__(self, split, vocab):
        self.labels = torch.tensor(imdb_dic[split]['label'])
        texts = imdb_dic[split]['text']
        self.X = torch.zeros((len(texts), len(vocab)+1), dtype=torch.float32)
        for i, line in enumerate(texts):
            for word in tokenizer(line, vocab):
                self.X[i, word] = 1
```

**Otimizações:**
- ✅ Vetorização pré-computada durante inicialização
- ✅ Uso de tensores PyTorch nativos
- ✅ Evita reprocessamento a cada batch
- ✅ Suporte eficiente para GPU

### 3. Arquitetura do Modelo

```python
class OneHotMLP(nn.Module):
    def __init__(self, vocab_size):
        super(OneHotMLP, self).__init__()
        self.fc = nn.Linear(vocab_size + 1, 2)
```

**Especificações:**
- Modelo: MLP simples (Linear + Softmax)
- Input: Vetor BoW de tamanho vocab_size + 1
- Output: 2 classes (negativo/positivo)
- Função de perda: CrossEntropyLoss
- Otimizador: SGD com learning rate 0.1

### 4. Treinamento

**Configuração:**
- Split: 80% treino / 20% validação
- Batch size: 32
- Épocas: 10
- Device: GPU (quando disponível)

**Processo:**
- Loop de treino com backpropagation
- Validação a cada época
- Monitoramento de loss e acurácia

## 📈 Resultados

| Métrica | Valor |
|---------|-------|
| Test Accuracy | **86.77%** |
| Treino | ~2s por época (GPU) |
| Velocidade | ~10x mais rápido com GPU |


## 🔍 Análise de Performance

### Antes das Otimizações
- ⏱️ ~50-60s por época (CPU)
- 🐌 Reprocessamento em cada batch
- 📉 Vocabulário inconsistente

### Depois das Otimizações
- ⚡ ~2s por época (GPU)
- 🚀 Vetorização pré-computada
- 📈 Pipeline unificado

## 📚 Conceitos Abordados

- **Bag of Words**: Representação vetorial de texto
- **Tokenização**: Processamento e normalização de texto
- **Vocabulário**: Construção e limitação de features
- **MLP**: Redes neurais feedforward
- **Binary Classification**: Classificação binária
- **PyTorch Dataset**: Implementação eficiente de datasets
- **GPU Optimization**: Uso de CUDA para aceleração

## 🎓 Aprendizados

1. Importância do pré-processamento consistente
2. Impacto da otimização no tempo de treinamento
3. Trade-off entre tamanho de vocabulário e performance
4. Benefícios da vetorização pré-computada
5. Uso eficiente de GPU em PyTorch

## 📝 Notas

- Este projeto foi desenvolvido como parte do processo seletivo para o curso
- Foco em implementação eficiente e otimizada
- Abordagem educacional com comentários explicativos
- Versão de referência: 13 de julho de 2025

## 🔗 Referências

- [IMDB Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

**Nota**: Para detalhes completos da implementação, consulte o notebook com todas as células e outputs.
