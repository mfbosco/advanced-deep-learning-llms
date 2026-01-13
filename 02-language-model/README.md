# 02 - Language Model (Modelo de Linguagem - Bengio 2003)

Implementação de um modelo de linguagem neural baseado no trabalho seminal de **Bengio et al. (2003)**, utilizando embeddings de palavras e redes neurais MLP para prever a próxima palavra em uma sequência.

## 📋 Descrição

Este projeto implementa um modelo de linguagem estatístico neural que aprende a prever a próxima palavra dado um contexto de palavras anteriores. A abordagem utiliza embeddings aprendíveis e uma arquitetura feedforward simples, representando um dos primeiros usos bem-sucedidos de redes neurais para modelagem de linguagem.

## 🎯 Objetivos

- Implementar modelo de linguagem neural do tipo feedforward
- Utilizar embeddings de palavras aprendíveis
- Treinar modelo para previsão da próxima palavra
- Calcular perplexidade (métrica de avaliação)
- Gerar texto de forma autoregressiva
- Alcançar perplexidade < 200


##  Fundamentação Teórica

### Modelo de Linguagem Neural (Bengio 2003)

O modelo proposto por Bengio revolucionou a área de NLP ao introduzir:

1. **Word Embeddings**: Representações vetoriais densas e de baixa dimensionalidade
2. **Arquitetura Neural**: MLP para capturar dependências entre palavras
3. **Aprendizado Conjunto**: Embeddings e pesos da rede aprendidos simultaneamente

### Arquitetura

```
Input (context_size palavras) 
    ↓
Embedding Layer (vocab_size → embedding_dim)
    ↓
Concatenação dos embeddings
    ↓
Hidden Layer (não-linear)
    ↓
Output Layer (→ vocab_size)
    ↓
Softmax (distribuição de probabilidade)
```

## 🛠️ Implementação

### 1. Preparação de Dados

**Dataset**: Obras de Machado de Assis (pré-processado)  
**Vocabulário**: 2001 tokens (top 2000 + `<unk>`)  
**Context Size**: 5 palavras anteriores  
**Target**: Próxima palavra (6ª palavra)

### 2. Dataset PyTorch

Implementação da classe `MachadoDataset`:

```python
class MachadoDataset(Dataset):
    """Dataset para modelagem de linguagem com contexto.
    
    Attributes:
        X: contextos (N, context_size)
        Y: targets (N,)
        context_size: Tamanho do contexto (janela de tokens)
    """
    
    def __init__(self, X, Y, context_size=5):
        # Validações
        assert len(X) == len(Y), "Número de contextos e alvos deve ser igual"
        assert all(len(ctx) == context_size for ctx in X)
        
        # Converte para tensores
        self.X = torch.tensor([[x for x in ctx] for ctx in X], dtype=torch.long)
        self.Y = torch.tensor([y for y in Y], dtype=torch.long)
        self.context_size = context_size
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
```

**Características:**
- Split: 80% treino / 20% validação
- Remoção de tokens `<unk>` (índice 0)
- Janela deslizante para criar exemplos
- Formato: (context_tensor, target_tensor)
- Conversão imediata para tensores PyTorch

### 3. Arquitetura do Modelo

```python
class LanguageModel(nn.Module):
    def __init__(self, vocab_size=2001, embedding_dim=128, 
                 hidden_dim=512, context_size=5):
        super(LanguageModel, self).__init__()
        
        # Camada de embedding
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Camada oculta (não-linear)
        self.hidden = nn.Sequential(
            nn.Linear(context_size * embedding_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Camada de saída
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        # x: (batch_size, context_size)
        embeds = self.embeddings(x)  # (batch, context, embed_dim)
        embeds = embeds.view(embeds.shape[0], -1)  # flatten
        hidden = self.hidden(embeds)  # (batch, hidden_dim)
        out = self.output(hidden)  # (batch, vocab_size)
        return out
```

**Hiperparâmetros (configuração real):**
- `vocab_size`: 2001 (top 2000 + `<unk>`)
- `embedding_dim`: 128
- `hidden_dim`: 512
- `context_size`: 5
- `batch_size`: 256
- Ativação: **Tanh** (como no paper original)

### 4. Treinamento

**Configuração:**
```python
epochs = 5
lr = 0.001
batch_size = 256
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
```

**Funções de Treino e Validação:**
```python
def train_batch(model, X, Y, optimizer, criterion, device):
    model.train()
    X, Y = X.to(device), Y.to(device)
    output = model(X)
    loss = criterion(output, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def validate_batch(model, X, Y, criterion, device):
    model.eval()
    X, Y = X.to(device), Y.to(device)
    output = model(X)
    loss = criterion(output, Y)
    return loss.item()
```

**Loop de Treino:**
- Validação de índices antes do treino (segurança)
- Cálculo de loss e perplexidade por época
- Monitoramento de tempo por época
- Plots de loss e perplexidade ao longo das épocas

### 5. Avaliação: Perplexidade

$perplexity = \exp(average_loss)$

**Interpretação:**
- Menor perplexidade = melhor modelo
- Perplexidade de 100 = modelo considera ~100 palavras igualmente prováveis
- **Meta**: Perplexidade < 200

### 6. Geração de Texto

Geração autoregressiva usando amostragem.

**Exemplos (português - Machado de Assis):**
```python
generate_text(model, vocab, "Era uma dia belo de sol", max_length=9)
# Output: "Era uma dia belo de sol e a casa de"
```

## 🔍 Conceitos Abordados

- **Language Modeling**: Modelagem estatística de sequências
- **Word Embeddings**: Representações vetoriais densas
- **N-gram Context**: Uso de contexto de tamanho fixo
- **Feedforward Neural Networks**: MLPs para NLP
- **Perplexity**: Métrica de avaliação de modelos de linguagem
- **Autoregressive Generation**: Geração sequencial de texto
- **Cross-Entropy Loss**: Função objetivo para classificação

## 🎓 Aprendizados

1. **Embeddings vs One-Hot**: Embeddings capturam semântica e reduzem dimensionalidade
2. **Context Window**: Trade-off entre contexto e complexidade
3. **Perplexity**: Métrica intuitiva para modelos probabilísticos
4. **Geração Autoregressiva**: Base para modelos modernos (GPT)
5. **Vocabulário**: Tratamento de OOV é crucial

## � Componentes e Métricas

| Componente | Descrição | Valor/Tipo |
|------------|-----------|------------|
| **Dataset** | Obras de Machado de Assis | ~176K pares (X,Y) |
| **Vocabulário** | Top tokens mais frequentes | 2001 |
| **Embedding Layer** | Converte tokens em vetores densos | dim=128 |
| **Hidden Layer** | Aprende representações não-lineares | dim=512, Tanh |
| **Output Layer** | Gera distribuição sobre vocabulário | dim=2001 |
| **Batch Size** | Tamanho do lote de treinamento | 256 |
| **Perplexity** | Métrica de avaliação | < 200 (meta) |


## 📖 Referência Original

**Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003)**  
[*"A Neural Probabilistic Language Model"*](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)  
Journal of Machine Learning Research, 3, 1137-1155

**Inovações do Paper:**
- Primeira aplicação bem-sucedida de embeddings
- Demonstração de que redes neurais superam n-gramas
- Base para Word2Vec, GloVe e modelos modernos


## 📝 Notas de Implementação

- **Dataset**: Obras completas de Machado de Assis (domínio público)
- **Pré-processamento**: Realizado no notebook `Preparação_de_dados.ipynb`
- **Créditos**: Dataset preparado por Augusto Zolet
- **Exercício**: Desenvolvido com suporte de ChatGPT/Copilot
- **Foco**: Compreensão dos fundamentos de modelos de linguagem
- **Geração**: Textos em português com estilo literário

---

**Material Educacional**: Implementação prática do modelo de Bengio 2003 aplicado a textos literários em português.

