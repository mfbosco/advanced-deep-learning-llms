# 04 - BERT (Bidirectional Encoder Representations from Transformers)

Uso de **BERT pré-treinado** como extrator de features para modelo de linguagem, combinando embeddings contextuais com MLP para predição de próxima palavra.

## 🎯 Objetivo

Implementar modelo de linguagem usando:
- BERT pré-treinado (feature extractor)
- MLP para predição de próxima palavra
- Dataset Machado de Assis (português)
- Loop de treinamento customizado

## 🗂️ Estrutura

```
04-BERT/
├── README.md
├── exercicio_BERT.ipynb
└── comentario-critico-BERT.pdf
```

## 📚 Conceitos

**BERT** (Devlin et al., 2018):
- Modelo bidirecional (contexto esquerda + direita)
- Pré-treinado com Masked Language Modeling (MLM)
- Transferência de aprendizado via embeddings contextuais

**Arquitetura:**
```
Input tokens (context_size)
    ↓
BertTokenizer (subword tokenization)
    ↓
BertModel.from_pretrained() [FROZEN/FINE-TUNED]
    ↓
Last hidden state → último token embedding
    ↓
MLP compacto (D → R → vocab_size)
    ↓
CrossEntropyLoss
```

## 🛠️ Implementação

### 1. Tokenização

```python
from transformers import BertTokenizer, BertModel

# BERT português
tokenizer = BertTokenizer.from_pretrained(
    'neuralmind/bert-base-portuguese-cased'
)

# Tokeniza dataset
tokens = tokenizer(text, 
                   max_length=context_size, 
                   padding='max_length',
                   truncation=True)
```

### 2. Modelo

```python
class LanguageModelBERT(nn.Module):
    def __init__(self, bert_model='prajjwal1/bert-tiny', 
                 hidden_dim=16, freeze_bert=True):
        super().__init__()
        
        # BERT pré-treinado
        self.bert = BertModel.from_pretrained(bert_model)
        
        # Congela pesos (opcional)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # MLP compacto (evita muitos parâmetros)
        bert_dim = self.bert.config.hidden_size  # ex: 128 (tiny)
        vocab_size = self.bert.config.vocab_size  # 30522
        
        self.mlp = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),  # 128 → 16
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)  # 16 → 30522
        )
    
    def forward(self, input_ids):
        # BERT forward
        outputs = self.bert(input_ids=input_ids)
        
        # Pega embedding do último token
        last_token_emb = outputs.last_hidden_state[:, -1, :]
        
        # Predição via MLP
        logits = self.mlp(last_token_emb)
        return logits
```

### 3. Dataset

```python
class BERTDataset(Dataset):
    def __init__(self, text, tokenizer, context_size=5):
        # Tokeniza todo o texto
        tokens = tokenizer.encode(text)
        
        # Cria pares (contexto, próximo token)
        self.X, self.Y = [], []
        for i in range(context_size, len(tokens)):
            context = tokens[i-context_size:i]
            target = tokens[i]
            self.X.append(context)
            self.Y.append(target)
        
        self.X = torch.tensor(self.X, dtype=torch.long)
        self.Y = torch.tensor(self.Y, dtype=torch.long)
```

## 📊 Desafios e Soluções

| Desafio | Solução |
|---------|---------|
| **Vocab muito grande** (30K tokens) | MLP com bottleneck: D→16→vocab_size |
| **Custo computacional** | Usar BERT-tiny (2 camadas, 128 dim) |
| **Contexto limitado** | Experimentar context_size = 5, 10, 20 |
| **Overfitting** | Congelar BERT, usar dropout |


## 🔍 Comparação

| Abordagem | Embeddings | Contexto | Performance |
|-----------|-----------|----------|-------------|
| **Bengio 2003** | Estáticos | Fixo | Baseline |
| **Attention** | Aprendíveis | Variável | Melhor |
| **BERT** | **Contextuais** | **Bidirecional** | **Estado da arte** |

## 📖 Referência

**Devlin, J., et al. (2018)**  
[*"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"*](https://arxiv.org/abs/1810.04805)  
NAACL 2019

**Inovações:**
- Bidirecionalidade (vs GPT unidirecional)
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)
- Transfer learning para NLP


