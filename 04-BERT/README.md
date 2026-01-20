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

**BERT** (Devlin et al., 2019):
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

**Devlin, J., et al. (2019)**  
[*"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"*](https://arxiv.org/abs/1810.04805)  


**Inovações:**
- Bidirecionalidade (vs GPT unidirecional)
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)
- Transfer learning para NLP


