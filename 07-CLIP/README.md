# CLIP - Embedding Multimodal

## Objetivo

Implementar busca multimodal alinhando embeddings de **imagem** e **texto** em um espaço comum, permitindo recuperar imagens através de queries textuais (e vice-versa).

## Estrutura

```
07-CLIP/
├── embedding_multimodal_clip.ipynb   # Implementação CLIP-like
└── comentario_CLIP.pdf               # Material complementar
```

## Conceitos Principais

### Embeddings Multimodais
- **Objetivo**: Alinhar representações de imagem e texto para que conceitos similares tenham embeddings próximos
- **Exemplo**: Embedding da palavra "car" deve ser similar ao embedding de uma imagem de carro
- **Aplicação**: Busca de imagens por query textual usando similaridade de cosseno

### Modelos Pré-treinados (Congelados)
- **Imagem**: EfficientNet-B0 (classificação de imagens ImageNet)
- **Texto**: BERT base uncased (processamento de linguagem natural)
- **Projeções**: Camadas lineares treináveis que mapeiam embeddings para espaço comum

### Funções de Perda

#### 1. MSE Loss (Baseline)
- **Objetivo**: Minimizar distância euclidiana entre pares positivos (imagem-texto correspondentes)
- **Limitação**: Não penaliza similaridade entre pares negativos
- **Resultado**: Funciona, mas similaridade com classes erradas ainda é alta

#### 2. Contrastive Loss (CLIP)
- **Objetivo**: Maximizar similaridade de pares positivos E minimizar similaridade de pares negativos
- **Implementação**: Cross-entropy simétrica sobre matriz de similaridades
- **Vantagem**: Melhor separação entre classes, similaridades mais discriminativas


## Resultados Esperados

### MSE Loss
- ✅ Recupera imagens corretas
- ❌ Alta similaridade com pares negativos (~0.7-0.9)
- Não discrimina bem entre classes

### Contrastive Loss (CLIP)
- ✅ Recupera imagens corretas
- ✅ Baixa similaridade com pares negativos (~0.2-0.4)
- ✅ Melhor separação entre classes
- ✅ Busca mais robusta e discriminativa


## 📖 Referência

**Radford, Alec, et al. (2021)**  
[*"Learning Transferable Visual Models From Natural Language Supervision"*](https://arxiv.org/abs/2103.00020)  
