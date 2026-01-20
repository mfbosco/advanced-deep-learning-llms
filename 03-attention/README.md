# 03 - Attention (Auto-Atenção)

Implementação de modelo de linguagem neural com **mecanismo de auto-atenção (Self-Attention)**, explorando os fundamentos do paper "Attention is All You Need" (Vaswani et al., 2017).


## 🎯 Objetivos

- Implementar camada de auto-atenção (self-attention)
- Desenvolver duas versões: **com loops** (didática) e **matricial** (eficiente)
- Integrar embeddings de posição
- Implementar projeções lineares (WQ, WK, WV, WO)
- Adicionar camada feed-forward (MLP de 2 camadas)
- Treinar modelo de linguagem com atenção
- Comparar com modelo sem atenção (Bengio 2003)

## 🗂️ Estrutura

```
03-attetion/
├── README.md
├── auto_atenção.ipynb
└── comentario-resumo-attetion-is-all-you-need.pdf
```

## 📚 Fundamentação Teórica

### Mecanismo de Auto-Atenção

O mecanismo de auto-atenção permite que o modelo "preste atenção" a diferentes partes da sequência de entrada ao processar cada token.

**Componentes principais:**

1. **Query (Q)**: "O que estou procurando?"
2. **Key (K)**: "O que eu tenho para oferecer?"
3. **Value (V)**: "O que eu realmente represento?"

**Fórmula:**
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

### Arquitetura

```
Input (context_size tokens)
    ↓
Token Embedding + Positional Embedding
    ↓
┌─────────────────────────────────────┐
│  Self-Attention Layer               │
│  ├─ Linear Projections (WQ,WK,WV)  │
│  ├─ Scaled Dot-Product Attention   │
│  └─ Output Projection (WO)         │
└─────────────────────────────────────┘
    ↓
Feed-Forward Network (2-layer MLP)
    ↓
Output Layer (vocab_size)
    ↓
Softmax (distribuição de probabilidade)
```


## 📊 Comparação de Performance

| Aspecto | Com Loops | Matricial |
|---------|-----------|-----------|
| **Tempo/batch** | ~500ms | ~50ms |
| **Velocidade** | Baseline | **10x mais rápido** |
| **Uso de GPU** | Baixo | Alto |
| **Paralelização** | Não | Sim |
| **Didática** | ✅ Excelente | ⚠️ Complexa |


## 🔍 Análise e Insights

### Vantagens da Auto-Atenção

✅ **Captura dependências longas**: Tokens distantes podem se "ver"  
✅ **Paralelização**: Operações matriciais eficientes em GPU  
✅ **Flexibilidade**: Funciona com sequências de tamanho variável  
✅ **Interpretabilidade**: Pesos de atenção são visualizáveis


### Aprendizados

1. **Loops vs Matricial**: Operações vetorizadas são muito mais rápidas
2. **Atenção é Contextual**: Cada token considera todos os outros
3. **Embeddings Posicionais**: Cruciais para ordem da sequência
4. **Projeções Lineares**: WQ, WK, WV aprendem representações úteis
5. **Escalabilidade**: Base para Transformers modernos (BERT, GPT)


## 📖 Referência Original

**Vaswani, A., et al. (2017)**  
[*"Attention is All You Need"*](https://arxiv.org/abs/1706.03762)  
Advances in Neural Information Processing Systems (NIPS)

**Contribuições do Paper:**
- Introdução do Transformer (arquitetura puramente baseada em atenção)
- Multi-head attention
- Positional encodings
- Estado da arte em tradução automática

## 📝 Notas de Implementação

- **Dataset**: Obras de Machado de Assis (mesmo do exercício anterior)
- **Duas versões**: Loop (didática) + Matricial (produção)
- **Validação**: Assert garante equivalência entre implementações
- **Treinamento**: Apenas com versão matricial (eficiência)
- **Comparação**: Modelo com/sem atenção
- **Material de apoio**: PDF com resumo do paper original

## 🔗 Arquivos do Projeto

- `auto_atenção.ipynb` - Implementação completa
- `comentario-resumo-attetion-is-all-you-need.pdf` - Resumo do paper