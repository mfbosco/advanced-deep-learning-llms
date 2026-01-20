# 05 - GPT-2 (Máscaras Causais e Geração Autoregressiva)

Implementação de modelo de linguagem **autoregressivo** com **máscara causal**, explorando a arquitetura GPT (Generative Pre-trained Transformer) para geração de texto.

## 🎯 Objetivo

Treinar modelo de linguagem que:
- Usa **máscara causal** (impede acesso a tokens futuros)
- Gera texto de forma **autoregressiva** (token por token)
- Implementa **multi-head attention**
- Suporta tokens especiais `<sos>` e `<eos>`

## 🗂️ Estrutura

```
05-GPT-2/
├── README.md
├── mascara_causal_gpt_2.ipynb
└── comentario-GPT-2.pdf
```

## 📚 Conceitos

**GPT** (Radford et al., 2018):
- **Decoder-only** architecture
- **Causal masking**: Token i não vê tokens > i
- **Autoregressive**: Prediz próximo token dado histórico
- Geração de texto de alta qualidade


## 🔍 GPT vs Modelos Anteriores

| Modelo | Contexto | Geração | Performance |
|--------|----------|---------|-------------|
| Bengio 2003 | Fixo | Simples | Baseline |
| Attention | Variável | Básica | Melhor |
| BERT | Bidirecional | ❌ Não gera | Compreensão |
| **GPT** | **Causal** | **✅ Excelente** | **Estado da arte** |

## 📖 Referência

**Radford, A., et al. (2019)**  
[*"Language Models are Unsupervised Multitask Learners"*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)  
OpenAI Technical Report


## 🎓 Aprendizados

1. **Máscara Causal**: Essencial para geração autoregressiva
2. **Tokens Especiais**: `<sos>`, `<eos>` delimitam sequências
3. **Temperature**: Controla criatividade vs coerência
4. **Autoregressive**: Gera um token por vez condicionado no histórico
5. **GPT = BERT invertido**: Decoder vs Encoder
