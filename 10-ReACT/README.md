# ReACT - Agente com Raciocínio e Ação

## Objetivo

Implementar um **agente ReACT** (Reasoning + Acting) com LangChain para responder perguntas sobre declaração de imposto de renda brasileiro (IRPF) usando busca em documentos legais.

## Estrutura

```
10-ReACT/
├── ReACT.ipynb               # Implementação do agente ReACT
└── comentario_ReACT.pdf      # Material sobre ReACT
```

## Conceitos Principais

### ReACT Framework
- **Definição**: Paradigma que combina **Reasoning** (raciocínio) com **Acting** (ações/ferramentas)
- **Ciclo**: Thought (pensamento) → Action (busca) → Observation (resultado) → Thought → ...
- **Diferencial**: Agente decide iterativamente quando buscar mais informações ou responder
- **Controle**: Limita número de buscas (máx 2) para evitar loops infinitos

### Dataset BR-TaxQA-R
- **Fonte**: Hugging Face (unicamp-dl/BR-TaxQA-R)
- **Conteúdo**: Perguntas e respostas sobre IRPF brasileiro
- **Documentos**: Base legal (leis, instruções normativas, soluções de consulta)
- **Avaliação**: 100 primeiras perguntas do conjunto de teste


## Pipeline de Execução

1. **Carregar dados**: BR-TaxQA-R (100 perguntas + documentos legais)
2. **Chunking**: Dividir documentos em chunks de 1000 chars (overlap=50)
3. **Embeddings**: sentence-transformers/multi-qa-mpnet-base-cos-v1
4. **Indexação FAISS**: Criar índice de busca por similaridade
5. **Criar agente ReACT**: LangGraph + LLM + ferramenta de busca
6. **Avaliar**: Executar 100 perguntas e calcular BERTScore + F1-BoW

## Vantagens do ReACT
| Aspecto | RAG Tradicional | ReACT |
|---------|-----------------|-------|
| Busca | Única busca fixa | Múltiplas buscas adaptativas |
| Controle | Sem controle sobre busca | Agente decide quando buscar |
| Raciocínio | Direto | Iterativo (think → act → observe) |

## Referências

**Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023)**  
[*"ReAct: Synergizing Reasoning and Acting in Language Models"*](https://arxiv.org/abs/2210.03629)


