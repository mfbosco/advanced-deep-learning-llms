# RAG (Retrieval-Augmented Generation)

## Objetivo

Implementar um sistema completo de **RAG** combinando busca densa (retrieval) com LLMs (generation) para responder perguntas do dataset IIRC.

## Estrutura

```
08-RAG/
├── exercicio_RAG.ipynb    # Pipeline RAG completo
└── comentario_RAG.pdf     # Material complementar
```

## Conceitos Principais

### RAG Pipeline
1. **Segmentação** (Chunking): Dividir documentos em chunks menores com janelamento
2. **Embedding**: Converter chunks em vetores usando sentence-transformers
3. **Indexação**: Armazenar embeddings em FAISS para busca eficiente
4. **Retrieval**: Buscar top-k chunks mais relevantes para a query
5. **Generation**: LLM (gpt-5-nano) gera resposta baseada nos contextos recuperados
6. **Avaliação**: Métricas F1-bag-of-words, precision, recall, exact match

### Dataset IIRC
- **Formato**: Perguntas que requerem múltiplos contextos/links
- **Tipos de resposta**: span (texto), binary (sim/não), value (numérico)
- **Desafio**: 150 primeiras perguntas com resposta (exclui perguntas sem resposta)

## Pipeline

1. **Carregar dados**: IIRC test set + context articles
2. **Filtrar artigos relevantes**: Apenas artigos mencionados nas perguntas (reduz indexação)
3. **Segmentação**: Janelamento com stride=2, window_size=3 sentenças
4. **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
5. **Indexação FAISS**: Criar e salvar índice
6. **Avaliar 150 perguntas**: Gerar respostas e calcular métricas


### Checklist de Implementação

- [x] Download e parsing do dataset IIRC
- [x] Filtragem de artigos relevantes (reduz de ~15k para ~500 artigos)
- [x] Segmentação com janelamento (stride=2, max_length=3)
- [x] Embeddings com sentence-transformers
- [x] Indexação FAISS com LangChain
- [x] Função de busca por similaridade (top-k)
- [x] Prompt engineering com contextos
- [x] Geração de respostas com gpt-5-nano
- [x] Métricas F1-BoW, precision, recall, exact match
- [x] Avaliação 150 perguntas
- [x] Análise por tipo de resposta (span/binary/value)

## 📖 Referência

**Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., & Wang, H. (2023)**  
[*"Retrieval-Augmented Generation for Large Language Models: A Survey"*](https://arxiv.org/abs/2312.10997)

**Pereira, J., Fidalgo, R., Lotufo, R., & Nogueira, R. (2022)**  
[*"Visconde: Multi-document QA with GPT-3 and Neural Reranking"*](https://arxiv.org/abs/2212.09656)

