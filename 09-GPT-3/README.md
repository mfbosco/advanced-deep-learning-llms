# GPT-3 - Prompt Engineering e Chain-of-Thought

## Objetivo

Comparar técnicas de **prompt engineering** para análise de sentimentos em reviews de filmes (IMDB) usando LLMs (gpt-5-nano): **zero-shot**, **few-shot** e **Chain-of-Thought (CoT)**.

## Estrutura

```
09-GPT-3/
├── exercicio_CoT.ipynb                # Implementação das 3 técnicas
├── comentario_GPT-3.pdf               # Material sobre GPT-3
└── comentario_Chain-of-Thought.pdf    # Material sobre CoT
```

## Conceitos Principais

### Técnicas de Prompt Engineering

#### 1. Zero-Shot
- **Definição**: Modelo classifica sem exemplos prévios, apenas com instrução
- **Prompt**: "Classify whether the following text is positive or negative"
- **Vantagem**: Simples, sem necessidade de exemplos
- **Limitação**: Performance pode ser inferior

#### 2. Few-Shot Learning
- **Definição**: Fornece poucos exemplos (2-4) de cada classe antes da tarefa
- **Prompt**: Exemplos positivos + Exemplos negativos + Review a classificar
- **Vantagem**: Modelo aprende padrão dos exemplos
- **Custo**: Maior tamanho de prompt (mais tokens)

#### 3. Chain-of-Thought (CoT)
- **Definição**: Modelo explicita raciocínio antes da resposta final
- **Prompt**: Exemplos com etapas de raciocínio ("The words 'loved', 'great' indicate positive sentiment... Final answer: 1")
- **Vantagem**: Melhora performance em tarefas que requerem raciocínio
- **Inspiração**: Simula processo humano de pensamento passo-a-passo

## Pipeline de Execução

1. **Carregar IMDB**: Dataset completo (50k reviews)
2. **Amostragem balanceada**: 50 treino (para few-shot) + 500 teste
3. **Zero-Shot**: Classificar 500 amostras teste sem exemplos
4. **CoT**: Classificar com exemplos de raciocínio
5. **Few-Shot**: Classificar com 4 exemplos (2 pos + 2 neg)
6. **Avaliar**: Comparar precision, recall, F1 das 3 técnicas


### Checklist de Implementação

- [x] Carregar dataset IMDB
- [x] Amostragem balanceada (50 treino + 500 teste)
- [x] Implementar prompt zero-shot
- [x] Implementar prompt CoT com reasoning
- [x] Implementar prompt few-shot (4 exemplos)
- [x] Função de inferência com gpt-5-nano
- [x] Avaliar 500 amostras com cada técnica
- [x] Salvar resultados (JSON)
- [x] Métricas: classification report + confusion matrix
- [x] Análise de erros e tempo de resposta

## Resultados Esperados

### Performance Comparativa
- **Zero-Shot**: ~85-90% accuracy (baseline)
- **Few-Shot**: ~88-93% accuracy (melhora com exemplos)
- **Chain-of-Thought**: ~90-95% accuracy (melhor com raciocínio explícito)

### Observações
- ✅ CoT geralmente supera zero-shot e few-shot em tarefas que requerem raciocínio
- ⚠️ CoT usa mais tokens (custo maior)
- ✅ Few-shot eficiente quando há poucos exemplos disponíveis
- ⏱️ Tempo médio de resposta: ~1-3 segundos/amostra (gpt-5-nano)


## Refrências

**Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E. H., Le, Q., & Zhou, D. (2022)**  
[*"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"*](https://arxiv.org/abs/2201.11903)

**Brown, T. B., et. al. (2020)**  
[*"Language Models Are Few-Shot Learners"*](https://arxiv.org/abs/2005.14165)

