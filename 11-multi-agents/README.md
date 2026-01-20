# Multi-Agentes - Sistema Colaborativo

## Objetivo

Implementar um **chatbot multiagente** com LangGraph para responder perguntas sobre imposto de renda brasileiro, combinando **agente buscador** (retrieval) + **agente redator** (adaptação de tom) com **memória persistente** e **interface gráfica**.

## Estrutura

```
11-multi-agents/
├── multiagentes.ipynb            # Sistema multiagente completo
└── comentario_multiagentes.pdf   # Material sobre multi-agentes
```

## Conceitos Principais

### Sistema Multiagente
- **Definição**: Múltiplos agentes especializados que colaboram para resolver uma tarefa
- **Arquitetura**: Agente Buscador → Agente Redator → Resposta final
- **Coordenação**: LangGraph gerencia fluxo entre agentes
- **Memória**: MemorySaver mantém histórico de conversas (contexto persistente)

### Agentes Especializados

#### 1. Agente Buscador (Retriever)
- **Função**: Buscar informações relevantes em documentos legais via FAISS
- **Ferramentas**: retriver_tool (acesso a base de conhecimento)
- **Output**: Resposta técnica baseada em documentos oficiais

#### 2. Agente Redator (Humor Agent)
- **Função**: Adaptar resposta técnica para linguagem jovial
- **Público-alvo**: Jovens fazendo primeira declaração
- **Output**: Resposta simplificada e amigável

## Pipeline de Execução

1. **Carregar dados**: BR-TaxQA-R (reutiliza chunks e FAISS do exercício anterior)
2. **Criar agentes**: Buscador (com tool) + Redator (adaptação de tom)
3. **Construir grafo**: LangGraph com memória persistente
4. **Interface**: Gradio para interação via browser
5. **Avaliação**: LLM-as-Judge (sem dataset de referência)


## Checklist de Implementação

- [x] Reutilizar chunks e FAISS do exercício ReACT
- [x] Criar agente buscador com retriver_tool
- [x] Criar agente redator (adaptação de tom jovial)
- [x] Construir grafo LangGraph com 2 agentes
- [x] Implementar memória persistente (MemorySaver)
- [x] Interface Gradio para chat
- [x] Sistema de chat_with_memory (thread_id)
- [x] Avaliação com LLM-as-Judge
- [x] Avaliar 40-100 perguntas
- [x] Métricas: nota_raciocinio, nota_resposta

## Resultados Esperados

### Exemplo de Fluxo
```
Usuário: "O que é IRPF?"

[Agente Buscador]
→ Busca documentos via retriver_tool
→ Gera resposta técnica citando leis

[Agente Redator]
→ Recebe resposta técnica
→ Adapta para linguagem jovial
→ Simplifica termos legais

Saída: "E aí! IRPF é basicamente aquele imposto que você declara anualmente 
pra Receita Federal saber quanto você ganhou no ano. Se você recebeu mais 
de R$ 28.559,70 em 2023, precisa declarar! 📊"
```


## Referências

**Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023)**  
[*"Improving Factuality and Reasoning in Language Models through Multiagent Debate"*](https://arxiv.org/abs/2305.14325)

