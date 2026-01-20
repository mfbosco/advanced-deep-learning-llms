# LoRA (Low-Rank Adaptation)

## Objetivo

Implementar e comparar **LoRA** (Low-Rank Adaptation) como técnica eficiente de fine-tuning de modelos de linguagem, reduzindo drasticamente o número de parâmetros treináveis.

## Estrutura

```
git/06-LoRA/
├── exercicio_LoRA.ipynb      # Implementação LoRA
└── comentario_LoRA.pdf       # Material complementar
```

## Conceitos Principais

### LoRA (Low-Rank Adaptation)
- **Decomposição de baixo rank**: Em vez de ajustar todos os pesos `W`, adiciona adaptação via matrizes menores `A` e `B`
- **Fórmula**: `W' = W + BA`, onde `A ∈ ℝ^(d×r)` e `B ∈ ℝ^(r×d)` com `r << d`
- **Vantagem**: Reduz parâmetros treináveis mantendo o modelo base congelado
- **Scaling factor**: `α/r` para controlar magnitude da adaptação

### Comparação com Fine-Tuning Total
- **Modelo base**: ~8.2M parâmetros treináveis (fine-tuning completo)
- **Modelo LoRA**: Apenas matrizes `A` e `B` são treináveis (rank `r=4`)
- **Eficiência**: Menos memória, treinamento mais rápido, mesma performance

## Implementação

### 1. Camada LoRA
```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        # Matrizes de baixo rank
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
    
    def forward(self, x):
        # LoRA: h = x @ (B @ A^T) * scaling
        delta_W = self.lora_B @ self.lora_A.T
        h = x @ delta_W.T
        return h * self.scaling
```

### 2. Linear com LoRA
```python
class LoRALinear(nn.Module):
    """
    Transforma uma camada linear com LoRA.
    """
    def __init__(self, original_layer, rank=4, alpha=1.0):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(original_layer.in_features, original_layer.out_features, rank, alpha)

        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.original_layer(x) + self.lora(x) # y = Wx + h
```

### 3. Aplicar LoRA ao Modelo
```python
def apply_lora_to_model(model, rank=4, alpha=1.0):
    """
    Retorna uma cópia do modelo com todas as camadas nn.Linear substituídas por LoRALinear.
    O modelo original NÃO é modificado.
    """
    model = copy.deepcopy(model)  # Faz uma cópia profunda do modelo original
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(model, name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            setattr(model, name, apply_lora_to_model(child, rank=rank, alpha=alpha))

    return model
```

## Pipeline de Treinamento

### Etapas
1. **Pré-treino do modelo base**: Treinar modelo completo com máscara causal (80% dos dados)
2. **Aplicar LoRA**: Converter camadas lineares para LoRALinear com `rank=4`
3. **Fine-tuning LoRA**: Treinar apenas matrizes `A` e `B` (20% dos dados)
4. **Comparação**: Avaliar perplexidade e qualidade de geração

### Hiperparâmetros
```python
# Modelo base (pré-treino)
epochs = 10
lr = 0.01
optimizer = AdamW(model.parameters(), lr=lr)

# LoRA (fine-tuning)
epochs_lora = 5
lr_lora = 0.001
rank = 4
alpha = 1.0
optimizer_lora = AdamW(
    filter(lambda p: p.requires_grad, lora_model.parameters()), 
    lr=lr_lora
)
```

## Resultados Esperados

- **Eficiência**: Redução drástica de parâmetros treináveis mantendo performance
- **Perplexidade**: Similar ao fine-tuning completo
- **Geração**: Qualidade comparável com custos computacionais menores
- **Aplicações**: Fine-tuning eficiente para tarefas específicas

## 📖 Referência

**Hu, Edward J., et al. (2021)**  
[*"LoRA: Low-Rank Adaptation of Large Language Models"*](https://arxiv.org/pdf/2106.09685)  
