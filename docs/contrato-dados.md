# Contrato de dados: limpeza → transformação

Define a interface entre `02_limpeza.py` e `04_transformacoes.py`, para que as duas etapas evoluam em paralelo.

## Regra de fronteira

**Corrigir coluna existente é limpeza. Criar coluna nova é transformação.**

| Limpeza (`02_limpeza.py`) | Transformação (`04_transformacoes.py`) |
|---|---|
| Padronização de grafia e caixa | Construção de atributos (classes clínicas de PA, IMC e FC; faixa etária; indicadores derivados) |
| Valores inválidos e sentinelas (0, negativos, fora de faixa) → NaN | Eliminação de atributos irrelevantes ou redundantes |
| Correção de datas e recálculo de colunas derivadas já existentes (`idade`, `imc`) | Discretização e codificação |
| Remoção de registros sem alvo | Escala (normalização/padronização) |
| Registro das decisões tomadas | Split treino/teste e pipeline ajustado apenas no treino |

## Interface

- **Arquivo:** `data/processed/UCMF_clean.csv` — UTF-8, separador `,`, decimal `.`, uma linha por atendimento.
- **Colunas:** mesmas 21 da base original, em `snake_case` (`pa_sistolica`, `normal_x_anormal`, `hda1`, ...). Nenhuma coluna nova; nenhuma removida além do que a limpeza descartar e documentar.
- **Tipos:** numéricos como `float`; datas em ISO (`AAAA-MM-DD`); categóricos como texto com um único rótulo por categoria.
- **Ausentes:** representados como vazio/NaN. **A limpeza não imputa.** Imputação depende de estatística da amostra e é feita na transformação, ajustada só no treino. Categorias que significam "avaliado e negativo" (ex.: sopro `ausente`) são valores válidos, não NaN.
- **Sentinelas:** nunca usar 0, -1 ou texto para indicar "não medido".
- **Registro de decisões:** `output/tabelas/limpeza_decisoes.csv` com `variavel; problema; tratamento; n_afetados`.

## Saídas da transformação

- `data/processed/UCMF_features.csv` — base com atributos construídos e eliminações aplicadas (transformações determinísticas, sem estatística da amostra).
- Pipeline (`ColumnTransformer`) com imputação, codificação e escala, ajustado no treino e serializado; split estratificado com semente fixa.

## Regra: transformações são modeladas no treino e apenas aplicadas no teste

Há dois tipos de transformação:

| Tipo | Exemplos | Depende dos dados? | Onde é aplicada |
|---|---|---|---|
| **Determinística** | IMC = peso / altura²; classe de PA por tabela clínica; faixa etária | Não — usa fórmula ou tabela externa | Na base inteira; resultado vai para `UCMF_features.csv` |
| **Estatística** | imputação pela mediana; Z-score (média e desvio); one-hot (lista de categorias); agregação de categorias raras | Sim — os parâmetros são calculados a partir da amostra | Parâmetros calculados **só no treino**; reaplicados iguais no teste |

Para as estatísticas, a ordem é obrigatória:

1. **Dividir primeiro.** `train_test_split` estratificado pelo alvo (mantém a proporção Normal/Anormal nos dois lados), `test_size=0.2`, `random_state=42` para que qualquer execução produza a mesma divisão.
2. **Ajustar só no treino.** `pre.fit(X_treino)` calcula mediana, média, desvio e categorias usando apenas os registros de treino.
3. **Aplicar sem reajustar.** `pre.transform(X_treino)` e `pre.transform(X_teste)` usam os mesmos parâmetros. Nunca chamar `fit` no teste.
4. **Serializar.** O pré-processador ajustado é salvo (`joblib`) e reutilizado pela modelagem e por qualquer dado novo, sem recalcular nada.

```python
# errado — a média e o desvio "viram" o teste antes da divisão
X_esc = StandardScaler().fit_transform(X)
X_tr, X_te = train_test_split(X_esc, ...)

# certo — parâmetros vêm só do treino e são reaplicados no teste
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pre.fit(X_tr)
X_tr_t = pre.transform(X_tr)
X_te_t = pre.transform(X_te)
```

Motivo: se os parâmetros forem calculados na base inteira, informação do teste entra no treino (vazamento) e a avaliação do modelo fica otimista. O teste deve se comportar como dado que o modelo nunca viu — o mesmo que acontecerá em produção.

Consequência para a limpeza: como a imputação é estatística, ela **não** pode ser feita em `02_limpeza.py` sobre a base inteira; a limpeza entrega NaN e a transformação imputa dentro do pipeline.
