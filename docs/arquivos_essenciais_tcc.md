# 📋 ARQUIVOS ESSENCIAIS DO TCC - MAPEAMENTO COMPLETO

**Projeto:** Predição de Qualidade de Vida com Wearables
**Autor:** João Victor Testeves
**Última atualização:** 30/12/2024

---

## **🔵 ETAPA 1: Análise Exploratória**

### Notebook Principal:
- ✅ `notebooks/exploratory/analise_exploratoria_final.ipynb`
  - Análise completa do dataset
  - Análise de participantes
  - Estatísticas descritivas
  - Qualidade dos dados
  - **Entrada:** Dataset cru (1.373 registros)
  - **Saída:** Relatórios de análise

---

## **🔵 ETAPA 2: Pipeline de Pré-processamento**

### 2.1 - Tratamento de Gaps Temporais (Imputação)

**Scripts:**
- ✅ `src/preprocessing/analise_gaps_temporais.py` (análise diagnóstica)
- ✅ `src/preprocessing/pipeline_imputacao_temporal.py` (execução)

**Datasets:**
- 📥 **Entrada:** `data/raw/20230625-processed-physical-qol.csv` (1.373 registros, 88 features)
- 📥 **Entrada:** `data/raw/20230625-processed-psychological-qol.csv` (1.373 registros, 88 features)
- 📤 **Saída:** `data/interim/dados_physical_imputado.csv` (2.267 registros, 88 features)
- 📤 **Saída:** `data/interim/dados_psychological_imputado.csv` (2.267 registros, 88 features)

**Técnicas:**
- Forward-fill para gaps < 7 dias
- Interpolação linear para gaps 7-30 dias
- Sem imputação para gaps > 30 dias

---

### 2.2 - Tratamento de Outliers (Flags)

**Scripts:**
- ✅ `src/preprocessing/analise_outliers.py` (análise diagnóstica)
- ✅ `src/preprocessing/adicionar_flags_outliers.py` (execução)

**Datasets:**
- 📥 **Entrada:** `data/interim/dados_physical_imputado.csv`
- 📥 **Entrada:** `data/interim/dados_psychological_imputado.csv`
- 📤 **Saída:** `data/interim/dados_physical_com_flags_outliers.csv`
- 📤 **Saída:** `data/interim/dados_psychological_com_flags_outliers.csv`

**Técnicas:**
- Método IQR (threshold 1.5)
- Outliers sinalizados com flags, NÃO removidos
- Estratégia conservadora: valores extremos podem ser comportamentos reais

---

### 2.3 - Feature Engineering

**Scripts:**
- ✅ `src/feature_engineering/criar_features.py`

**Datasets:**
- 📥 **Entrada:** `data/interim/dados_physical_com_flags_outliers.csv`
- 📥 **Entrada:** `data/interim/dados_psychological_com_flags_outliers.csv`
- 📤 **Saída:** `data/processed/dados_physical_com_features.csv` (118 features: 88 originais + 30 derivadas)
- 📤 **Saída:** `data/processed/dados_psychological_com_features.csv` (118 features: 88 originais + 30 derivadas)

**30 Features Derivadas:**
- **Sono (9):** eficiência, ratios, quality score, short/long sleeper
- **HRV (5):** balanço autonômico, stress index, recovery score, fitness cardiovascular
- **Atividade Física (4):** dias ativos/sedentários, intensidade, corrida
- **Comunicação (6):** total calls, call balance, duração média, contato social
- **Temporais (4):** dia da semana, weekend, mês, dias desde início
- **Compostas (3):** recovery index, stress index, wellbeing composite

---

### 2.4 - Redução de Multicolinearidade (VIF)

**Scripts:**
- ✅ `src/preprocessing/criar_datasets_pos_vif.py` (principal)
- ✅ `src/preprocessing/criar_dados_crus_pos_vif.py` (versão sem features derivadas)

**Datasets:**
- 📥 **Entrada:** `data/processed/dados_physical_com_features.csv` (118 features)
- 📥 **Entrada:** `data/processed/dados_psychological_com_features.csv` (118 features)
- 📤 **Saída:** `data/processed/dados_physical_apos_vif.csv` (60 features após VIF + one-hot encoding)
- 📤 **Saída:** `data/processed/dados_psychological_apos_vif.csv` (60 features após VIF + one-hot encoding)
- 📤 **Saída:** `data/processed/dados_crus_physical_apos_vif.csv` (versão sem engenharia)
- 📤 **Saída:** `data/processed/dados_crus_psychological_apos_vif.csv` (versão sem engenharia)

**Técnicas:**
- VIF iterativo com threshold = 10
- Redução: 118 → 60 features

---

## **🔵 ETAPA 3: Modelagem**

### Notebook Principal:
- ✅ `notebooks/modeling/comparacao_metodologias_5_cenarios_completa.ipynb`
  - **5 cenários experimentais (A, B, C, D, E)**
  - Comparação KFold (k=10) vs GroupKFold (k=5)
  - Comparação Featurewiz (40 features) vs VIF (60 features)
  - 4 modelos tradicionais + 3 modelos avançados
  - Teste de 4 hipóteses sobre data leakage e multicolinearidade

**Scripts auxiliares:**
- ✅ `src/preprocessing/preparar_dados_modelagem.py` (preparação final para ML)

**Datasets de entrada:**
- `data/raw/20230625-processed-physical-qol.csv` (dataset Pedro Almir - 1.373 registros)
- `data/raw/20230625-processed-psychological-qol.csv` (dataset Pedro Almir - 1.373 registros)
- `data/processed/dados_physical_apos_vif.csv` (dataset João - 2.267 registros, 60 features)
- `data/processed/dados_psychological_apos_vif.csv` (dataset João - 2.267 registros, 60 features)

**Resultados:**
- `results/comparacao_5_cenarios/comparacao_5_cenarios_completa.csv`
- `results/comparacao_5_cenarios/resumo_5_cenarios.csv`
- Gráficos de comparação RMSE e R²

---

## **🔵 CINCO CENÁRIOS EXPERIMENTAIS**

| Cenário | Validação | Dataset | Features | Modelos |
|---------|-----------|---------|----------|---------|
| **A** | KFold (k=10) shuffle | Featurewiz | 40 | 4 Tradicionais |
| **B** | KFold (k=10) shuffle | Pós-VIF | 60 | 4 Tradicionais |
| **C** | GroupKFold (k=5) | Featurewiz | 40 | 4 Tradicionais |
| **D** | GroupKFold (k=5) | Pós-VIF | 60 | 4 Tradicionais |
| **E** | GroupKFold (k=5) | Pós-VIF | 60 | 3 Avançados |

**Modelos Tradicionais:** Linear Regression, Decision Tree, Random Forest, Gradient Boosting

**Modelos Avançados:** XGBoost, LightGBM, CatBoost

**4 Hipóteses Testadas:**
- **H1 (B > A):** VIF melhora desempenho mesmo COM data leakage → ✅ Confirmada (29% melhoria)
- **H2 (A >> D):** Data leakage infla dramaticamente as métricas → ✅ Confirmada (131% inflação)
- **H3 (C, D, E):** R² negativo com validação rigorosa → ✅ Confirmada (todos negativos)
- **H4 (E > D):** Modelos avançados superam tradicionais → ✅ Confirmada (16.8% melhoria)

---

## **❌ NOTEBOOKS OBSOLETOS/RASCUNHOS** (Podem ser ignorados)

- ❌ `notebooks/modeling/comparacao_metodologias_completa_4_cenarios.ipynb` (versão antiga - 4 cenários)
- ❌ `notebooks/modeling/comparacao_metodologias_pedro_vs_joao.ipynb` (versão inicial - 2 cenários)
- ❌ `notebooks/modeling/feature_selection_e_modelagem.ipynb` (exploratório)
- ❌ `notebooks/modeling/modelagem_multiplos_algoritmos.ipynb` (exploratório)
- ❌ `notebooks/modeling/preparar_dados_modelagem.ipynb` (virou script .py)

---

## **📊 FLUXO COMPLETO DE DADOS**

```
data/raw/20230625-processed-{physical|psychological}-qol.csv
(1.373 registros, 88 features)
           ↓
[src/preprocessing/pipeline_imputacao_temporal.py]
           ↓
data/interim/dados_{physical|psychological}_imputado.csv
(2.267 registros, 88 features)
           ↓
[src/preprocessing/adicionar_flags_outliers.py]
           ↓
data/interim/dados_{physical|psychological}_com_flags_outliers.csv
(2.267 registros, 88 features + flags de outliers)
           ↓
[src/feature_engineering/criar_features.py]
           ↓
data/processed/dados_{physical|psychological}_com_features.csv
(2.267 registros, 118 features: 88 originais + 30 derivadas)
           ↓
[src/preprocessing/criar_datasets_pos_vif.py]
           ↓
data/processed/dados_{physical|psychological}_apos_vif.csv
(2.267 registros, 60 features após VIF)
           ↓
[notebooks/modeling/comparacao_metodologias_5_cenarios_completa.ipynb]
           ↓
results/comparacao_5_cenarios/
  ├── comparacao_5_cenarios_completa.csv
  ├── resumo_5_cenarios.csv
  ├── comparacao_rmse_5_cenarios.png
  └── comparacao_r2_groupkfold.png
```

---

## **📁 ESTRUTURA FINAL - ARQUIVOS ESSENCIAIS**

```
TCC/
├── notebooks/
│   ├── exploratory/
│   │   └── ✅ analise_exploratoria_final.ipynb
│   └── modeling/
│       └── ✅ comparacao_metodologias_5_cenarios_completa.ipynb
│
├── src/
│   ├── preprocessing/
│   │   ├── ✅ analise_gaps_temporais.py
│   │   ├── ✅ pipeline_imputacao_temporal.py
│   │   ├── ✅ analise_outliers.py
│   │   ├── ✅ adicionar_flags_outliers.py
│   │   ├── ✅ criar_datasets_pos_vif.py
│   │   └── ✅ preparar_dados_modelagem.py
│   └── feature_engineering/
│       └── ✅ criar_features.py
│
└── data/
    ├── raw/
    │   ├── ✅ 20230625-processed-physical-qol.csv (1.373 registros)
    │   └── ✅ 20230625-processed-psychological-qol.csv (1.373 registros)
    ├── interim/
    │   ├── ✅ dados_physical_imputado.csv (2.267 registros)
    │   ├── ✅ dados_psychological_imputado.csv (2.267 registros)
    │   ├── ✅ dados_physical_com_flags_outliers.csv
    │   └── ✅ dados_psychological_com_flags_outliers.csv
    └── processed/
        ├── ✅ dados_physical_com_features.csv (118 features)
        ├── ✅ dados_psychological_com_features.csv (118 features)
        ├── ✅ dados_physical_apos_vif.csv (60 features)
        └── ✅ dados_psychological_apos_vif.csv (60 features)
```

---

## **📌 ORDEM DE EXECUÇÃO DO PIPELINE**

### Passo a passo para reproduzir o trabalho completo:

```bash
# 1. Análise Exploratória
jupyter notebook notebooks/exploratory/analise_exploratoria_final.ipynb

# 2. Imputação Temporal
python src/preprocessing/pipeline_imputacao_temporal.py

# 3. Detecção de Outliers
python src/preprocessing/adicionar_flags_outliers.py

# 4. Feature Engineering
python src/feature_engineering/criar_features.py

# 5. Redução VIF
python src/preprocessing/criar_datasets_pos_vif.py

# 6. Modelagem Completa
jupyter notebook notebooks/modeling/comparacao_metodologias_5_cenarios_completa.ipynb
```

---

## **📊 RESUMO QUANTITATIVO**

**Total de arquivos essenciais:**
- ✅ **2 notebooks principais** (exploratória + modelagem)
- ✅ **8 scripts Python** (pipeline completo de pré-processamento)
- ✅ **12 datasets** (raw → interim → processed)

**Transformações de dados:**
- Dataset inicial: 1.373 registros × 88 features
- Após imputação: 2.267 registros × 88 features
- Após feature engineering: 2.267 registros × 118 features
- Após VIF: 2.267 registros × 60 features

**Modelos treinados:**
- 4 tradicionais × 5 cenários × 2 domínios = 40 modelos
- 3 avançados × 1 cenário × 2 domínios = 6 modelos
- **Total: 46 modelos treinados**

---

## **🎯 PRINCIPAIS CONTRIBUIÇÕES**

1. **Pipeline completo de pré-processamento** para dados de wearables
2. **30 features derivadas** baseadas em conhecimento de domínio
3. **Comparação sistemática** de estratégias de validação (KFold vs GroupKFold)
4. **Quantificação do data leakage** (131% de inflação nas métricas)
5. **Demonstração de R² negativo** em validação rigorosa
6. **Evidência empírica** de que multicolinearidade importa mesmo com data leakage

---

**Documento gerado automaticamente em:** 30/12/2024
**Versão:** 1.0
**Status:** Pipeline completo validado e documentado
