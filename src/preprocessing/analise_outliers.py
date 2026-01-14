"""
Análise de Outliers - Fase 1.b
================================
Objetivo: Identificar e caracterizar outliers, especialmente em incomingcalls e outgoingcalls
"""

import pandas as pd
import numpy as np
import json

print("="*80)
print("ANÁLISE DE OUTLIERS - FASE 1.b")
print("="*80)
print()

# ============================================================================
# 1. CARREGAR DADOS IMPUTADOS
# ============================================================================

print("1. Carregando dados imputados...")
psych = pd.read_csv('dados_psychological_imputado.csv')
phys = pd.read_csv('dados_physical_imputado.csv')

print(f"   Registros psicológicos: {len(psych)}")
print(f"   Registros físicos: {len(phys)}")
print()

# ============================================================================
# 2. IDENTIFICAR COLUNAS NUMÉRICAS
# ============================================================================

# Excluir colunas não-numéricas e flags
colunas_excluir = ['id', 'day', 'flag_original', 'flag_imputado_forward',
                   'flag_imputado_interpolado', 'flag_baixa_confianca']

colunas_numericas_psych = [col for col in psych.columns
                           if col not in colunas_excluir and
                           pd.api.types.is_numeric_dtype(psych[col])]

colunas_numericas_phys = [col for col in phys.columns
                          if col not in colunas_excluir and
                          pd.api.types.is_numeric_dtype(phys[col])]

print(f"2. Colunas numéricas identificadas:")
print(f"   Psicológico: {len(colunas_numericas_psych)} colunas")
print(f"   Físico: {len(colunas_numericas_phys)} colunas")
print()

# ============================================================================
# 3. FUNÇÃO PARA DETECTAR OUTLIERS (IQR METHOD)
# ============================================================================

def detectar_outliers_iqr(series, multiplicador=1.5):
    """
    Detecta outliers usando método IQR (Interquartile Range)

    Outliers são valores que estão:
    - Abaixo de Q1 - multiplicador * IQR
    - Acima de Q3 + multiplicador * IQR

    Multiplicador padrão: 1.5 (outliers moderados)
    Multiplicador 3.0: outliers extremos
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - multiplicador * IQR
    limite_superior = Q3 + multiplicador * IQR

    outliers = (series < limite_inferior) | (series > limite_superior)

    return {
        'n_outliers': outliers.sum(),
        'percentual': (outliers.sum() / len(series)) * 100,
        'limite_inferior': limite_inferior,
        'limite_superior': limite_superior,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'min_valor': series.min(),
        'max_valor': series.max(),
        'media': series.mean(),
        'mediana': series.median(),
        'desvio_padrao': series.std()
    }

# ============================================================================
# 4. ANALISAR OUTLIERS EM TODAS AS VARIÁVEIS
# ============================================================================

print("3. Analisando outliers em todas as variáveis...")
print()

resultados_psych = {}
resultados_phys = {}

# Analisar dataset psicológico
for col in colunas_numericas_psych:
    resultados_psych[col] = detectar_outliers_iqr(psych[col])

# Analisar dataset físico
for col in colunas_numericas_phys:
    resultados_phys[col] = detectar_outliers_iqr(phys[col])

# ============================================================================
# 5. IDENTIFICAR VARIÁVEIS COM MAIS DE 5% DE OUTLIERS
# ============================================================================

print("="*80)
print("VARIÁVEIS COM >5% DE OUTLIERS (CRÍTICAS)")
print("="*80)
print()

variaveis_criticas_psych = {col: info for col, info in resultados_psych.items()
                             if info['percentual'] > 5}
variaveis_criticas_phys = {col: info for col, info in resultados_phys.items()
                            if info['percentual'] > 5}

print("DATASET PSICOLÓGICO:")
if variaveis_criticas_psych:
    for col, info in sorted(variaveis_criticas_psych.items(),
                            key=lambda x: x[1]['percentual'], reverse=True):
        print(f"\n{col}:")
        print(f"  • Outliers: {info['n_outliers']} ({info['percentual']:.1f}%)")
        print(f"  • Valores: min={info['min_valor']:.2f}, max={info['max_valor']:.2f}")
        print(f"  • Limites IQR: [{info['limite_inferior']:.2f}, {info['limite_superior']:.2f}]")
        print(f"  • Estatísticas: média={info['media']:.2f}, mediana={info['mediana']:.2f}, std={info['desvio_padrao']:.2f}")
else:
    print("  ✓ Nenhuma variável com >5% outliers")
print()

print("DATASET FÍSICO:")
if variaveis_criticas_phys:
    for col, info in sorted(variaveis_criticas_phys.items(),
                            key=lambda x: x[1]['percentual'], reverse=True):
        print(f"\n{col}:")
        print(f"  • Outliers: {info['n_outliers']} ({info['percentual']:.1f}%)")
        print(f"  • Valores: min={info['min_valor']:.2f}, max={info['max_valor']:.2f}")
        print(f"  • Limites IQR: [{info['limite_inferior']:.2f}, {info['limite_superior']:.2f}]")
        print(f"  • Estatísticas: média={info['media']:.2f}, mediana={info['mediana']:.2f}, std={info['desvio_padrao']:.2f}")
else:
    print("  ✓ Nenhuma variável com >5% outliers")
print()

# ============================================================================
# 6. ANÁLISE ESPECÍFICA: INCOMINGCALLS E OUTGOINGCALLS
# ============================================================================

print("="*80)
print("ANÁLISE DETALHADA: INCOMINGCALLS E OUTGOINGCALLS")
print("="*80)
print()

# Verificar se existem essas colunas
variaveis_comunicacao = []
for var in ['incomingcalls', 'outgoingcalls']:
    if var in colunas_numericas_psych:
        variaveis_comunicacao.append(('psych', var))
    if var in colunas_numericas_phys:
        variaveis_comunicacao.append(('phys', var))

if variaveis_comunicacao:
    for dataset_tipo, var in variaveis_comunicacao:
        df = psych if dataset_tipo == 'psych' else phys

        print(f"{var.upper()} (dataset {dataset_tipo}):")
        print("-"*60)

        info = resultados_psych[var] if dataset_tipo == 'psych' else resultados_phys[var]

        print(f"Estatísticas básicas:")
        print(f"  • Média: {info['media']:.2f}")
        print(f"  • Mediana: {info['mediana']:.2f}")
        print(f"  • Desvio padrão: {info['desvio_padrao']:.2f}")
        print(f"  • Mínimo: {info['min_valor']:.2f}")
        print(f"  • Máximo: {info['max_valor']:.2f}")
        print()

        print(f"Análise de outliers (IQR):")
        print(f"  • Q1 (25%): {info['Q1']:.2f}")
        print(f"  • Q3 (75%): {info['Q3']:.2f}")
        print(f"  • IQR: {info['IQR']:.2f}")
        print(f"  • Limite inferior: {info['limite_inferior']:.2f}")
        print(f"  • Limite superior: {info['limite_superior']:.2f}")
        print(f"  • Outliers: {info['n_outliers']} ({info['percentual']:.1f}%)")
        print()

        # Distribuição dos valores
        print(f"Distribuição por percentil:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            valor = df[var].quantile(p/100)
            print(f"  • P{p}: {valor:.2f}")
        print()

        # Verificar zeros
        n_zeros = (df[var] == 0).sum()
        print(f"Valores zero: {n_zeros} ({n_zeros/len(df)*100:.1f}%)")
        print()

        # Participantes com valores extremos
        outlier_mask = (df[var] < info['limite_inferior']) | (df[var] > info['limite_superior'])
        participantes_com_outliers = df[outlier_mask]['id'].unique()
        print(f"Participantes com outliers: {len(participantes_com_outliers)}/35")
        print()

else:
    print("  ⚠️ Variáveis 'incomingcalls' ou 'outgoingcalls' não encontradas")
    print()

# ============================================================================
# 7. RESUMO GERAL
# ============================================================================

print("="*80)
print("RESUMO GERAL DE OUTLIERS")
print("="*80)
print()

# Contar variáveis por faixa de outliers
def categorizar_outliers(resultados):
    categorias = {
        'criticas': [],      # >5%
        'moderadas': [],     # 2-5%
        'baixas': [],        # 0.5-2%
        'minimas': []        # <0.5%
    }

    for col, info in resultados.items():
        perc = info['percentual']
        if perc > 5:
            categorias['criticas'].append((col, perc))
        elif perc > 2:
            categorias['moderadas'].append((col, perc))
        elif perc > 0.5:
            categorias['baixas'].append((col, perc))
        else:
            categorias['minimas'].append((col, perc))

    return categorias

cat_psych = categorizar_outliers(resultados_psych)
cat_phys = categorizar_outliers(resultados_phys)

print("DATASET PSICOLÓGICO:")
print(f"  • Variáveis críticas (>5% outliers): {len(cat_psych['criticas'])}")
if cat_psych['criticas']:
    for col, perc in sorted(cat_psych['criticas'], key=lambda x: x[1], reverse=True):
        print(f"    - {col}: {perc:.1f}%")
print(f"  • Variáveis moderadas (2-5% outliers): {len(cat_psych['moderadas'])}")
print(f"  • Variáveis baixas (0.5-2% outliers): {len(cat_psych['baixas'])}")
print(f"  • Variáveis mínimas (<0.5% outliers): {len(cat_psych['minimas'])}")
print()

print("DATASET FÍSICO:")
print(f"  • Variáveis críticas (>5% outliers): {len(cat_phys['criticas'])}")
if cat_phys['criticas']:
    for col, perc in sorted(cat_phys['criticas'], key=lambda x: x[1], reverse=True):
        print(f"    - {col}: {perc:.1f}%")
print(f"  • Variáveis moderadas (2-5% outliers): {len(cat_phys['moderadas'])}")
print(f"  • Variáveis baixas (0.5-2% outliers): {len(cat_phys['baixas'])}")
print(f"  • Variáveis mínimas (<0.5% outliers): {len(cat_phys['minimas'])}")
print()

# ============================================================================
# 8. SALVAR RESULTADOS
# ============================================================================

print("="*80)
print("SALVANDO RESULTADOS")
print("="*80)
print()

# Criar DataFrame com resumo
resumo_data = []
for col, info in resultados_psych.items():
    resumo_data.append({
        'dataset': 'psychological',
        'variavel': col,
        'n_outliers': info['n_outliers'],
        'percentual_outliers': round(info['percentual'], 2),
        'min': round(info['min_valor'], 2),
        'max': round(info['max_valor'], 2),
        'media': round(info['media'], 2),
        'mediana': round(info['mediana'], 2),
        'desvio_padrao': round(info['desvio_padrao'], 2),
        'limite_inferior_IQR': round(info['limite_inferior'], 2),
        'limite_superior_IQR': round(info['limite_superior'], 2)
    })

for col, info in resultados_phys.items():
    resumo_data.append({
        'dataset': 'physical',
        'variavel': col,
        'n_outliers': info['n_outliers'],
        'percentual_outliers': round(info['percentual'], 2),
        'min': round(info['min_valor'], 2),
        'max': round(info['max_valor'], 2),
        'media': round(info['media'], 2),
        'mediana': round(info['mediana'], 2),
        'desvio_padrao': round(info['desvio_padrao'], 2),
        'limite_inferior_IQR': round(info['limite_inferior'], 2),
        'limite_superior_IQR': round(info['limite_superior'], 2)
    })

df_resumo = pd.DataFrame(resumo_data)
df_resumo = df_resumo.sort_values('percentual_outliers', ascending=False)
df_resumo.to_csv('analise_outliers_detalhada.csv', index=False)
print("✓ Salvo: analise_outliers_detalhada.csv")

# Salvar JSON
relatorio = {
    'resumo': {
        'psychological': {
            'variaveis_criticas': len(cat_psych['criticas']),
            'variaveis_moderadas': len(cat_psych['moderadas']),
            'variaveis_baixas': len(cat_psych['baixas']),
            'variaveis_minimas': len(cat_psych['minimas']),
            'lista_criticas': [col for col, _ in cat_psych['criticas']]
        },
        'physical': {
            'variaveis_criticas': len(cat_phys['criticas']),
            'variaveis_moderadas': len(cat_phys['moderadas']),
            'variaveis_baixas': len(cat_phys['baixas']),
            'variaveis_minimas': len(cat_phys['minimas']),
            'lista_criticas': [col for col, _ in cat_phys['criticas']]
        }
    },
    'metodo': 'IQR (Interquartile Range) com multiplicador 1.5',
    'criterio_critico': 'Variáveis com >5% de outliers'
}

with open('relatorio_outliers.json', 'w', encoding='utf-8') as f:
    json.dump(relatorio, f, indent=2, ensure_ascii=False)
print("✓ Salvo: relatorio_outliers.json")

print()
print("="*80)
print("ANÁLISE CONCLUÍDA!")
print("="*80)
