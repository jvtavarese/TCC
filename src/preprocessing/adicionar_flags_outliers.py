"""
Adicionar Flags de Outliers - Fase 1.b
========================================
Estratégia Conservadora: MANTER outliers, mas marcar para análise de sensibilidade
"""

import pandas as pd
import numpy as np
import json

print("="*80)
print("ADICIONANDO FLAGS DE OUTLIERS - ESTRATÉGIA CONSERVADORA")
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
# 2. FUNÇÃO PARA DETECTAR OUTLIERS (IQR METHOD)
# ============================================================================

def detectar_outliers_iqr(series, multiplicador=1.5):
    """
    Detecta outliers usando método IQR (Interquartile Range)
    Retorna máscara booleana de outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - multiplicador * IQR
    limite_superior = Q3 + multiplicador * IQR

    outliers = (series < limite_inferior) | (series > limite_superior)

    return outliers

# ============================================================================
# 3. IDENTIFICAR VARIÁVEIS PARA FLAGGING
# ============================================================================

print("2. Identificando variáveis para flagging...")
print()

# Variáveis de interesse primário (comunicação)
variaveis_comunicacao = [
    'incomingcalls', 'outgoingcalls', 'missedcalls', 'rejectedcalls', 'blockedcalls',
    'incomingcallsaverageduration', 'outgoingcallsaverageduration'
]

# Variáveis fisiológicas importantes
variaveis_fisiologicas = [
    'max_hr', 'median_nni', 'sdsd', 'rmssd', 'mean_nni', 'sdnn'
]

# Variáveis de sono
variaveis_sono = [
    'lightsleep', 'deepsleep', 'remsleep', 'awakesleep'
]

# Variáveis de atividade física
variaveis_atividade = [
    'steps', 'calories', 'running'
]

# Combinar todas as variáveis de interesse
variaveis_interesse = list(set(
    variaveis_comunicacao + variaveis_fisiologicas +
    variaveis_sono + variaveis_atividade
))

# Verificar quais existem nos datasets
variaveis_psych = [v for v in variaveis_interesse if v in psych.columns]
variaveis_phys = [v for v in variaveis_interesse if v in phys.columns]

print(f"   Variáveis para flagging (psych): {len(variaveis_psych)}")
print(f"   Variáveis para flagging (phys): {len(variaveis_phys)}")
print()

# ============================================================================
# 4. CRIAR FLAGS DE OUTLIERS
# ============================================================================

print("3. Criando flags de outliers...")
print()

# Inicializar flag geral (qualquer variável de interesse tem outlier)
psych['flag_tem_outlier'] = False
phys['flag_tem_outlier'] = False

# Contador de variáveis por registro
psych['n_variaveis_outlier'] = 0
phys['n_variaveis_outlier'] = 0

# Flags específicas para variáveis críticas
for var in variaveis_psych:
    flag_nome = f'flag_outlier_{var}'
    outliers = detectar_outliers_iqr(psych[var])
    psych[flag_nome] = outliers

    # Atualizar flag geral e contador
    psych['flag_tem_outlier'] = psych['flag_tem_outlier'] | outliers
    psych['n_variaveis_outlier'] += outliers.astype(int)

    n_outliers = outliers.sum()
    perc = (n_outliers / len(psych)) * 100
    print(f"   {var} (psych): {n_outliers} outliers ({perc:.1f}%)")

print()

for var in variaveis_phys:
    flag_nome = f'flag_outlier_{var}'
    outliers = detectar_outliers_iqr(phys[var])
    phys[flag_nome] = outliers

    # Atualizar flag geral e contador
    phys['flag_tem_outlier'] = phys['flag_tem_outlier'] | outliers
    phys['n_variaveis_outlier'] += outliers.astype(int)

    n_outliers = outliers.sum()
    perc = (n_outliers / len(phys)) * 100
    print(f"   {var} (phys): {n_outliers} outliers ({perc:.1f}%)")

print()

# ============================================================================
# 5. ESTATÍSTICAS DAS FLAGS
# ============================================================================

print("="*80)
print("ESTATÍSTICAS DAS FLAGS DE OUTLIERS")
print("="*80)
print()

print("DATASET PSICOLÓGICO:")
print(f"   Registros com algum outlier: {psych['flag_tem_outlier'].sum()} ({psych['flag_tem_outlier'].sum()/len(psych)*100:.1f}%)")
print(f"   Média de variáveis outlier por registro: {psych['n_variaveis_outlier'].mean():.2f}")
print(f"   Máximo de variáveis outlier em um registro: {psych['n_variaveis_outlier'].max()}")
print()

# Distribuição do número de variáveis outlier
print("   Distribuição de n_variaveis_outlier:")
for n in range(0, min(6, psych['n_variaveis_outlier'].max() + 1)):
    count = (psych['n_variaveis_outlier'] == n).sum()
    print(f"     {n} variáveis: {count} registros ({count/len(psych)*100:.1f}%)")
if psych['n_variaveis_outlier'].max() >= 6:
    count = (psych['n_variaveis_outlier'] >= 6).sum()
    print(f"     6+ variáveis: {count} registros ({count/len(psych)*100:.1f}%)")
print()

print("DATASET FÍSICO:")
print(f"   Registros com algum outlier: {phys['flag_tem_outlier'].sum()} ({phys['flag_tem_outlier'].sum()/len(phys)*100:.1f}%)")
print(f"   Média de variáveis outlier por registro: {phys['n_variaveis_outlier'].mean():.2f}")
print(f"   Máximo de variáveis outlier em um registro: {phys['n_variaveis_outlier'].max()}")
print()

# Distribuição do número de variáveis outlier
print("   Distribuição de n_variaveis_outlier:")
for n in range(0, min(6, phys['n_variaveis_outlier'].max() + 1)):
    count = (phys['n_variaveis_outlier'] == n).sum()
    print(f"     {n} variáveis: {count} registros ({count/len(phys)*100:.1f}%)")
if phys['n_variaveis_outlier'].max() >= 6:
    count = (phys['n_variaveis_outlier'] >= 6).sum()
    print(f"     6+ variáveis: {count} registros ({count/len(phys)*100:.1f}%)")
print()

# ============================================================================
# 6. ANÁLISE POR PARTICIPANTE
# ============================================================================

print("="*80)
print("ANÁLISE DE OUTLIERS POR PARTICIPANTE")
print("="*80)
print()

participantes_stats = []

for participant_id in sorted(psych['id'].unique()):
    df_part_psych = psych[psych['id'] == participant_id]
    df_part_phys = phys[phys['id'] == participant_id]

    stats = {
        'participant': participant_id,
        'total_registros': len(df_part_psych),
        'registros_com_outlier': df_part_psych['flag_tem_outlier'].sum(),
        'percentual_outlier': (df_part_psych['flag_tem_outlier'].sum() / len(df_part_psych)) * 100,
        'media_variaveis_outlier': df_part_psych['n_variaveis_outlier'].mean()
    }

    participantes_stats.append(stats)

df_participantes = pd.DataFrame(participantes_stats)
df_participantes = df_participantes.sort_values('percentual_outlier', ascending=False)

print("TOP 10 PARTICIPANTES COM MAIS OUTLIERS:")
print(df_participantes[['participant', 'total_registros', 'registros_com_outlier',
                        'percentual_outlier']].head(10).to_string(index=False))
print()

print("TOP 10 PARTICIPANTES COM MENOS OUTLIERS:")
print(df_participantes[['participant', 'total_registros', 'registros_com_outlier',
                        'percentual_outlier']].tail(10).to_string(index=False))
print()

# ============================================================================
# 7. SALVAR DATASETS COM FLAGS
# ============================================================================

print("="*80)
print("SALVANDO DATASETS COM FLAGS")
print("="*80)
print()

# Salvar datasets atualizados
psych.to_csv('dados_psychological_com_flags_outliers.csv', index=False)
print("✓ Salvo: dados_psychological_com_flags_outliers.csv")
print(f"  Colunas: {len(psych.columns)} (originais + {len(variaveis_psych) + 2} flags)")

phys.to_csv('dados_physical_com_flags_outliers.csv', index=False)
print("✓ Salvo: dados_physical_com_flags_outliers.csv")
print(f"  Colunas: {len(phys.columns)} (originais + {len(variaveis_phys) + 2} flags)")
print()

# Salvar estatísticas por participante
df_participantes.to_csv('outliers_por_participante.csv', index=False)
print("✓ Salvo: outliers_por_participante.csv")
print()

# Salvar relatório
relatorio = {
    'estrategia': 'Conservadora - Outliers mantidos, apenas flagged',
    'metodo_deteccao': 'IQR (Interquartile Range) com multiplicador 1.5',
    'variaveis_flagged': {
        'psychological': variaveis_psych,
        'physical': variaveis_phys
    },
    'estatisticas': {
        'psychological': {
            'registros_com_outlier': int(psych['flag_tem_outlier'].sum()),
            'percentual_com_outlier': round((psych['flag_tem_outlier'].sum() / len(psych)) * 100, 1),
            'media_variaveis_outlier': round(psych['n_variaveis_outlier'].mean(), 2)
        },
        'physical': {
            'registros_com_outlier': int(phys['flag_tem_outlier'].sum()),
            'percentual_com_outlier': round((phys['flag_tem_outlier'].sum() / len(phys)) * 100, 1),
            'media_variaveis_outlier': round(phys['n_variaveis_outlier'].mean(), 2)
        }
    },
    'flags_criadas': {
        'flag_tem_outlier': 'TRUE se registro tem outlier em qualquer variável de interesse',
        'n_variaveis_outlier': 'Número de variáveis com outlier neste registro',
        'flag_outlier_[variavel]': 'TRUE se variável específica tem outlier neste registro'
    },
    'justificativa': 'Outliers mantidos pois representam padrões comportamentais válidos (distribuições zero-inflated) e não erros de medição'
}

with open('relatorio_flags_outliers.json', 'w', encoding='utf-8') as f:
    json.dump(relatorio, f, indent=2, ensure_ascii=False)
print("✓ Salvo: relatorio_flags_outliers.json")
print()

# ============================================================================
# 8. GUIA DE USO
# ============================================================================

print("="*80)
print("GUIA DE USO DAS FLAGS")
print("="*80)
print()

print("As flags foram criadas para análise de sensibilidade.")
print("Valores originais foram MANTIDOS (estratégia conservadora).")
print()
print("COMO USAR:")
print()
print("1. Análise COMPLETA (padrão recomendado):")
print("   df = pd.read_csv('dados_psychological_com_flags_outliers.csv')")
print("   # Usar todos os dados, incluindo outliers")
print()
print("2. Análise SEM OUTLIERS (teste de robustez):")
print("   df = pd.read_csv('dados_psychological_com_flags_outliers.csv')")
print("   df_sem_outliers = df[~df['flag_tem_outlier']]")
print("   # Remove registros com qualquer outlier")
print()
print("3. Análise SEM OUTLIERS EXTREMOS (conservadora):")
print("   df = pd.read_csv('dados_psychological_com_flags_outliers.csv')")
print("   df_conservador = df[df['n_variaveis_outlier'] <= 2]")
print("   # Mantém registros com até 2 variáveis outlier")
print()
print("4. Análise excluindo variável específica (ex: chamadas):")
print("   df = pd.read_csv('dados_psychological_com_flags_outliers.csv')")
print("   df_sem_calls = df[~df['flag_outlier_incomingcalls'] & ~df['flag_outlier_outgoingcalls']]")
print()

print("="*80)
print("CONCLUÍDO COM SUCESSO!")
print("="*80)
