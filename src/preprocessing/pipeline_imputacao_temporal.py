"""
Pipeline de Imputação Temporal Conservadora
============================================
Estratégia:
1. Forward-fill para gaps <7 dias (comportamento tem inércia)
2. Interpolação linear para gaps 7-30 dias (transição gradual)
3. NÃO imputar gaps >30 dias (marcar como baixa confiança)
4. Criar flags para análise de sensibilidade
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

print("="*80)
print("PIPELINE DE IMPUTAÇÃO TEMPORAL CONSERVADORA")
print("="*80)
print()

# ============================================================================
# 1. CARREGAR DADOS
# ============================================================================

print("1. Carregando dados...")
psych = pd.read_csv('archive (1)/20230625-processed-psychological-qol.csv')
phys = pd.read_csv('archive (1)/20230625-processed-physical-qol.csv')

# Converter datas
psych['date'] = pd.to_datetime(psych['day'], format='%Y%m%d')
phys['date'] = pd.to_datetime(phys['day'], format='%Y%m%d')

print(f"   Registros psicológicos: {len(psych)}")
print(f"   Registros físicos: {len(phys)}")
print(f"   Participantes: {psych['id'].nunique()}")
print()

# ============================================================================
# 2. IDENTIFICAR PARTICIPANTES COM GAPS >30 DIAS
# ============================================================================

print("2. Identificando participantes com gaps >30 dias...")

participantes_baixa_confianca = [
    '2XgOFdSrM0', 'D39iyp2dlm', 'E8jhIJvIVj', 'GXajcvW5pn',
    'HufZzocYQR', 'eJDbYeq7uh', 'nfW9KrEKe7', 'p2Pz4vSJhH', 'pjtTEh7bxf'
]

print(f"   {len(participantes_baixa_confianca)} participantes marcados")
print()

# ============================================================================
# 3. FUNÇÃO DE IMPUTAÇÃO CONSERVADORA
# ============================================================================

def imputar_participante(df_part, colunas_numericas):
    """
    Imputa gaps temporais de um participante usando estratégia conservadora

    Retorna:
    - DataFrame com dados imputados
    - Estatísticas de imputação
    """
    df_part = df_part.sort_values('date').copy()

    # Criar range completo de datas
    data_min = df_part['date'].min()
    data_max = df_part['date'].max()
    datas_completas = pd.date_range(start=data_min, end=data_max, freq='D')

    # Criar DataFrame com todas as datas
    df_completo = pd.DataFrame({'date': datas_completas})
    df_completo = df_completo.merge(df_part, on='date', how='left')

    # Inicializar flags de imputação
    df_completo['flag_original'] = df_completo['id'].notna()
    df_completo['flag_imputado_forward'] = False
    df_completo['flag_imputado_interpolado'] = False

    # Preencher colunas não-numéricas (id, etc)
    df_completo['id'] = df_part['id'].iloc[0]

    # Identificar gaps
    gaps_info = []
    gap_size = 0
    gap_start_idx = None

    for i in range(len(df_completo)):
        if not df_completo.loc[i, 'flag_original']:
            if gap_start_idx is None:
                gap_start_idx = i
            gap_size += 1
        else:
            if gap_start_idx is not None:
                gaps_info.append({
                    'start_idx': gap_start_idx,
                    'end_idx': i - 1,
                    'size': gap_size
                })
                gap_start_idx = None
                gap_size = 0

    # Estatísticas
    stats = {
        'total_dias': len(df_completo),
        'dias_originais': df_completo['flag_original'].sum(),
        'gaps_curtos_imputados': 0,
        'gaps_medios_imputados': 0,
        'gaps_longos_nao_imputados': 0
    }

    # Imputar cada gap
    for gap in gaps_info:
        gap_size = gap['size']
        start_idx = gap['start_idx']
        end_idx = gap['end_idx']

        if gap_size < 7:
            # ESTRATÉGIA 1: Forward-fill para gaps curtos
            if start_idx > 0:
                for col in colunas_numericas:
                    valor_anterior = df_completo.loc[start_idx - 1, col]
                    df_completo.loc[start_idx:end_idx, col] = valor_anterior

                df_completo.loc[start_idx:end_idx, 'flag_imputado_forward'] = True
                stats['gaps_curtos_imputados'] += gap_size

        elif gap_size <= 30:
            # ESTRATÉGIA 2: Interpolação linear para gaps médios
            if start_idx > 0 and end_idx < len(df_completo) - 1:
                for col in colunas_numericas:
                    # Interpolar entre valor anterior e posterior
                    valor_antes = df_completo.loc[start_idx - 1, col]
                    valor_depois = df_completo.loc[end_idx + 1, col]

                    # Criar valores interpolados
                    valores_interpolados = np.linspace(
                        valor_antes,
                        valor_depois,
                        gap_size + 2
                    )[1:-1]  # Excluir extremos

                    df_completo.loc[start_idx:end_idx, col] = valores_interpolados

                df_completo.loc[start_idx:end_idx, 'flag_imputado_interpolado'] = True
                stats['gaps_medios_imputados'] += gap_size

        else:
            # ESTRATÉGIA 3: NÃO imputar gaps longos
            stats['gaps_longos_nao_imputados'] += gap_size

    return df_completo, stats

# ============================================================================
# 4. PROCESSAR TODOS OS PARTICIPANTES
# ============================================================================

print("3. Processando imputação por participante...")
print()

# Identificar colunas numéricas para cada dataset separadamente
colunas_excluir = ['id', 'day', 'date', 'flag_original',
                   'flag_imputado_forward', 'flag_imputado_interpolado']

# Colunas numéricas do dataset psicológico
colunas_numericas_psych = [col for col in psych.columns
                           if col not in colunas_excluir and
                           pd.api.types.is_numeric_dtype(psych[col])]

# Colunas numéricas do dataset físico
colunas_numericas_phys = [col for col in phys.columns
                          if col not in colunas_excluir and
                          pd.api.types.is_numeric_dtype(phys[col])]

print(f"   Colunas a serem imputadas (psych): {len(colunas_numericas_psych)}")
print(f"   Colunas a serem imputadas (phys): {len(colunas_numericas_phys)}")
print()

# DataFrames para armazenar resultados
psych_imputado_list = []
phys_imputado_list = []
estatisticas_por_participante = []

for i, participant_id in enumerate(sorted(psych['id'].unique()), 1):
    # Processar dados psicológicos
    df_psych_part = psych[psych['id'] == participant_id]
    df_psych_imputado, stats_psych = imputar_participante(df_psych_part, colunas_numericas_psych)

    # Processar dados físicos
    df_phys_part = phys[phys['id'] == participant_id]
    df_phys_imputado, stats_phys = imputar_participante(df_phys_part, colunas_numericas_phys)

    # Adicionar flag de baixa confiança
    baixa_confianca = participant_id in participantes_baixa_confianca
    df_psych_imputado['flag_baixa_confianca'] = baixa_confianca
    df_phys_imputado['flag_baixa_confianca'] = baixa_confianca

    # Remover linhas que não foram imputadas (gaps >30 dias)
    df_psych_imputado = df_psych_imputado[
        df_psych_imputado['flag_original'] |
        df_psych_imputado['flag_imputado_forward'] |
        df_psych_imputado['flag_imputado_interpolado']
    ]

    df_phys_imputado = df_phys_imputado[
        df_phys_imputado['flag_original'] |
        df_phys_imputado['flag_imputado_forward'] |
        df_phys_imputado['flag_imputado_interpolado']
    ]

    psych_imputado_list.append(df_psych_imputado)
    phys_imputado_list.append(df_phys_imputado)

    # Estatísticas
    estatisticas_por_participante.append({
        'participant': participant_id,
        'baixa_confianca': baixa_confianca,
        'registros_originais': stats_psych['dias_originais'],
        'registros_finais': len(df_psych_imputado),
        'gaps_curtos_imputados': stats_psych['gaps_curtos_imputados'],
        'gaps_medios_imputados': stats_psych['gaps_medios_imputados'],
        'gaps_longos_nao_imputados': stats_psych['gaps_longos_nao_imputados']
    })

    if i % 10 == 0:
        print(f"   Processados {i}/35 participantes...")

print(f"   ✓ Todos os 35 participantes processados")
print()

# ============================================================================
# 5. CONSOLIDAR RESULTADOS
# ============================================================================

print("4. Consolidando resultados...")

psych_imputado = pd.concat(psych_imputado_list, ignore_index=True)
phys_imputado = pd.concat(phys_imputado_list, ignore_index=True)

# Recriar coluna 'day' no formato YYYYMMDD
psych_imputado['day'] = psych_imputado['date'].dt.strftime('%Y%m%d').astype(int)
phys_imputado['day'] = phys_imputado['date'].dt.strftime('%Y%m%d').astype(int)

# Remover coluna date temporária
psych_imputado = psych_imputado.drop('date', axis=1)
phys_imputado = phys_imputado.drop('date', axis=1)

# Reorganizar colunas (flags no final)
colunas_originais_psych = [col for col in psych.columns if col != 'date']
colunas_originais_phys = [col for col in phys.columns if col != 'date']
colunas_flags = ['flag_original', 'flag_imputado_forward',
                 'flag_imputado_interpolado', 'flag_baixa_confianca']

colunas_ordem_psych = colunas_originais_psych + colunas_flags
colunas_ordem_phys = colunas_originais_phys + colunas_flags

psych_imputado = psych_imputado[colunas_ordem_psych]
phys_imputado = phys_imputado[colunas_ordem_phys]

print(f"   ✓ Consolidação concluída")
print()

# ============================================================================
# 6. ESTATÍSTICAS FINAIS
# ============================================================================

print("="*80)
print("ESTATÍSTICAS FINAIS")
print("="*80)
print()

print("DATASET PSICOLÓGICO:")
print(f"   Registros originais: {len(psych)}")
print(f"   Registros finais: {len(psych_imputado)}")
print(f"   Crescimento: +{len(psych_imputado) - len(psych)} registros (+{(len(psych_imputado)/len(psych) - 1)*100:.1f}%)")
print()

total_original = psych_imputado['flag_original'].sum()
total_forward = psych_imputado['flag_imputado_forward'].sum()
total_interpolado = psych_imputado['flag_imputado_interpolado'].sum()

print(f"   Dados originais: {total_original} ({total_original/len(psych_imputado)*100:.1f}%)")
print(f"   Imputados (forward-fill): {total_forward} ({total_forward/len(psych_imputado)*100:.1f}%)")
print(f"   Imputados (interpolação): {total_interpolado} ({total_interpolado/len(psych_imputado)*100:.1f}%)")
print(f"   Total imputados: {total_forward + total_interpolado} ({(total_forward + total_interpolado)/len(psych_imputado)*100:.1f}%)")
print()

participantes_bc = psych_imputado['flag_baixa_confianca'].sum() // len(psych_imputado) * 35  # aproximado
print(f"   Participantes com flag 'baixa_confianca': {len(participantes_baixa_confianca)}/35")
print()

# ============================================================================
# 7. SALVAR RESULTADOS
# ============================================================================

print("="*80)
print("SALVANDO RESULTADOS")
print("="*80)
print()

# Salvar datasets imputados
psych_imputado.to_csv('dados_psychological_imputado.csv', index=False)
print("✓ Salvo: dados_psychological_imputado.csv")

phys_imputado.to_csv('dados_physical_imputado.csv', index=False)
print("✓ Salvo: dados_physical_imputado.csv")

# Salvar estatísticas por participante
df_stats = pd.DataFrame(estatisticas_por_participante)
df_stats.to_csv('estatisticas_imputacao_por_participante.csv', index=False)
print("✓ Salvo: estatisticas_imputacao_por_participante.csv")

# Salvar relatório JSON
relatorio = {
    'data_processamento': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'estrategia': {
        'gaps_curtos': 'forward-fill (<7 dias)',
        'gaps_medios': 'interpolação linear (7-30 dias)',
        'gaps_longos': 'não imputados (>30 dias)'
    },
    'resultados': {
        'registros_originais': int(len(psych)),
        'registros_finais': int(len(psych_imputado)),
        'crescimento_percentual': round((len(psych_imputado)/len(psych) - 1)*100, 1),
        'dados_originais': int(total_original),
        'dados_imputados_forward': int(total_forward),
        'dados_imputados_interpolados': int(total_interpolado),
        'percentual_imputado_total': round((total_forward + total_interpolado)/len(psych_imputado)*100, 1)
    },
    'flags': {
        'flag_original': 'Dados originais do dataset',
        'flag_imputado_forward': 'Imputado via forward-fill (gaps <7 dias)',
        'flag_imputado_interpolado': 'Imputado via interpolação (gaps 7-30 dias)',
        'flag_baixa_confianca': 'Participante com gaps >30 dias'
    },
    'participantes_baixa_confianca': participantes_baixa_confianca
}

with open('relatorio_imputacao.json', 'w', encoding='utf-8') as f:
    json.dump(relatorio, f, indent=2, ensure_ascii=False)
print("✓ Salvo: relatorio_imputacao.json")

print()
print("="*80)
print("PIPELINE CONCLUÍDO COM SUCESSO!")
print("="*80)
print()
print("Próximos passos:")
print("1. Verificar qualidade dos dados imputados")
print("2. Análise de sensibilidade (com/sem dados imputados)")
print("3. Feature engineering com dados completos")
print("="*80)
