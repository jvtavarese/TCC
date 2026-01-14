"""
Análise Detalhada de Gaps Temporais
====================================
Objetivo: Entender a distribuição e padrões dos gaps antes de implementar imputação
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Carregar dados
print("Carregando dados...")
psych = pd.read_csv('archive (1)/20230625-processed-psychological-qol.csv')
phys = pd.read_csv('archive (1)/20230625-processed-physical-qol.csv')

print(f"Registros psicológicos: {len(psych)}")
print(f"Registros físicos: {len(phys)}")
print(f"Participantes únicos: {psych['id'].nunique()}")
print()

# Converter data para datetime (formato YYYYMMDD como inteiro)
psych['date'] = pd.to_datetime(psych['day'], format='%Y%m%d')
phys['date'] = pd.to_datetime(phys['day'], format='%Y%m%d')

# Função para analisar gaps de um participante
def analisar_gaps_participante(df_participante):
    """Analisa gaps temporais de um participante"""
    df_sorted = df_participante.sort_values('date')
    dates = df_sorted['date'].values

    gaps = []
    for i in range(1, len(dates)):
        diff = (dates[i] - dates[i-1]) / np.timedelta64(1, 'D')  # em dias
        if diff > 1:  # se não é dia consecutivo
            gaps.append({
                'data_inicio': dates[i-1],
                'data_fim': dates[i],
                'dias': int(diff)
            })

    return gaps

# Analisar gaps por participante
print("="*80)
print("ANÁLISE DE GAPS POR PARTICIPANTE")
print("="*80)
print()

resultados = []
gaps_curtos = []  # <7 dias
gaps_medios = []  # 7-30 dias
gaps_longos = []  # >30 dias

participantes_gap_longo = []

for participant in sorted(psych['id'].unique()):
    df_part = psych[psych['id'] == participant]
    gaps = analisar_gaps_participante(df_part)

    total_registros = len(df_part)
    total_gaps = len(gaps)

    if gaps:
        dias_gaps = [g['dias'] for g in gaps]
        max_gap = max(dias_gaps)
        media_gap = np.mean(dias_gaps)

        # Classificar gaps
        curtos = sum(1 for d in dias_gaps if d < 7)
        medios = sum(1 for d in dias_gaps if 7 <= d <= 30)
        longos = sum(1 for d in dias_gaps if d > 30)

        gaps_curtos.extend([d for d in dias_gaps if d < 7])
        gaps_medios.extend([d for d in dias_gaps if 7 <= d <= 30])
        gaps_longos.extend([d for d in dias_gaps if d > 30])

        if longos > 0:
            participantes_gap_longo.append(participant)
    else:
        max_gap = 0
        media_gap = 0
        curtos = 0
        medios = 0
        longos = 0

    resultado = {
        'participant': participant,
        'total_registros': total_registros,
        'total_gaps': total_gaps,
        'gaps_curtos_(<7d)': curtos,
        'gaps_medios_(7-30d)': medios,
        'gaps_longos_(>30d)': longos,
        'maior_gap_dias': max_gap,
        'media_gap_dias': round(media_gap, 1) if media_gap > 0 else 0,
        'periodo_total_dias': (df_part['date'].max() - df_part['date'].min()).days
    }

    resultados.append(resultado)

# Criar DataFrame de resultados
df_resultados = pd.DataFrame(resultados)

# Ordenar por maior gap
df_resultados = df_resultados.sort_values('maior_gap_dias', ascending=False)

print("TOP 15 PARTICIPANTES COM MAIORES GAPS:")
print("-"*80)
print(df_resultados[['participant', 'total_registros', 'total_gaps',
                      'gaps_curtos_(<7d)', 'gaps_medios_(7-30d)', 'gaps_longos_(>30d)',
                      'maior_gap_dias']].head(15).to_string(index=False))
print()

# Estatísticas gerais
print("="*80)
print("ESTATÍSTICAS GERAIS DOS GAPS")
print("="*80)
print()

total_gaps_sistema = len(gaps_curtos) + len(gaps_medios) + len(gaps_longos)

print(f"Total de gaps identificados: {total_gaps_sistema}")

if total_gaps_sistema > 0:
    print(f"  • Gaps curtos (<7 dias): {len(gaps_curtos)} ({len(gaps_curtos)/total_gaps_sistema*100:.1f}%)")
    print(f"  • Gaps médios (7-30 dias): {len(gaps_medios)} ({len(gaps_medios)/total_gaps_sistema*100:.1f}%)")
    print(f"  • Gaps longos (>30 dias): {len(gaps_longos)} ({len(gaps_longos)/total_gaps_sistema*100:.1f}%)")
else:
    print("  ✓ Nenhum gap temporal identificado nos dados!")
print()

if gaps_curtos:
    print(f"GAPS CURTOS (<7 dias):")
    print(f"  • Média: {np.mean(gaps_curtos):.1f} dias")
    print(f"  • Mediana: {np.median(gaps_curtos):.1f} dias")
    print(f"  • Mínimo: {min(gaps_curtos)} dias")
    print(f"  • Máximo: {max(gaps_curtos)} dias")
    print()

if gaps_medios:
    print(f"GAPS MÉDIOS (7-30 dias):")
    print(f"  • Média: {np.mean(gaps_medios):.1f} dias")
    print(f"  • Mediana: {np.median(gaps_medios):.1f} dias")
    print(f"  • Mínimo: {min(gaps_medios)} dias")
    print(f"  • Máximo: {max(gaps_medios)} dias")
    print()

if gaps_longos:
    print(f"GAPS LONGOS (>30 dias):")
    print(f"  • Média: {np.mean(gaps_longos):.1f} dias")
    print(f"  • Mediana: {np.median(gaps_longos):.1f} dias")
    print(f"  • Mínimo: {min(gaps_longos)} dias")
    print(f"  • Máximo: {max(gaps_longos)} dias")
    print()

# Análise dos 9 participantes com gaps >30 dias
print("="*80)
print(f"ANÁLISE DOS {len(participantes_gap_longo)} PARTICIPANTES COM GAPS >30 DIAS")
print("="*80)
print()

for participant in participantes_gap_longo:
    df_part = psych[psych['id'] == participant]
    gaps = analisar_gaps_participante(df_part)

    print(f"Participante: {participant}")
    print(f"  • Total de registros: {len(df_part)}")
    print(f"  • Período: {df_part['date'].min().strftime('%Y-%m-%d')} a {df_part['date'].max().strftime('%Y-%m-%d')}")
    print(f"  • Total de gaps: {len(gaps)}")

    # Mostrar gaps longos
    gaps_longos_part = [g for g in gaps if g['dias'] > 30]
    for i, gap in enumerate(gaps_longos_part, 1):
        print(f"    Gap #{i}: {gap['dias']} dias (de {pd.Timestamp(gap['data_inicio']).strftime('%Y-%m-%d')} até {pd.Timestamp(gap['data_fim']).strftime('%Y-%m-%d')})")

    # Ver se é super participante
    if len(df_part) >= 100:
        print(f"  ⭐ SUPER PARTICIPANTE (≥100 registros)")

    print()

# Calcular impacto da imputação
print("="*80)
print("IMPACTO DA ESTRATÉGIA CONSERVADORA")
print("="*80)
print()

dias_totais_gaps_curtos = sum(gaps_curtos)
dias_totais_gaps_medios = sum(gaps_medios)
dias_totais_gaps_longos = sum(gaps_longos)

registros_originais = len(psych)
registros_apos_imputacao_curta = registros_originais + dias_totais_gaps_curtos
registros_apos_imputacao_total = registros_originais + dias_totais_gaps_curtos + dias_totais_gaps_medios

print(f"Registros originais: {registros_originais}")
print()
print(f"Após forward-fill (<7 dias):")
print(f"  • Dias a imputar: {dias_totais_gaps_curtos}")
print(f"  • Total registros: {registros_apos_imputacao_curta}")
print(f"  • % dados imputados: {dias_totais_gaps_curtos/registros_apos_imputacao_curta*100:.1f}%")
print()
print(f"Após interpolação (7-30 dias):")
print(f"  • Dias a imputar: {dias_totais_gaps_medios}")
print(f"  • Total registros: {registros_apos_imputacao_total}")
print(f"  • % dados imputados: {(dias_totais_gaps_curtos+dias_totais_gaps_medios)/registros_apos_imputacao_total*100:.1f}%")
print()
print(f"Gaps longos (>30 dias) - NÃO serão imputados:")
print(f"  • Dias: {dias_totais_gaps_longos}")
print(f"  • Participantes afetados: {len(participantes_gap_longo)}")
print()

# Salvar resultados detalhados
print("="*80)
print("Salvando resultados...")

# Salvar CSV
df_resultados.to_csv('analise_gaps_detalhada.csv', index=False)
print("✓ Salvo: analise_gaps_detalhada.csv")

# Salvar JSON com informações completas
relatorio = {
    'resumo_geral': {
        'total_participantes': int(psych['id'].nunique()),
        'total_registros_originais': int(len(psych)),
        'total_gaps_identificados': int(total_gaps_sistema),
        'participantes_com_gaps_longos': len(participantes_gap_longo)
    },
    'distribuicao_gaps': {
        'gaps_curtos': {
            'quantidade': len(gaps_curtos),
            'percentual': round(len(gaps_curtos)/total_gaps_sistema*100, 1),
            'dias_totais': int(dias_totais_gaps_curtos),
            'media_dias': round(np.mean(gaps_curtos), 1) if gaps_curtos else 0,
            'mediana_dias': round(np.median(gaps_curtos), 1) if gaps_curtos else 0
        },
        'gaps_medios': {
            'quantidade': len(gaps_medios),
            'percentual': round(len(gaps_medios)/total_gaps_sistema*100, 1),
            'dias_totais': int(dias_totais_gaps_medios),
            'media_dias': round(np.mean(gaps_medios), 1) if gaps_medios else 0,
            'mediana_dias': round(np.median(gaps_medios), 1) if gaps_medios else 0
        },
        'gaps_longos': {
            'quantidade': len(gaps_longos),
            'percentual': round(len(gaps_longos)/total_gaps_sistema*100, 1),
            'dias_totais': int(dias_totais_gaps_longos),
            'media_dias': round(np.mean(gaps_longos), 1) if gaps_longos else 0,
            'mediana_dias': round(np.median(gaps_longos), 1) if gaps_longos else 0
        }
    },
    'impacto_imputacao_conservadora': {
        'registros_originais': int(registros_originais),
        'apos_forward_fill': {
            'total_registros': int(registros_apos_imputacao_curta),
            'registros_imputados': int(dias_totais_gaps_curtos),
            'percentual_imputado': round(dias_totais_gaps_curtos/registros_apos_imputacao_curta*100, 1)
        },
        'apos_interpolacao': {
            'total_registros': int(registros_apos_imputacao_total),
            'registros_imputados': int(dias_totais_gaps_curtos + dias_totais_gaps_medios),
            'percentual_imputado': round((dias_totais_gaps_curtos+dias_totais_gaps_medios)/registros_apos_imputacao_total*100, 1)
        }
    },
    'participantes_gap_longo': participantes_gap_longo
}

with open('relatorio_gaps_temporais.json', 'w', encoding='utf-8') as f:
    json.dump(relatorio, f, indent=2, ensure_ascii=False)
print("✓ Salvo: relatorio_gaps_temporais.json")

print()
print("="*80)
print("ANÁLISE CONCLUÍDA!")
print("="*80)
