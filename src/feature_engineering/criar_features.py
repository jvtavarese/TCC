"""
Feature Engineering - Fase 1.c
================================
Criar features derivadas baseadas em conhecimento de domínio
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

print("="*80)
print("FEATURE ENGINEERING - FASE 1.c")
print("="*80)
print()

# ============================================================================
# 1. CARREGAR DADOS COM FLAGS
# ============================================================================

print("1. Carregando dados com flags de outliers...")
psych = pd.read_csv('dados_psychological_com_flags_outliers.csv')
phys = pd.read_csv('dados_physical_com_flags_outliers.csv')

print(f"   Registros psicológicos: {len(psych)}")
print(f"   Registros físicos: {len(phys)}")
print(f"   Colunas originais (psych): {len(psych.columns)}")
print(f"   Colunas originais (phys): {len(phys.columns)}")
print()

# ============================================================================
# 2. FEATURES DE SONO
# ============================================================================

print("2. Criando features de SONO...")

def criar_features_sono(df):
    """Cria features derivadas de sono"""
    features_criadas = []

    # 1. Total de sono (minutos)
    df['total_sleep_minutes'] = df['lightsleep'] + df['deepsleep'] + df['remsleep']
    features_criadas.append('total_sleep_minutes')

    # 2. Eficiência do sono (0-1)
    # total_sleep / (total_sleep + awakesleep)
    df['sleep_efficiency'] = np.where(
        (df['total_sleep_minutes'] + df['awakesleep']) > 0,
        df['total_sleep_minutes'] / (df['total_sleep_minutes'] + df['awakesleep']),
        np.nan
    )
    features_criadas.append('sleep_efficiency')

    # 3. Ratio sono profundo (0-1)
    df['deep_sleep_ratio'] = np.where(
        df['total_sleep_minutes'] > 0,
        df['deepsleep'] / df['total_sleep_minutes'],
        np.nan
    )
    features_criadas.append('deep_sleep_ratio')

    # 4. Ratio sono REM (0-1)
    df['rem_sleep_ratio'] = np.where(
        df['total_sleep_minutes'] > 0,
        df['remsleep'] / df['total_sleep_minutes'],
        np.nan
    )
    features_criadas.append('rem_sleep_ratio')

    # 5. Ratio sono leve (0-1)
    df['light_sleep_ratio'] = np.where(
        df['total_sleep_minutes'] > 0,
        df['lightsleep'] / df['total_sleep_minutes'],
        np.nan
    )
    features_criadas.append('light_sleep_ratio')

    # 6. Ratio awakesleep (fragmentação)
    df['awakesleep_ratio'] = np.where(
        df['total_sleep_minutes'] > 0,
        df['awakesleep'] / df['total_sleep_minutes'],
        np.nan
    )
    features_criadas.append('awakesleep_ratio')

    # 7. Sleep quality score composto (0-1)
    # Baseado em: deep (40%), REM (30%), eficiência (30%)
    df['sleep_quality_score'] = np.where(
        df['sleep_efficiency'].notna() & df['deep_sleep_ratio'].notna() & df['rem_sleep_ratio'].notna(),
        (df['deep_sleep_ratio'] * 0.4) +
        (df['rem_sleep_ratio'] * 0.3) +
        (df['sleep_efficiency'] * 0.3),
        np.nan
    )
    features_criadas.append('sleep_quality_score')

    # 8. Short sleeper (< 6 horas)
    df['is_short_sleeper'] = (df['total_sleep_minutes'] < 360).astype(int)
    features_criadas.append('is_short_sleeper')

    # 9. Long sleeper (> 9 horas)
    df['is_long_sleeper'] = (df['total_sleep_minutes'] > 540).astype(int)
    features_criadas.append('is_long_sleeper')

    return df, features_criadas

psych, features_sono = criar_features_sono(psych)
phys, _ = criar_features_sono(phys)

print(f"   ✓ {len(features_sono)} features de sono criadas:")
for f in features_sono:
    print(f"     - {f}")
print()

# ============================================================================
# 3. FEATURES DE HRV (VARIABILIDADE DA FREQUÊNCIA CARDÍACA)
# ============================================================================

print("3. Criando features de HRV...")

def criar_features_hrv(df):
    """Cria features derivadas de HRV"""
    features_criadas = []

    # 1. Balanço autonômico (RMSSD/SDNN)
    df['hrv_balance'] = np.where(
        df['sdnn'] > 0,
        df['rmssd'] / df['sdnn'],
        np.nan
    )
    features_criadas.append('hrv_balance')

    # 2. Stress index (inverso de RMSSD, normalizado)
    # Valores altos de stress = RMSSD baixo
    df['hrv_stress_index'] = np.where(
        df['rmssd'] > 0,
        100 / df['rmssd'],  # normalizado para escala 0-100
        np.nan
    )
    features_criadas.append('hrv_stress_index')

    # 3. Recovery score (média de métricas de recuperação)
    df['hrv_recovery_score'] = (df['rmssd'] + df['sdsd']) / 2
    features_criadas.append('hrv_recovery_score')

    # 4. Cardiovascular fitness (mean_nni relativo a max_hr)
    df['cardiovascular_fitness'] = np.where(
        df['max_hr'] > 0,
        df['mean_nni'] / df['max_hr'],
        np.nan
    )
    features_criadas.append('cardiovascular_fitness')

    # 5. HRV category (binning)
    # Baseado em literatura: RMSSD <40=baixo, 40-70=médio, >70=alto
    df['hrv_category'] = pd.cut(
        df['rmssd'],
        bins=[0, 40, 70, np.inf],
        labels=['low', 'medium', 'high']
    )
    features_criadas.append('hrv_category')

    return df, features_criadas

psych, features_hrv = criar_features_hrv(psych)
phys, _ = criar_features_hrv(phys)

print(f"   ✓ {len(features_hrv)} features de HRV criadas:")
for f in features_hrv:
    print(f"     - {f}")
print()

# ============================================================================
# 4. FEATURES DE ATIVIDADE FÍSICA
# ============================================================================

print("4. Criando features de ATIVIDADE FÍSICA...")

def criar_features_atividade(df):
    """Cria features derivadas de atividade física"""
    features_criadas = []

    # 1. Dia ativo (>10k passos - OMS)
    df['is_active_day'] = (df['steps'] >= 10000).astype(int)
    features_criadas.append('is_active_day')

    # 2. Dia sedentário (<5k passos)
    df['is_sedentary_day'] = (df['steps'] < 5000).astype(int)
    features_criadas.append('is_sedentary_day')

    # 3. Intensidade da atividade (calorias por passo)
    df['activity_intensity'] = np.where(
        df['steps'] > 0,
        df['calories'] / df['steps'],
        np.nan
    )
    features_criadas.append('activity_intensity')

    # 4. Tem corrida (exercício estruturado)
    df['has_running'] = (df['running'] > 0).astype(int)
    features_criadas.append('has_running')

    return df, features_criadas

psych, features_atividade = criar_features_atividade(psych)
phys, _ = criar_features_atividade(phys)

print(f"   ✓ {len(features_atividade)} features de atividade criadas:")
for f in features_atividade:
    print(f"     - {f}")
print()

# ============================================================================
# 5. FEATURES DE COMUNICAÇÃO
# ============================================================================

print("5. Criando features de COMUNICAÇÃO...")

def criar_features_comunicacao(df):
    """Cria features derivadas de comunicação"""
    features_criadas = []

    # 1. Total de chamadas
    df['total_calls'] = df['incomingcalls'] + df['outgoingcalls']
    features_criadas.append('total_calls')

    # 2. Balanço de chamadas (quem inicia mais contato)
    # +1 para evitar divisão por zero
    df['call_balance'] = df['outgoingcalls'] / (df['incomingcalls'] + 1)
    features_criadas.append('call_balance')

    # 3. Duração média de chamadas
    df['avg_call_duration'] = (
        df['incomingcallsaverageduration'] + df['outgoingcallsaverageduration']
    ) / 2
    features_criadas.append('avg_call_duration')

    # 4. Tem contato social
    df['has_social_contact'] = (df['total_calls'] > 0).astype(int)
    features_criadas.append('has_social_contact')

    # 5. Chamadas problemáticas
    df['problematic_calls'] = (
        df['missedcalls'] + df['rejectedcalls'] + df['blockedcalls']
    )
    features_criadas.append('problematic_calls')

    # 6. Total WhatsApp
    df['total_whatsapp'] = (
        df['whatsappnotification'] + df['whatsappinvoice'] +
        df['whatsappoutvideo'] + df['whatsappoutvoice']
    )
    features_criadas.append('total_whatsapp')

    return df, features_criadas

psych, features_comunicacao = criar_features_comunicacao(psych)
phys, _ = criar_features_comunicacao(phys)

print(f"   ✓ {len(features_comunicacao)} features de comunicação criadas:")
for f in features_comunicacao:
    print(f"     - {f}")
print()

# ============================================================================
# 6. FEATURES TEMPORAIS
# ============================================================================

print("6. Criando features TEMPORAIS...")

def criar_features_temporal(df):
    """Cria features derivadas de tempo"""
    features_criadas = []

    # Converter day para datetime
    df['date_parsed'] = pd.to_datetime(df['day'], format='%Y%m%d')

    # 1. Dia da semana (1=segunda, 7=domingo)
    df['day_of_week'] = df['date_parsed'].dt.dayofweek + 1
    features_criadas.append('day_of_week')

    # 2. Fim de semana
    df['is_weekend'] = (df['day_of_week'] >= 6).astype(int)
    features_criadas.append('is_weekend')

    # 3. Mês (1-12)
    df['month'] = df['date_parsed'].dt.month
    features_criadas.append('month')

    # 4. Dias desde o início (por participante)
    for participant_id in df['id'].unique():
        mask = df['id'] == participant_id
        first_date = df.loc[mask, 'date_parsed'].min()
        df.loc[mask, 'days_since_start'] = (
            df.loc[mask, 'date_parsed'] - first_date
        ).dt.days

    features_criadas.append('days_since_start')

    # Remover coluna auxiliar
    df = df.drop('date_parsed', axis=1)

    return df, features_criadas

psych, features_temporal = criar_features_temporal(psych)
phys, _ = criar_features_temporal(phys)

print(f"   ✓ {len(features_temporal)} features temporais criadas:")
for f in features_temporal:
    print(f"     - {f}")
print()

# ============================================================================
# 7. FEATURES COMPOSTAS (DOMÍNIO COMPLEXO)
# ============================================================================

print("7. Criando features COMPOSTAS...")

def criar_features_compostas(df):
    """Cria features compostas baseadas em múltiplas dimensões"""
    features_criadas = []

    # 1. Recovery index (sono + HRV)
    # Normalizar RMSSD para 0-1 (dividir por 100, típico máximo)
    df['recovery_index'] = np.where(
        df['sleep_efficiency'].notna() & df['rmssd'].notna(),
        (df['sleep_efficiency'] * 0.5) + ((df['rmssd'] / 100) * 0.5),
        np.nan
    )
    features_criadas.append('recovery_index')

    # 2. Stress index (sono ruim + HRV baixo + problemas sociais)
    # Normalizar componentes para 0-1
    df['stress_index'] = np.where(
        df['awakesleep_ratio'].notna() & df['hrv_balance'].notna(),
        (df['awakesleep_ratio'] * 0.4) +
        ((1 - np.clip(df['hrv_balance'], 0, 1)) * 0.3) +
        (np.clip(df['problematic_calls'] / 10, 0, 1) * 0.3),
        np.nan
    )
    features_criadas.append('stress_index')

    # 3. Wellbeing composite (score holístico)
    # Normalizar atividade (steps/10000) e social (calls/5)
    df['wellbeing_composite'] = np.where(
        df['sleep_quality_score'].notna() & df['hrv_recovery_score'].notna(),
        (df['sleep_quality_score'] * 0.3) +
        ((df['hrv_recovery_score'] / 100) * 0.3) +
        (np.clip(df['steps'] / 10000, 0, 1) * 0.2) +
        (np.clip(df['total_calls'] / 5, 0, 1) * 0.2),
        np.nan
    )
    features_criadas.append('wellbeing_composite')

    return df, features_criadas

psych, features_compostas = criar_features_compostas(psych)
phys, _ = criar_features_compostas(phys)

print(f"   ✓ {len(features_compostas)} features compostas criadas:")
for f in features_compostas:
    print(f"     - {f}")
print()

# ============================================================================
# 8. RESUMO DAS FEATURES CRIADAS
# ============================================================================

print("="*80)
print("RESUMO DAS FEATURES CRIADAS")
print("="*80)
print()

todas_features = (features_sono + features_hrv + features_atividade +
                  features_comunicacao + features_temporal + features_compostas)

print(f"Total de features criadas: {len(todas_features)}")
print()

print("Por categoria:")
print(f"  • Sono: {len(features_sono)}")
print(f"  • HRV: {len(features_hrv)}")
print(f"  • Atividade Física: {len(features_atividade)}")
print(f"  • Comunicação: {len(features_comunicacao)}")
print(f"  • Temporal: {len(features_temporal)}")
print(f"  • Compostas: {len(features_compostas)}")
print()

print(f"Colunas totais (psych): {len(psych.columns)} (era {116} + {len(todas_features)} novas)")
print(f"Colunas totais (phys): {len(phys.columns)} (era {116} + {len(todas_features)} novas)")
print()

# ============================================================================
# 9. VALIDAÇÃO BÁSICA DAS FEATURES
# ============================================================================

print("="*80)
print("VALIDAÇÃO DAS FEATURES CRIADAS")
print("="*80)
print()

print("9. Verificando missing values...")
print()

# Contar missing por feature criada
missing_info = []
for feature in todas_features:
    if feature in psych.columns:
        n_missing = psych[feature].isna().sum()
        perc_missing = (n_missing / len(psych)) * 100
        missing_info.append({
            'feature': feature,
            'n_missing': n_missing,
            'perc_missing': perc_missing
        })

df_missing = pd.DataFrame(missing_info).sort_values('perc_missing', ascending=False)

print("Features com missing values:")
features_com_missing = df_missing[df_missing['perc_missing'] > 0]
if len(features_com_missing) > 0:
    print(features_com_missing.to_string(index=False))
else:
    print("  ✓ Nenhuma feature com missing values!")
print()

# ============================================================================
# 10. CORRELAÇÃO COM TARGETS
# ============================================================================

print("10. Calculando correlação com targets...")
print()

# Correlação com target psicológico
correlacoes_psych = []
for feature in todas_features:
    if feature in psych.columns and psych[feature].dtype in ['int64', 'float64']:
        corr = psych[feature].corr(psych['psy_ref_score'])
        correlacoes_psych.append({
            'feature': feature,
            'correlation': abs(corr),
            'correlation_signed': corr
        })

df_corr_psych = pd.DataFrame(correlacoes_psych).sort_values('correlation', ascending=False)

print("TOP 10 FEATURES COM MAIOR CORRELAÇÃO (psy_ref_score):")
print(df_corr_psych[['feature', 'correlation_signed']].head(10).to_string(index=False))
print()

# Correlação com target físico
correlacoes_phys = []
for feature in todas_features:
    if feature in phys.columns and phys[feature].dtype in ['int64', 'float64']:
        corr = phys[feature].corr(phys['phy_ref_score'])
        correlacoes_phys.append({
            'feature': feature,
            'correlation': abs(corr),
            'correlation_signed': corr
        })

df_corr_phys = pd.DataFrame(correlacoes_phys).sort_values('correlation', ascending=False)

print("TOP 10 FEATURES COM MAIOR CORRELAÇÃO (phys_ref_score):")
print(df_corr_phys[['feature', 'correlation_signed']].head(10).to_string(index=False))
print()

# ============================================================================
# 11. SALVAR DATASETS FINAIS
# ============================================================================

print("="*80)
print("SALVANDO DATASETS FINAIS")
print("="*80)
print()

psych.to_csv('dados_psychological_com_features.csv', index=False)
print(f"✓ Salvo: dados_psychological_com_features.csv")
print(f"  {len(psych)} registros × {len(psych.columns)} colunas")

phys.to_csv('dados_physical_com_features.csv', index=False)
print(f"✓ Salvo: dados_physical_com_features.csv")
print(f"  {len(phys)} registros × {len(phys.columns)} colunas")
print()

# Salvar lista de features
df_missing.to_csv('features_missing_values.csv', index=False)
print("✓ Salvo: features_missing_values.csv")

df_corr_psych.to_csv('features_correlation_psych.csv', index=False)
print("✓ Salvo: features_correlation_psych.csv")

df_corr_phys.to_csv('features_correlation_phys.csv', index=False)
print("✓ Salvo: features_correlation_phys.csv")
print()

# Salvar dicionário de features
features_dict = {
    'total_features_criadas': len(todas_features),
    'categorias': {
        'sono': features_sono,
        'hrv': features_hrv,
        'atividade': features_atividade,
        'comunicacao': features_comunicacao,
        'temporal': features_temporal,
        'compostas': features_compostas
    },
    'descricoes': {
        'total_sleep_minutes': 'Duração total do sono (lightsleep + deepsleep + remsleep)',
        'sleep_efficiency': 'Eficiência do sono: total_sleep / (total_sleep + awakesleep)',
        'deep_sleep_ratio': 'Proporção de sono profundo: deepsleep / total_sleep',
        'rem_sleep_ratio': 'Proporção de sono REM: remsleep / total_sleep',
        'light_sleep_ratio': 'Proporção de sono leve: lightsleep / total_sleep',
        'awakesleep_ratio': 'Fragmentação do sono: awakesleep / total_sleep',
        'sleep_quality_score': 'Score composto: deep*0.4 + rem*0.3 + efficiency*0.3',
        'is_short_sleeper': '1 se sono < 6h, 0 caso contrário',
        'is_long_sleeper': '1 se sono > 9h, 0 caso contrário',
        'hrv_balance': 'Balanço autonômico: RMSSD / SDNN',
        'hrv_stress_index': 'Índice de stress: 100 / RMSSD',
        'hrv_recovery_score': 'Score de recuperação: (RMSSD + sdsd) / 2',
        'cardiovascular_fitness': 'Fitness cardiovascular: mean_nni / max_hr',
        'hrv_category': 'Categoria HRV: low (<40), medium (40-70), high (>70)',
        'is_active_day': '1 se steps >= 10000 (OMS), 0 caso contrário',
        'is_sedentary_day': '1 se steps < 5000, 0 caso contrário',
        'activity_intensity': 'Intensidade: calories / steps',
        'has_running': '1 se running > 0, 0 caso contrário',
        'total_calls': 'Total de chamadas: incomingcalls + outgoingcalls',
        'call_balance': 'Balanço de chamadas: outgoingcalls / (incomingcalls + 1)',
        'avg_call_duration': 'Duração média: (incoming_duration + outgoing_duration) / 2',
        'has_social_contact': '1 se total_calls > 0, 0 caso contrário',
        'problematic_calls': 'Chamadas problemáticas: missed + rejected + blocked',
        'total_whatsapp': 'Total WhatsApp: notification + invoice + outvideo + outvoice',
        'day_of_week': 'Dia da semana: 1=segunda, 7=domingo',
        'is_weekend': '1 se sábado/domingo, 0 caso contrário',
        'month': 'Mês: 1-12',
        'days_since_start': 'Dias desde primeiro registro do participante',
        'recovery_index': 'Índice de recuperação: sleep_efficiency*0.5 + RMSSD/100*0.5',
        'stress_index': 'Índice de stress: awakesleep_ratio*0.4 + (1-hrv_balance)*0.3 + problematic_calls/10*0.3',
        'wellbeing_composite': 'Score holístico: sleep_quality*0.3 + hrv/100*0.3 + steps/10k*0.2 + calls/5*0.2'
    }
}

with open('features_dicionario.json', 'w', encoding='utf-8') as f:
    json.dump(features_dict, f, indent=2, ensure_ascii=False)
print("✓ Salvo: features_dicionario.json")
print()

print("="*80)
print("FEATURE ENGINEERING CONCLUÍDO COM SUCESSO!")
print("="*80)
