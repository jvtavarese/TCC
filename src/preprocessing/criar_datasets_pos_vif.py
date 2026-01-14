"""
Script para criar datasets limpos após remoção de features com VIF alto.

Este script:
1. Carrega as recomendações de remoção baseadas em VIF
2. Remove as features recomendadas dos datasets originais
3. Salva os novos datasets: dados_physical_apos_vif.csv e dados_psychological_apos_vif.csv
"""

import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path('/Users/jvtesteves/Projetos/TCC')
MULTICOLINEARIDADE_DIR = DATA_DIR / 'notebooks' / 'multicolinearidade'

print("="*80)
print("CRIANDO DATASETS LIMPOS APÓS ANÁLISE DE VIF")
print("="*80)

# ============================================================================
# 1. CARREGAR RECOMENDAÇÕES
# ============================================================================

print("\n1️⃣  Carregando recomendações de remoção...")
print("-"*80)

# Carregar recomendações
rec_physical = pd.read_csv(MULTICOLINEARIDADE_DIR / 'recomendacoes_remocao_Physical.csv')
rec_psychological = pd.read_csv(MULTICOLINEARIDADE_DIR / 'recomendacoes_remocao_Psychological.csv')

# Extrair lista de features a remover
features_to_remove_physical = rec_physical['feature'].tolist()
features_to_remove_psychological = rec_psychological['feature'].tolist()

print(f"✓ Physical: {len(features_to_remove_physical)} features para remover")
print(f"✓ Psychological: {len(features_to_remove_psychological)} features para remover")

# ============================================================================
# 2. CRIAR DATASET PHYSICAL LIMPO
# ============================================================================

print("\n2️⃣  Criando dados_physical_apos_vif.csv...")
print("-"*80)

# Carregar dataset original
df_physical = pd.read_csv(DATA_DIR / 'dados_physical_com_features.csv')
print(f"Dataset original: {df_physical.shape[0]} linhas × {df_physical.shape[1]} colunas")

# Remover features recomendadas (se existirem no dataset)
features_existentes = [f for f in features_to_remove_physical if f in df_physical.columns]
features_nao_encontradas = [f for f in features_to_remove_physical if f not in df_physical.columns]

if features_nao_encontradas:
    print(f"⚠️  {len(features_nao_encontradas)} features não encontradas no dataset:")
    for feat in features_nao_encontradas[:5]:
        print(f"     - {feat}")
    if len(features_nao_encontradas) > 5:
        print(f"     ... e mais {len(features_nao_encontradas) - 5}")

df_physical_clean = df_physical.drop(columns=features_existentes)

print(f"Dataset limpo: {df_physical_clean.shape[0]} linhas × {df_physical_clean.shape[1]} colunas")
print(f"✓ Removidas {len(features_existentes)} features")

# Salvar
output_file_physical = DATA_DIR / 'dados_physical_apos_vif.csv'
df_physical_clean.to_csv(output_file_physical, index=False)
print(f"✓ Salvo em: {output_file_physical}")

# ============================================================================
# 3. CRIAR DATASET PSYCHOLOGICAL LIMPO
# ============================================================================

print("\n3️⃣  Criando dados_psychological_apos_vif.csv...")
print("-"*80)

# Carregar dataset original
df_psychological = pd.read_csv(DATA_DIR / 'dados_psychological_com_features.csv')
print(f"Dataset original: {df_psychological.shape[0]} linhas × {df_psychological.shape[1]} colunas")

# Remover features recomendadas (se existirem no dataset)
features_existentes_psy = [f for f in features_to_remove_psychological if f in df_psychological.columns]
features_nao_encontradas_psy = [f for f in features_to_remove_psychological if f not in df_psychological.columns]

if features_nao_encontradas_psy:
    print(f"⚠️  {len(features_nao_encontradas_psy)} features não encontradas no dataset:")
    for feat in features_nao_encontradas_psy[:5]:
        print(f"     - {feat}")
    if len(features_nao_encontradas_psy) > 5:
        print(f"     ... e mais {len(features_nao_encontradas_psy) - 5}")

df_psychological_clean = df_psychological.drop(columns=features_existentes_psy)

print(f"Dataset limpo: {df_psychological_clean.shape[0]} linhas × {df_psychological_clean.shape[1]} colunas")
print(f"✓ Removidas {len(features_existentes_psy)} features")

# Salvar
output_file_psychological = DATA_DIR / 'dados_psychological_apos_vif.csv'
df_psychological_clean.to_csv(output_file_psychological, index=False)
print(f"✓ Salvo em: {output_file_psychological}")

# ============================================================================
# 4. RESUMO FINAL
# ============================================================================

print("\n" + "="*80)
print("RESUMO")
print("="*80)

print("\n📊 PHYSICAL:")
print(f"   Original: {df_physical.shape[1]} colunas")
print(f"   Removidas: {len(features_existentes)} colunas")
print(f"   Final: {df_physical_clean.shape[1]} colunas")
print(f"   Redução: {(len(features_existentes)/df_physical.shape[1])*100:.1f}%")

print("\n📊 PSYCHOLOGICAL:")
print(f"   Original: {df_psychological.shape[1]} colunas")
print(f"   Removidas: {len(features_existentes_psy)} colunas")
print(f"   Final: {df_psychological_clean.shape[1]} colunas")
print(f"   Redução: {(len(features_existentes_psy)/df_psychological.shape[1])*100:.1f}%")

print("\n" + "="*80)
print("✅ DATASETS LIMPOS CRIADOS COM SUCESSO!")
print("="*80)

print("\n📁 Arquivos criados:")
print(f"   1. {output_file_physical.name}")
print(f"   2. {output_file_psychological.name}")

# Listar algumas features removidas mais importantes
print("\n🔍 Principais features removidas (Physical):")
print("   - Features HRV: sdsd, rmssd, hrv_recovery_score, recovery_index")
print("   - Features Sleep: total_sleep_minutes, sleep_efficiency, sleep_quality_score")
print("   - Features Calls: avg_call_duration, problematic_calls, total_calls")
print("   - Features WhatsApp: whatsappnotification, total_whatsapp, whatsappoutvoice")
print("   - Features demográficas: maritalstatus_single, maritalstatus_married")
print(f"   ... e mais {len(features_existentes) - 15} features")

print("\n💡 Próximos passos:")
print("   1. Use os novos datasets para treinar modelos")
print("   2. Compare performance com datasets originais")
print("   3. Verifique se R² melhorou com a remoção da multicolinearidade")
