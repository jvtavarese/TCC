"""
Script para criar datasets CRUS limpos após remoção de features com VIF alto.

Este script:
1. Carrega as recomendações de remoção baseadas em VIF
2. Remove as features recomendadas dos datasets CRUS originais
3. Salva os novos datasets: dados_crus_physical_apos_vif.csv e dados_crus_psychological_apos_vif.csv
"""

import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path('/Users/jvtesteves/Projetos/TCC')
MULTICOLINEARIDADE_DIR = DATA_DIR / 'notebooks' / 'multicolinearidade'

print("="*80)
print("CRIANDO DATASETS CRUS LIMPOS APÓS ANÁLISE DE VIF")
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
print(f"   Features: {', '.join(features_to_remove_physical[:5])}...")

print(f"\n✓ Psychological: {len(features_to_remove_psychological)} features para remover")
print(f"   Features: {', '.join(features_to_remove_psychological[:5])}...")

# ============================================================================
# 2. CRIAR DATASET PHYSICAL CRU LIMPO
# ============================================================================

print("\n2️⃣  Criando dados_crus_physical_apos_vif.csv...")
print("-"*80)

# Carregar dataset CRU original
df_physical_cru = pd.read_csv(DATA_DIR / '20230625-processed-physical-qol.csv')
print(f"Dataset CRU original: {df_physical_cru.shape[0]} linhas × {df_physical_cru.shape[1]} colunas")

# Verificar quais features existem no dataset cru
features_existentes_phy = [f for f in features_to_remove_physical if f in df_physical_cru.columns]
features_nao_encontradas_phy = [f for f in features_to_remove_physical if f not in df_physical_cru.columns]

print(f"\n📊 Status das features a remover:")
print(f"   ✓ Encontradas no dataset: {len(features_existentes_phy)}")
print(f"   ⚠️  Não encontradas: {len(features_nao_encontradas_phy)}")

if features_nao_encontradas_phy:
    print(f"\n⚠️  Features não encontradas no dataset CRU:")
    for feat in features_nao_encontradas_phy:
        print(f"     - {feat}")

# Remover features existentes
if features_existentes_phy:
    df_physical_cru_clean = df_physical_cru.drop(columns=features_existentes_phy)
    print(f"\n✓ Removidas {len(features_existentes_phy)} features")
else:
    df_physical_cru_clean = df_physical_cru.copy()
    print("\n⚠️  Nenhuma feature foi removida (nenhuma encontrada no dataset)")

print(f"Dataset CRU limpo: {df_physical_cru_clean.shape[0]} linhas × {df_physical_cru_clean.shape[1]} colunas")

# Salvar
output_file_physical = DATA_DIR / 'dados_crus_physical_apos_vif.csv'
df_physical_cru_clean.to_csv(output_file_physical, index=False)
print(f"✓ Salvo em: {output_file_physical}")

# ============================================================================
# 3. CRIAR DATASET PSYCHOLOGICAL CRU LIMPO
# ============================================================================

print("\n3️⃣  Criando dados_crus_psychological_apos_vif.csv...")
print("-"*80)

# Carregar dataset CRU original
df_psychological_cru = pd.read_csv(DATA_DIR / '20230625-processed-psychological-qol.csv')
print(f"Dataset CRU original: {df_psychological_cru.shape[0]} linhas × {df_psychological_cru.shape[1]} colunas")

# Verificar quais features existem no dataset cru
features_existentes_psy = [f for f in features_to_remove_psychological if f in df_psychological_cru.columns]
features_nao_encontradas_psy = [f for f in features_to_remove_psychological if f not in df_psychological_cru.columns]

print(f"\n📊 Status das features a remover:")
print(f"   ✓ Encontradas no dataset: {len(features_existentes_psy)}")
print(f"   ⚠️  Não encontradas: {len(features_nao_encontradas_psy)}")

if features_nao_encontradas_psy:
    print(f"\n⚠️  Features não encontradas no dataset CRU:")
    for feat in features_nao_encontradas_psy:
        print(f"     - {feat}")

# Remover features existentes
if features_existentes_psy:
    df_psychological_cru_clean = df_psychological_cru.drop(columns=features_existentes_psy)
    print(f"\n✓ Removidas {len(features_existentes_psy)} features")
else:
    df_psychological_cru_clean = df_psychological_cru.copy()
    print("\n⚠️  Nenhuma feature foi removida (nenhuma encontrada no dataset)")

print(f"Dataset CRU limpo: {df_psychological_cru_clean.shape[0]} linhas × {df_psychological_cru_clean.shape[1]} colunas")

# Salvar
output_file_psychological = DATA_DIR / 'dados_crus_psychological_apos_vif.csv'
df_psychological_cru_clean.to_csv(output_file_psychological, index=False)
print(f"✓ Salvo em: {output_file_psychological}")

# ============================================================================
# 4. RESUMO FINAL
# ============================================================================

print("\n" + "="*80)
print("RESUMO")
print("="*80)

print("\n📊 PHYSICAL (CRU):")
print(f"   Original: {df_physical_cru.shape[1]} colunas")
print(f"   Removidas: {len(features_existentes_phy)} colunas")
print(f"   Final: {df_physical_cru_clean.shape[1]} colunas")
if len(features_existentes_phy) > 0:
    print(f"   Redução: {(len(features_existentes_phy)/df_physical_cru.shape[1])*100:.1f}%")

print("\n📊 PSYCHOLOGICAL (CRU):")
print(f"   Original: {df_psychological_cru.shape[1]} colunas")
print(f"   Removidas: {len(features_existentes_psy)} colunas")
print(f"   Final: {df_psychological_cru_clean.shape[1]} colunas")
if len(features_existentes_psy) > 0:
    print(f"   Redução: {(len(features_existentes_psy)/df_psychological_cru.shape[1])*100:.1f}%")

print("\n" + "="*80)
print("✅ DATASETS CRUS LIMPOS CRIADOS COM SUCESSO!")
print("="*80)

print("\n📁 Arquivos criados:")
print(f"   1. {output_file_physical.name}")
print(f"   2. {output_file_psychological.name}")

# Listar features removidas
if len(features_existentes_phy) > 0:
    print("\n🔍 Features removidas:")
    print(f"   Physical: {', '.join(features_existentes_phy)}")

if len(features_existentes_psy) > 0:
    print(f"\n   Psychological: {', '.join(features_existentes_psy)}")

print("\n💡 Próximos passos:")
print("   1. Use os novos datasets CRUS para pipeline de feature engineering")
print("   2. Compare com datasets originais")
print("   3. Treine modelos e verifique se R² melhorou")
