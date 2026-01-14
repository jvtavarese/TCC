"""
FASE 2.a - PREPARAÇÃO DOS DADOS PARA MODELAGEM

Este script implementa:
1. Criação de 6 cenários de dados para análise de sensibilidade
2. Separação de features, flags e identificadores
3. One-Hot Encoding de hrv_category
4. Validação cruzada por participante (GroupKFold)
5. Cálculo de sample weights por participante
6. Modelos naive baseline (média global e por participante)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from pathlib import Path

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

# Paths dos dados
DATA_DIR = Path("/Users/jvtesteves/Projetos/TCC")
FASE2_DIR = DATA_DIR / "fase 2"

# Colunas a remover (identificadores e flags)
COLUNAS_IDENTIFICADORES = ['id', 'day']

# Colunas de flags (não usar como features)
COLUNAS_FLAGS = [
    'flag_original', 'flag_imputado_forward', 'flag_imputado_interpolado',
    'flag_baixa_confianca', 'flag_tem_outlier', 'n_variaveis_outlier',
    'flag_outlier_outgoingcallsaverageduration', 'flag_outlier_deepsleep',
    'flag_outlier_incomingcallsaverageduration', 'flag_outlier_calories',
    'flag_outlier_rejectedcalls', 'flag_outlier_missedcalls',
    'flag_outlier_max_hr', 'flag_outlier_steps', 'flag_outlier_mean_nni',
    'flag_outlier_running', 'flag_outlier_rmssd', 'flag_outlier_sdnn',
    'flag_outlier_median_nni', 'flag_outlier_lightsleep', 'flag_outlier_sdsd',
    'flag_outlier_remsleep', 'flag_outlier_blockedcalls',
    'flag_outlier_awakesleep', 'flag_outlier_incomingcalls',
    'flag_outlier_outgoingcalls'
]

# Participantes com baixa confiança (identificados na Fase 1.a)
PARTICIPANTES_BAIXA_CONFIANCA = [
    '0aQYg86Fqb', '1jdPXJ7Ha6', '3BuTHYfEEb', '5Xvl3juEsg',
    'AWSoZE4B1Z', 'bDVQK2sxSv', 'DpSKnVXdIv', 'LfCTWrBx7i', 'VRCa1fREqR'
]

# Top 6 participantes com mais dados (super participantes)
SUPER_PARTICIPANTES = [
    'WPooa9cRhe',  # 226 registros
    'GYQzQAZ8Xm',  # 154 registros
    'pDqyc8VC9j',  # 150 registros
    'QjKLsVxw1F',  # 140 registros
    'HdtU5Dxr13',  # 134 registros
    'IgJAYaDVl3'   # 128 registros
]

# ============================================================================
# FUNÇÕES DE CARREGAMENTO E PREPARAÇÃO
# ============================================================================

def carregar_dados(target_type='psychological'):
    """
    Carrega dados psychological ou physical.

    Parameters:
    -----------
    target_type : str
        'psychological' ou 'physical'

    Returns:
    --------
    df : pd.DataFrame
        DataFrame carregado
    target_col : str
        Nome da coluna target
    """
    if target_type == 'psychological':
        file_path = DATA_DIR / 'dados_psychological_com_features.csv'
        target_col = 'psy_ref_score'
    else:
        file_path = DATA_DIR / 'dados_physical_com_features.csv'
        target_col = 'phy_ref_score'  # Nota: é 'phy' não 'phys'

    df = pd.read_csv(file_path)
    print(f"\n{'='*80}")
    print(f"Dados carregados: {file_path.name}")
    print(f"Shape: {df.shape}")
    print(f"Target: {target_col}")
    print(f"{'='*80}\n")

    return df, target_col


def separar_colunas(df, target_col):
    """
    Separa colunas em identificadores, flags, features e target.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame completo
    target_col : str
        Nome da coluna target

    Returns:
    --------
    dict com as separações
    """
    todas_colunas = set(df.columns)

    # Identificadores
    ids = df[COLUNAS_IDENTIFICADORES].copy()

    # Target
    target = df[[target_col]].copy()

    # Flags
    flags_disponiveis = [col for col in COLUNAS_FLAGS if col in df.columns]
    flags = df[flags_disponiveis].copy()

    # Features = tudo que não é identificador, flag ou target
    colunas_remover = set(COLUNAS_IDENTIFICADORES + flags_disponiveis + [target_col])
    colunas_features = list(todas_colunas - colunas_remover)

    features = df[colunas_features].copy()

    print(f"Separação de colunas:")
    print(f"  - Identificadores: {len(COLUNAS_IDENTIFICADORES)} colunas")
    print(f"  - Flags: {len(flags_disponiveis)} colunas")
    print(f"  - Features: {len(colunas_features)} colunas")
    print(f"  - Target: 1 coluna ({target_col})")
    print(f"  - Total: {len(COLUNAS_IDENTIFICADORES) + len(flags_disponiveis) + len(colunas_features) + 1} colunas\n")

    return {
        'ids': ids,
        'flags': flags,
        'features': features,
        'target': target,
        'colunas_features': colunas_features
    }


def aplicar_one_hot_encoding(features):
    """
    Aplica One-Hot Encoding em colunas categóricas (hrv_category e group).
    """
    features_encoded = features.copy()
    
    # 1. hrv_category
    if 'hrv_category' in features_encoded.columns:
        print(f"Aplicando One-Hot Encoding em hrv_category...")
        print(f"  Categorias únicas: {features_encoded['hrv_category'].unique()}")
        features_encoded = pd.get_dummies(
            features_encoded,
            columns=['hrv_category'],
            prefix='hrv',
            drop_first=False
        )
        print(f"  ✓ hrv_category codificado\n")

    # 2. group
    if 'group' in features_encoded.columns:
        print(f"Aplicando One-Hot Encoding em group...")
        print(f"  Categorias únicas: {features_encoded['group'].unique()}")
        features_encoded = pd.get_dummies(
            features_encoded,
            columns=['group'],
            prefix='group',
            drop_first=False,
            dummy_na=True
        )
        print(f"  ✓ group codificado\n")

    # Verificar outras categóricas
    object_cols = features_encoded.select_dtypes(include=['object']).columns.tolist()
    if len(object_cols) > 0:
        print(f"⚠️  Removendo colunas categóricas: {object_cols}")
        features_encoded = features_encoded.drop(columns=object_cols)

    print(f"Resumo: {features.shape[1]} → {features_encoded.shape[1]} colunas\n")
    return features_encoded




# ============================================================================
# FUNÇÕES DE CRIAÇÃO DE CENÁRIOS
# ============================================================================

def criar_cenarios(df, flags, ids):
    """
    Cria 6 cenários de dados para análise de sensibilidade.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame completo
    flags : pd.DataFrame
        DataFrame com flags
    ids : pd.DataFrame
        DataFrame com identificadores

    Returns:
    --------
    cenarios : dict
        Dicionário com os 6 cenários
    """
    cenarios = {}

    # CENÁRIO 1 - COMPLETO
    cenario1_idx = df.index
    cenarios['completo'] = {
        'indices': cenario1_idx,
        'descricao': 'Todos os registros (com todas as imputações)'
    }

    # CENÁRIO 2 - SEM INTERPOLAÇÃO
    cenario2_idx = df[flags['flag_imputado_interpolado'] == False].index
    cenarios['sem_interpolacao'] = {
        'indices': cenario2_idx,
        'descricao': 'Remove registros com imputação por interpolação'
    }

    # CENÁRIO 3 - APENAS ORIGINAIS
    cenario3_idx = df[flags['flag_original'] == True].index
    cenarios['apenas_originais'] = {
        'indices': cenario3_idx,
        'descricao': 'Apenas registros originais (zero imputação)'
    }

    # CENÁRIO 4 - SEM BAIXA CONFIANÇA
    cenario4_idx = df[~ids['id'].isin(PARTICIPANTES_BAIXA_CONFIANCA)].index
    cenarios['sem_baixa_confianca'] = {
        'indices': cenario4_idx,
        'descricao': f'Remove {len(PARTICIPANTES_BAIXA_CONFIANCA)} participantes com baixa confiança'
    }

    # CENÁRIO 5 - SEM OUTLIERS EXTREMOS
    cenario5_idx = df[flags['n_variaveis_outlier'] <= 2].index
    cenarios['sem_outliers_extremos'] = {
        'indices': cenario5_idx,
        'descricao': 'Remove registros com mais de 2 variáveis outlier'
    }

    # CENÁRIO 6 - SEM SUPER PARTICIPANTES
    cenario6_idx = df[~ids['id'].isin(SUPER_PARTICIPANTES)].index
    cenarios['sem_super_participantes'] = {
        'indices': cenario6_idx,
        'descricao': f'Remove top {len(SUPER_PARTICIPANTES)} participantes com mais dados'
    }

    # Relatório
    print(f"{'='*80}")
    print("CENÁRIOS CRIADOS:")
    print(f"{'='*80}\n")

    for nome, info in cenarios.items():
        n_registros = len(info['indices'])
        n_participantes = ids.loc[info['indices'], 'id'].nunique()
        pct = (n_registros / len(df)) * 100

        print(f"{nome.upper()}")
        print(f"  {info['descricao']}")
        print(f"  Registros: {n_registros:,} ({pct:.1f}%)")
        print(f"  Participantes: {n_participantes}")
        print()

    return cenarios


def preparar_cenario(cenario_indices, features, target, ids):
    """
    Prepara dados de um cenário específico.

    Parameters:
    -----------
    cenario_indices : pd.Index
        Índices dos registros do cenário
    features : pd.DataFrame
        DataFrame com features
    target : pd.DataFrame
        DataFrame com target
    ids : pd.DataFrame
        DataFrame com identificadores

    Returns:
    --------
    dict com X, y, participant_ids
    """
    X = features.loc[cenario_indices].reset_index(drop=True)
    y = target.loc[cenario_indices].reset_index(drop=True)
    participant_ids = ids.loc[cenario_indices, 'id'].reset_index(drop=True)

    return {
        'X': X,
        'y': y.values.ravel(),  # converter para array 1D
        'participant_ids': participant_ids
    }


# ============================================================================
# FUNÇÕES DE VALIDAÇÃO CRUZADA
# ============================================================================

def criar_validacao_cruzada(n_splits=5):
    """
    Cria objeto de validação cruzada por participante.

    Parameters:
    -----------
    n_splits : int
        Número de folds

    Returns:
    --------
    cv : GroupKFold
        Objeto de validação cruzada
    """
    return GroupKFold(n_splits=n_splits)


def calcular_sample_weights(participant_ids):
    """
    Calcula sample weights por participante.
    Peso = 1 / número de registros do participante

    Objetivo: balancear influência de participantes com muitos vs poucos dados

    Parameters:
    -----------
    participant_ids : pd.Series ou np.array
        IDs dos participantes

    Returns:
    --------
    weights : np.array
        Peso de cada observação
    """
    # Contar registros por participante
    contagens = pd.Series(participant_ids).value_counts()

    # Calcular peso: 1 / n_registros
    weights = pd.Series(participant_ids).map(lambda x: 1.0 / contagens[x]).values

    print(f"Sample Weights calculados:")
    print(f"  Participantes únicos: {len(contagens)}")
    print(f"  Registros por participante - min: {contagens.min()}, max: {contagens.max()}, média: {contagens.mean():.1f}")
    print(f"  Weights - min: {weights.min():.6f}, max: {weights.max():.6f}")
    print(f"  Participante com mais dados: {contagens.idxmax()} ({contagens.max()} registros, peso={1/contagens.max():.6f})")
    print(f"  Participante com menos dados: {contagens.idxmin()} ({contagens.min()} registros, peso={1/contagens.min():.6f})")
    print()

    return weights


# ============================================================================
# MODELOS BASELINE NAIVE
# ============================================================================

class NaiveMeanPredictor:
    """
    Modelo naive que sempre prediz a média do target no conjunto de treino.
    """
    def __init__(self):
        self.mean_ = None

    def fit(self, y_train):
        """Calcula a média do target."""
        self.mean_ = np.mean(y_train)
        return self

    def predict(self, n_samples):
        """Retorna array com a média repetida n_samples vezes."""
        return np.full(n_samples, self.mean_)


class NaiveParticipantMeanPredictor:
    """
    Modelo naive que prediz a média por participante.
    Se o participante não existir no treino, usa a média global.
    """
    def __init__(self):
        self.participant_means_ = {}
        self.global_mean_ = None

    def fit(self, y_train, participant_ids_train):
        """Calcula média por participante e média global."""
        df_train = pd.DataFrame({
            'y': y_train,
            'id': participant_ids_train
        })

        self.participant_means_ = df_train.groupby('id')['y'].mean().to_dict()
        self.global_mean_ = np.mean(y_train)

        return self

    def predict(self, participant_ids_test):
        """Prediz usando média do participante ou média global."""
        predictions = []
        for pid in participant_ids_test:
            if pid in self.participant_means_:
                predictions.append(self.participant_means_[pid])
            else:
                predictions.append(self.global_mean_)

        return np.array(predictions)


def avaliar_modelo_naive(y_true, y_pred, nome_modelo):
    """
    Avalia modelo naive e retorna métricas.

    Parameters:
    -----------
    y_true : np.array
        Valores reais
    y_pred : np.array
        Predições
    nome_modelo : str
        Nome do modelo para relatório

    Returns:
    --------
    dict com métricas
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'modelo': nome_modelo,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def testar_modelos_naive(X, y, participant_ids, cv, verbose=True):
    """
    Testa modelos naive usando validação cruzada.

    Parameters:
    -----------
    X : pd.DataFrame
        Features (não usado pelos modelos naive, mas necessário para índices)
    y : np.array
        Target
    participant_ids : pd.Series
        IDs dos participantes
    cv : GroupKFold
        Objeto de validação cruzada
    verbose : bool
        Se True, imprime resultados

    Returns:
    --------
    dict com resultados dos dois modelos
    """
    # Listas para armazenar métricas de cada fold
    resultados_mean = []
    resultados_participant = []

    # Validação cruzada
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=participant_ids), 1):
        y_train = y[train_idx]
        y_test = y[test_idx]

        participant_ids_train = participant_ids.iloc[train_idx]
        participant_ids_test = participant_ids.iloc[test_idx]

        # Modelo 1: Média global
        model_mean = NaiveMeanPredictor()
        model_mean.fit(y_train)
        y_pred_mean = model_mean.predict(len(y_test))

        metrics_mean = avaliar_modelo_naive(y_test, y_pred_mean, 'Naive Mean')
        resultados_mean.append(metrics_mean)

        # Modelo 2: Média por participante
        model_participant = NaiveParticipantMeanPredictor()
        model_participant.fit(y_train, participant_ids_train)
        y_pred_participant = model_participant.predict(participant_ids_test)

        metrics_participant = avaliar_modelo_naive(y_test, y_pred_participant, 'Naive Participant Mean')
        resultados_participant.append(metrics_participant)

    # Calcular médias
    def calcular_media_metricas(resultados):
        return {
            'rmse_mean': np.mean([r['rmse'] for r in resultados]),
            'rmse_std': np.std([r['rmse'] for r in resultados]),
            'mae_mean': np.mean([r['mae'] for r in resultados]),
            'mae_std': np.std([r['mae'] for r in resultados]),
            'r2_mean': np.mean([r['r2'] for r in resultados]),
            'r2_std': np.std([r['r2'] for r in resultados])
        }

    resultado_final = {
        'naive_mean_global': calcular_media_metricas(resultados_mean),
        'naive_mean_por_participante': calcular_media_metricas(resultados_participant)
    }

    if verbose:
        print(f"{'='*80}")
        print("RESULTADOS MODELOS NAIVE (BASELINE)")
        print(f"{'='*80}\n")

        print("1. NAIVE MEAN GLOBAL (sempre prediz média)")
        print(f"   RMSE: {resultado_final['naive_mean_global']['rmse_mean']:.4f} ± {resultado_final['naive_mean_global']['rmse_std']:.4f}")
        print(f"   MAE:  {resultado_final['naive_mean_global']['mae_mean']:.4f} ± {resultado_final['naive_mean_global']['mae_std']:.4f}")
        print(f"   R²:   {resultado_final['naive_mean_global']['r2_mean']:.4f} ± {resultado_final['naive_mean_global']['r2_std']:.4f}")
        print()

        print("2. NAIVE MEAN POR PARTICIPANTE")
        print(f"   RMSE: {resultado_final['naive_mean_por_participante']['rmse_mean']:.4f} ± {resultado_final['naive_mean_por_participante']['rmse_std']:.4f}")
        print(f"   MAE:  {resultado_final['naive_mean_por_participante']['mae_mean']:.4f} ± {resultado_final['naive_mean_por_participante']['mae_std']:.4f}")
        print(f"   R²:   {resultado_final['naive_mean_por_participante']['r2_mean']:.4f} ± {resultado_final['naive_mean_por_participante']['r2_std']:.4f}")
        print()

    return resultado_final


# ============================================================================
# FUNÇÃO PRINCIPAL DE PREPARAÇÃO
# ============================================================================

def preparar_dados_completo(target_type='psychological', n_splits=5):
    """
    Pipeline completo de preparação dos dados.

    Parameters:
    -----------
    target_type : str
        'psychological' ou 'physical'
    n_splits : int
        Número de folds para validação cruzada

    Returns:
    --------
    dict com todos os cenários preparados e informações adicionais
    """
    print(f"\n{'#'*80}")
    print(f"# FASE 2.a - PREPARAÇÃO DOS DADOS PARA MODELAGEM")
    print(f"# Target: {target_type}")
    print(f"{'#'*80}\n")

    # 1. Carregar dados
    df, target_col = carregar_dados(target_type)

    # 2. Separar colunas
    separacao = separar_colunas(df, target_col)
    ids = separacao['ids']
    flags = separacao['flags']
    features = separacao['features']
    target = separacao['target']

    # 3. One-Hot Encoding de hrv_category
    features_encoded = aplicar_one_hot_encoding(features)

    # 4. Criar cenários
    cenarios = criar_cenarios(df, flags, ids)

    # 5. Preparar cada cenário
    cenarios_preparados = {}

    for nome_cenario, info_cenario in cenarios.items():
        print(f"\nPreparando cenário: {nome_cenario.upper()}")
        print(f"-" * 80)

        dados_cenario = preparar_cenario(
            info_cenario['indices'],
            features_encoded,
            target,
            ids
        )

        # Calcular sample weights
        weights = calcular_sample_weights(dados_cenario['participant_ids'])
        dados_cenario['sample_weights'] = weights

        # Criar validação cruzada
        cv = criar_validacao_cruzada(n_splits)

        # Testar modelos naive
        resultados_naive = testar_modelos_naive(
            dados_cenario['X'],
            dados_cenario['y'],
            dados_cenario['participant_ids'],
            cv
        )

        dados_cenario['resultados_naive'] = resultados_naive
        dados_cenario['descricao'] = info_cenario['descricao']
        dados_cenario['n_registros'] = len(dados_cenario['X'])
        dados_cenario['n_participantes'] = dados_cenario['participant_ids'].nunique()

        cenarios_preparados[nome_cenario] = dados_cenario

    # 6. Criar objeto de validação cruzada para retornar
    cv = criar_validacao_cruzada(n_splits)

    # 7. Salvar resumo
    resumo = {
        'target_type': target_type,
        'target_col': target_col,
        'n_splits': n_splits,
        'n_features_original': len(separacao['colunas_features']),
        'n_features_encoded': features_encoded.shape[1],
        'cenarios': {}
    }

    for nome, dados in cenarios_preparados.items():
        resumo['cenarios'][nome] = {
            'descricao': dados['descricao'],
            'n_registros': dados['n_registros'],
            'n_participantes': dados['n_participantes'],
            'naive_mean_global': dados['resultados_naive']['naive_mean_global'],
            'naive_mean_por_participante': dados['resultados_naive']['naive_mean_por_participante']
        }

    # Salvar resumo em JSON
    output_file = FASE2_DIR / f'resumo_preparacao_{target_type}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Resumo salvo em: {output_file}")
    print(f"{'='*80}\n")

    return {
        'cenarios': cenarios_preparados,
        'cv': cv,
        'target_col': target_col,
        'feature_names': features_encoded.columns.tolist(),
        'resumo': resumo
    }


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# PREPARAÇÃO DE DADOS PARA MODELAGEM")
    print("# Processando ambos targets: psychological E physical")
    print("#"*80 + "\n")

    # Preparar dados PSYCHOLOGICAL
    print("\n" + ">"*80)
    print("> INICIANDO: PSYCHOLOGICAL")
    print(">"*80 + "\n")
    resultado_psy = preparar_dados_completo(target_type='psychological', n_splits=5)

    # Preparar dados PHYSICAL
    print("\n" + ">"*80)
    print("> INICIANDO: PHYSICAL")
    print(">"*80 + "\n")
    resultado_phys = preparar_dados_completo(target_type='physical', n_splits=5)

    # Resumo final
    print("\n" + "="*80)
    print("PREPARAÇÃO CONCLUÍDA PARA AMBOS TARGETS!")
    print("="*80)

    print("\n📊 PSYCHOLOGICAL (psy_ref_score):")
    print(f"   Cenários: {len(resultado_psy['cenarios'])}")
    print(f"   Features: {len(resultado_psy['feature_names'])}")
    print(f"   CV Folds: {resultado_psy['cv'].n_splits}")
    print(f"   Arquivo:  fase 2/resumo_preparacao_psychological.json")

    print("\n📊 PHYSICAL (phys_ref_score):")
    print(f"   Cenários: {len(resultado_phys['cenarios'])}")
    print(f"   Features: {len(resultado_phys['feature_names'])}")
    print(f"   CV Folds: {resultado_phys['cv'].n_splits}")
    print(f"   Arquivo:  fase 2/resumo_preparacao_physical.json")

    print("\n✅ Ambos targets preparados com sucesso!")
