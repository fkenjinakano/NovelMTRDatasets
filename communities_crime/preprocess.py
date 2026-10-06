import pandas as pd

# Dicionário com as colunas de saída perPop anotadas com 'target_'
TARGET_PER_POP = {
    130: 'target_murdPerPop',
    132: 'target_rapesPerPop',
    134: 'target_robberiesPerPop',
    136: 'target_assaultsPerPop',
    138: 'target_burglariesPerPop',
    140: 'target_larceniesPerPop',
    142: 'target_autoTheftPerPop',
    144: 'target_arsonsPerPop',
    145: 'target_ViolentCrimesPerPop',
    146: 'target_nonViolPerPop'
}

# Índices das contagens absolutas brutas a remover totalmente
RAW_COUNT_TARGET_INDICES = [129, 131, 133, 135, 137, 139, 141, 143]

def preprocess_single_file(file_path, output_path='communities_crime_preprocessed.csv'):
    """
    Pré-processa o dataset Communities and Crime sem padronização, 
    mantendo apenas métricas perPop e exportando tudo para um único ficheiro CSV.
    """
    print("[1/5] A carregar o dataset...")
    df = pd.read_csv(file_path, header=None, na_values='?')
    print(f"Dimensão original: {df.shape[0]} linhas x {df.shape[1]} colunas")

    # 1. Remover identificadores geográficos e metadados (colunas 0 a 4)
    df_clean = df.drop(columns=[0, 1, 2, 3, 4])

    # 2. Remover contagens absolutas brutas de crimes
    df_clean = df_clean.drop(columns=RAW_COUNT_TARGET_INDICES)

    # 3. Remover colunas de policiamento esparsas (>80% de valores omissos)
    limiar_nulos = 0.80
    proporcao_nulos = df_clean.isna().sum() / len(df_clean)
    colunas_esparsas = proporcao_nulos[proporcao_nulos > limiar_nulos].index
    df_clean = df_clean.drop(columns=colunas_esparsas)

    # 4. Eliminar instâncias (linhas) com valores faltantes remanescentes
    df_clean = df_clean.dropna()

    # 5. Renomear apenas as colunas de saída com o prefixo 'target_'
    df_final = df_clean.rename(columns=TARGET_PER_POP)

    # 6. Exportar tudo para um único ficheiro CSV
    df_final.to_csv(output_path, index=False)

    print("[5/5] Ficheiro único gerado com sucesso!")
    print(f"Dimensão final do ficheiro: {df_final.shape[0]} linhas x {df_final.shape[1]} colunas")
    print(f"Ficheiro guardado em: {output_path}")

    return df_final

if __name__ == "__main__":
    df_clean = preprocess_single_file(file_path='CommViolPredUnnormalizedData.txt')