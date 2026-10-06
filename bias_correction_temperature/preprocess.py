import os
import pandas as pd


def preprocess_drop_missing(input_path='bias_correction_ucl.csv'):
    # Gerar automaticamente o nome de saída: <nome_original>_preprocessed.csv
    base, _ = os.path.splitext(input_path)
    output_path = "bias_correction_temperature_preprocessed.csv"

    # 1. Carregar o arquivo bruto
    df_raw = pd.read_csv(input_path)

    # Identificar índices de linhas com pelo menos um valor faltante
    missing_rows_mask = df_raw.isnull().any(axis=1)
    missing_rows_indices = df_raw[missing_rows_mask].index.tolist()

    print('=' * 70)
    print('      RELATÓRIO DE PRÉ-PROCESSAMENTO (DROP DE LINHAS FALTANTES)')
    print('=' * 70)
    print('\n--- [ANTES DO PRÉ-PROCESSAMENTO] ---')
    print(f'• Número total de linhas: {df_raw.shape[0]}')
    print(f'• Número total de colunas: {df_raw.shape[1]}')
    print(
        f'• Total de células faltantes (NaN): {df_raw.isnull().sum().sum()}'
    )
    print(
        f'• Número de linhas descartadas: {len(missing_rows_indices)} ({(len(missing_rows_indices)/len(df_raw))*100:.2f}%)'
    )

    print(
        f'\n📍 Índices das {len(missing_rows_indices)} linhas removidas (Índice 0 do DataFrame / Linha do CSV = Índice + 2):'
    )
    print(f'   {missing_rows_indices}')

    print('\nValores faltantes por coluna antes do remoção:')
    missing_before = df_raw.isnull().sum()[df_raw.isnull().sum() > 0]
    for col, count in missing_before.items():
        percent = (count / len(df_raw)) * 100
        print(f'  - {col}: {count} nulos ({percent:.2f}%)')

    # 2. Remover TODAS as linhas que possuem qualquer valor faltante
    df = df_raw.dropna().copy()

    # 3. Remover a coluna 'Date' completamente
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])

    # 4. Converter estação para inteiro
    if 'station' in df.columns:
        df['station'] = df['station'].astype(int)

    # 5. Reordenar e renomear colunas de saída com o prefixo 'target_'
    features = [c for c in df.columns if c not in ['Next_Tmax', 'Next_Tmin']]
    df_processed = df[features + ['Next_Tmax', 'Next_Tmin']].rename(
        columns={
            'Next_Tmax': 'target_Next_Tmax',
            'Next_Tmin': 'target_Next_Tmin',
        }
    )

    # 6. Salvar o arquivo processado
    df_processed.to_csv(output_path, index=False)

    # Exibir relatório final
    target_cols = [c for c in df_processed.columns if c.startswith('target_')]
    print('\n--- [APÓS O PRÉ-PROCESSAMENTO] ---')
    print(f'• Número de linhas restantes: {df_processed.shape[0]}')
    print(f'• Número de colunas: {df_processed.shape[1]}')
    print(
        f'• Total de valores faltantes (NaN): {df_processed.isnull().sum().sum()}'
    )
    print(f'• Colunas de entrada (Features): {len(features)}')
    print(f'• Colunas de saída (Targets): {target_cols}')

    print(f"\n✅ Arquivo limpo salvo em: '{output_path}'")
    print('=' * 70)


if __name__ == '__main__':
    preprocess_drop_missing('bias_correction_ucl.csv')