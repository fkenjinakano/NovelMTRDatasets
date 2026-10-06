from pathlib import Path
import pandas as pd
import scipy.io as sio

# Nomes das 28 colunas
columns = (
    [f'q_{i}' for i in range(1, 8)]
    + [f'qdot_{i}' for i in range(1, 8)]
    + [f'qddot_{i}' for i in range(1, 8)]
    + [f'target_tau_{i}' for i in range(1, 8)]
)

dfs = []

# Carregar conjunto de treino
if Path('sarcos_inv.mat').exists():
    raw_train = sio.loadmat('sarcos_inv.mat')['sarcos_inv']
    dfs.append(pd.DataFrame(raw_train, columns=columns))

# Carregar conjunto de teste
if Path('sarcos_inv_test.mat').exists():
    raw_test = sio.loadmat('sarcos_inv_test.mat')['sarcos_inv_test']
    dfs.append(pd.DataFrame(raw_test, columns=columns))

# Unificar e exportar com a nova nomenclatura
if dfs:
    df_full = pd.concat(dfs, ignore_index=True)
    output_filename = 'sarcos_preprocessed.csv'
    df_full.to_csv(output_filename, index=False)
    print(
        f'✓ Ficheiro guardado com sucesso: {output_filename} ({len(df_full)}'
        ' linhas)'
    )