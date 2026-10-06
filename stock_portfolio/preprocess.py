import pandas as pd

# 1. Carregar o conjunto de dados usando apenas o segundo cabeçalho (linha de índice 1)
df = pd.read_csv("stock portfolio performance data set.csv", header=1)

# Limpar espaços em branco dos nomes das colunas
df.columns = [c.strip() for c in df.columns]

# 2. Verificar e reportar linhas com valores em falta
missing_rows = df[df.isna().any(axis=1)]
print(f"Total de linhas: {len(df)}")
print(f"Linhas com valores em falta: {len(missing_rows)} (Total de NaN: {df.isna().sum().sum()})")

# 3. Mapeamento explícito das variáveis-alvo (targets)
target_mapping = {
    "Annual Return": "target_annual_rate",
    "Excess Return": "target_excess_rate",
    "Systematic Risk": "target_systematic_risk",
    "Total Risk": "target_total_risk",
    "Abs. Win Rate": "target_abs_win_rate",
    "Rel. Win Rate": "target_rel_win_rate",
}

# 4. Renomear apenas a primeira ocorrência das variáveis-alvo
new_cols = []
seen_targets = set()

for col in df.columns:
    if col in target_mapping and col not in seen_targets:
        new_cols.append(target_mapping[col])
        seen_targets.add(col)
    else:
        new_cols.append(col)

df.columns = new_cols

# 5. Limpar símbolos de % e converter todas as colunas para valores numéricos
for col in df.columns:
    df[col] = df[col].astype(str).str.rstrip("%").astype(float)

# 6. Remover a coluna ID
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

# 7. Reordenar para colocar as colunas target no final
target_cols = [
    "target_annual_rate",
    "target_excess_rate",
    "target_systematic_risk",
    "target_total_risk",
    "target_abs_win_rate",
    "target_rel_win_rate",
]

feature_cols = [col for col in df.columns if col not in target_cols]
df = df[feature_cols + target_cols]

# 8. Guardar o ficheiro pré-processado
output_filename = "stock_portfolio_preprocessed.csv"
df.to_csv(output_filename, index=False)
print(f"\nFicheiro guardado com sucesso em '{output_filename}'")
print(f"Dimensões finais: {df.shape}")