# -*- coding: utf-8 -*-
"""
Extrai as tabelas de pressao arterial pediatrica (NHBPEP, 4th Report, Pediatrics
2004; Tabelas 3 e 4) do PDF do material complementar para CSV longo:

    sexo, idade_anos, percentil_pa, percentil_altura, pas, pad

Uso: python3 extrair_nhbpep.py "<caminho do PDF>" [saida.csv]
"""
import re
import subprocess
import sys

import pandas as pd

pdf = sys.argv[1]
saida = sys.argv[2] if len(sys.argv) > 2 else "nhbpep_2004_pa.csv"
texto = subprocess.run(["pdftotext", "-layout", pdf, "-"],
                       capture_output=True, text=True, check=True).stdout

ALTURA = [5, 10, 25, 50, 75, 90, 95]
linhas, sexo, idade = [], None, None
for ln in texto.splitlines():
    if "for Boys" in ln:
        sexo = "M"
    elif "for Girls" in ln:
        sexo = "F"
    # o PDF grava a idade 1 das meninas como a letra "l"
    m = re.match(r"^\s*(\d{1,2}|l)?\s+(50|90|95|99)th\s+((?:\d+\s+){13}\d+)\s*$", ln)
    if not m:
        continue
    if m.group(1):
        idade = 1 if m.group(1) == "l" else int(m.group(1))
    perc = int(m.group(2))
    vals = list(map(int, m.group(3).split()))
    for i, h in enumerate(ALTURA):
        linhas.append({"sexo": sexo, "idade_anos": idade, "percentil_pa": perc,
                       "percentil_altura": h, "pas": vals[i], "pad": vals[7 + i]})

df = pd.DataFrame(linhas)
esperado = 2 * 17 * 4 * 7
assert len(df) == esperado, f"{len(df)} linhas, esperado {esperado}"
assert df.groupby(["sexo"]).idade_anos.nunique().eq(17).all()
df.to_csv(saida, index=False)
print(f"[ok] {saida}: {len(df)} linhas, sexos {sorted(df.sexo.unique())}, "
      f"idades {df.idade_anos.min()}-{df.idade_anos.max()}")
