# -*- coding: utf-8 -*-
"""
REFERENCIAS - curvas de crescimento da OMS (tabelas LMS)

Baixa as planilhas oficiais usadas pelo Modulo 2 para julgar se peso, altura
e IMC sao plausiveis para a idade e o sexo. Os arquivos ja estao versionados
em data/referencias/oms/; este script documenta a origem e permite
reconstrui-los (so baixa o que estiver faltando).

  - WHO Child Growth Standards (2006), 0 a 5 anos, tabelas por dia:
    https://www.who.int/tools/child-growth-standards/standards
  - WHO Growth Reference (2007), 5 a 19 anos, tabelas por mes:
    https://www.who.int/tools/growth-reference-data-for-5to19-years
    (de Onis et al., Bull World Health Organ 2007;85:660-7)

Limites de implausibilidade (WHO Anthro / AnthroPlus): peso/idade < -6 ou
> +5; altura/idade < -6 ou > +6; IMC/idade < -5 ou > +5.
"""

import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qualidade import DIR_OMS, validar_tabelas_oms

CGS = "https://cdn.who.int/media/docs/default-source/child-growth/child-growth-standards/indicators"
GR = "https://cdn.who.int/media/docs/default-source/child-growth/growth-reference-5-19-years"

ARQUIVOS = {
    "wfa_boys_0-5.xlsx":   f"{CGS}/weight-for-age/expanded-tables/wfa-boys-zscore-expanded-tables.xlsx",
    "wfa_girls_0-5.xlsx":  f"{CGS}/weight-for-age/expanded-tables/wfa-girls-zscore-expanded-tables.xlsx",
    "lhfa_boys_0-5.xlsx":  f"{CGS}/length-height-for-age/expandable-tables/lhfa-boys-zscore-expanded-tables.xlsx?sfvrsn=7b4a3428_12",
    "lhfa_girls_0-5.xlsx": f"{CGS}/length-height-for-age/expandable-tables/lhfa-girls-zscore-expanded-tables.xlsx?sfvrsn=27f1e2cb_10",
    "bfa_boys_0-5.xlsx":   f"{CGS}/body-mass-index-for-age/expanded-tables/bfa-boys-zscore-expanded-tables.xlsx?sfvrsn=f8e1fbe2_10",
    "bfa_girls_0-5.xlsx":  f"{CGS}/body-mass-index-for-age/expanded-tables/bfa-girls-zscore-expanded-tables.xlsx?sfvrsn=ae4cb8d1_12",
    "wfa_boys_5-10.xlsx":  f"{GR}/weight-for-age-(5-10-years)/hfa-boys-z-who-2007-exp_0ff9c43c-8cc0-4c23-9fc6-81290675e08b.xlsx?sfvrsn=b3ca0d6f_4",
    "wfa_girls_5-10.xlsx": f"{GR}/weight-for-age-(5-10-years)/hfa-girls-z-who-2007-exp_7ea58763-36a2-436d-bef0-7fcfbadd2820.xlsx?sfvrsn=6ede55a4_4",
    "hfa_boys_5-19.xlsx":  f"{GR}/height-for-age-(5-19-years)/hfa-boys-z-who-2007-exp.xlsx?sfvrsn=7fa263d_2",
    "hfa_girls_5-19.xlsx": f"{GR}/height-for-age-(5-19-years)/hfa-girls-z-who-2007-exp.xlsx?sfvrsn=79d310ee_2",
    "bfa_boys_5-19.xlsx":  f"{GR}/bmi-for-age-(5-19-years)/bmi-boys-z-who-2007-exp.xlsx?sfvrsn=a84bca93_2",
    "bfa_girls_5-19.xlsx": f"{GR}/bmi-for-age-(5-19-years)/bmi-girls-z-who-2007-exp.xlsx?sfvrsn=79222875_2",
}


def main():
    os.makedirs(DIR_OMS, exist_ok=True)
    for nome, url in ARQUIVOS.items():
        destino = os.path.join(DIR_OMS, nome)
        if os.path.exists(destino):
            print(f"  [ok] {nome}")
            continue
        # o CDN da OMS recusa (403) o User-Agent padrao do urllib
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(destino, "wb") as f:
            f.write(resp.read())
        print(f"  [baixado] {nome}")
    erro = validar_tabelas_oms()
    print(f"\nEscore-z confere com as tabelas da OMS (erro máximo {erro:.4f}).")


if __name__ == "__main__":
    main()
