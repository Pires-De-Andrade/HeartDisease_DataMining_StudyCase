# Tabelas de referência clínica

Usadas por `scripts/clinico.py` para construir atributos determinísticos (não dependem de estatística da amostra).

| Arquivo | Conteúdo | Fonte |
|---|---|---|
| `nhbpep_2004_pa.csv` | PAS/PAD nos percentis 50, 90, 95 e 99 por sexo, idade (1–17 anos) e percentil de altura (5–95) | National High Blood Pressure Education Program. *The Fourth Report on the Diagnosis, Evaluation, and Treatment of High Blood Pressure in Children and Adolescents.* Pediatrics 2004;114(2):555-576, Tabelas 3 e 4. Extraído do PDF do material complementar com `extrair_nhbpep.py`. |
| `cdc_estatura_lms.csv` | Parâmetros LMS de estatura por idade (24–240 meses), sexo 1 = M, 2 = F | CDC Growth Charts 2000, `statage.csv` (cdc.gov/growthcharts/data/zscore), baixado em 2026-09-27 |
| `cdc_comprimento_0a36m_lms.csv` | LMS de comprimento por idade (0–36 meses) | CDC Growth Charts 2000, `lenageinf.csv` |
| `cdc_imc_lms.csv` | LMS de IMC por idade (24–240 meses) | CDC Growth Charts 2000, `bmiagerev.csv` |
| `fc_faixas.csv` | Faixas de frequência cardíaca de repouso por faixa etária | MedlinePlus (NIH), *Pulse — normal resting heart rates by age*; mesmas faixas divulgadas pela American Heart Association. Faixas de repouso ambulatorial (as faixas PALS, de emergência, têm limite inferior alto demais e classificariam 16% da base como bradicardia). Esquema estável: pode ser substituída pelos percentis de Fleming et al. (Lancet 2011;377:1011) sem alterar código. |

Cortes usados em `clinico.py`:

- **IMC** (percentil para idade e sexo, CDC): < p5 baixo peso; p5–p85 eutrofia; p85–p95 excesso de peso; ≥ p95 obesidade. Só para ≥ 24 meses.
- **PA** (NHBPEP 2004, critério PAS **ou** PAD): normal < p90; pré-hipertensão p90–p95 (ou ≥ 120/80 a partir de 12 anos); HAS estágio 1 p95 até p99 + 5 mmHg; HAS estágio 2 ≥ p99 + 5. Para ≥ 18 anos, critério adulto: ≥ 120/80 pré-HAS, ≥ 140/90 HAS-1, ≥ 160/100 HAS-2. < 1 ano: não avaliável.
- **FC**: abaixo da faixa = bradicardia; acima = taquicardia.
