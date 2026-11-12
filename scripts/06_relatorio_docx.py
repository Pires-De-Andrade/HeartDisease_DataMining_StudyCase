# -*- coding: utf-8 -*-
"""
RELATORIO FINAL - pre-processamento (limpeza e transformacao)

Monta o .docx a partir das figuras de 05_figuras_relatorio.py e dos numeros
dos modulos 1 a 4. Formato: A4, margens de 2 cm, Arial 10 no corpo; tabelas,
legendas e referencias em corpo menor.

Saida: output/relatorio/relatorio_preprocessamento.docx
"""

import os
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "output", "relatorio")
SAIDA = os.path.join(FIG, "relatorio_preprocessamento.docx")

FONTE = "Arial"
CORPO = Pt(10)
MENOR = Pt(8.5)
LARGURA_UTIL = 17.0            # cm (21 - 2 x 2)
CINZA_CAB = "E8E7E3"

doc = Document()


# =========================================================================
# ESTILOS E PAGINA
# =========================================================================
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
for lado in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, lado, Cm(2))
sec.footer_distance = Cm(1)

normal = doc.styles["Normal"]
normal.font.name = FONTE
normal.font.size = CORPO
normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONTE)
pf = normal.paragraph_format
pf.space_before, pf.space_after = Pt(0), Pt(4)
pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

for nome, tam in (("Heading 1", 11), ("Heading 2", 10)):
    st = doc.styles[nome]
    st.font.name, st.font.size, st.font.bold = FONTE, Pt(tam), True
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.element.rPr.rFonts.set(qn("w:asciiTheme"), "")
    rf = st.element.rPr.rFonts
    for k in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(k), FONTE)
    st.paragraph_format.space_before = Pt(7)
    st.paragraph_format.space_after = Pt(3)
    st.paragraph_format.keep_with_next = True


def rodape_numero():
    p = sec.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.font.size, run.font.name = Pt(8), FONTE
    rf = run._r.get_or_add_rPr().get_or_add_rFonts()
    for k in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(k), FONTE)
    for tipo, texto in (("begin", None), (None, "PAGE"), ("end", None)):
        if tipo:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tipo)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = texto
        run._r.append(el)


rodape_numero()


# =========================================================================
# AUXILIARES DE TEXTO
# =========================================================================
def par(texto="", tam=None, negrito=False, italico=False, alinhar=None,
        depois=None, antes=None, manter=False):
    """Paragrafo com trechos em **negrito** e *italico* marcados no texto."""
    p = doc.add_paragraph()
    if alinhar is not None:
        p.alignment = alinhar
    if depois is not None:
        p.paragraph_format.space_after = Pt(depois)
    if antes is not None:
        p.paragraph_format.space_before = Pt(antes)
    if manter:
        p.paragraph_format.keep_with_next = True
    for trecho, neg, ita in _marcacao(texto):
        r = p.add_run(trecho)
        r.bold = negrito or neg
        r.italic = italico or ita
        if tam:
            r.font.size = tam
    return p


def _marcacao(texto):
    """Divide '**x**' (negrito) e '*x*' (italico) em trechos formatados."""
    out, i, neg, ita, buf = [], 0, False, False, ""
    while i < len(texto):
        if texto.startswith("**", i):
            if buf:
                out.append((buf, neg, ita))
            buf, neg, i = "", not neg, i + 2
        elif texto[i] == "*":
            if buf:
                out.append((buf, neg, ita))
            buf, ita, i = "", not ita, i + 1
        else:
            buf += texto[i]
            i += 1
    if buf:
        out.append((buf, neg, ita))
    return out


def titulo(texto, nivel=1):
    return doc.add_heading(texto, level=nivel)


def legenda(texto, antes=2, depois=6):
    p = par(texto, tam=MENOR, alinhar=WD_ALIGN_PARAGRAPH.LEFT,
            antes=antes, depois=depois)
    return p


def figura(arquivo, texto_legenda):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(os.path.join(FIG, arquivo), width=Cm(LARGURA_UTIL))
    legenda(texto_legenda)


# =========================================================================
# AUXILIARES DE TABELA (estilo "booktabs": so filetes horizontais)
# =========================================================================
def _borda(cell, **lados):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for lado, (sz, cor) in lados.items():
        el = OxmlElement(f"w:{lado}")
        el.set(qn("w:val"), "single" if sz else "nil")
        if sz:
            el.set(qn("w:sz"), str(sz))
            el.set(qn("w:color"), cor)
        borders.append(el)


def _sombra(cell, cor):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), cor)
    cell._tc.get_or_add_tcPr().append(shd)


def _margens_celula(tabela, cima=25, baixo=25, lados=70):
    tblPr = tabela._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for lado, v in (("top", cima), ("bottom", baixo), ("left", lados), ("right", lados)):
        el = OxmlElement(f"w:{lado}")
        el.set(qn("w:w"), str(v))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblPr.append(mar)


def tabela(cabecalho, linhas, larguras_cm, tam=MENOR, alinh_num=()):
    """Tabela com cabecalho sombreado e filetes horizontais finos."""
    t = doc.add_table(rows=1 + len(linhas), cols=len(cabecalho))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    _margens_celula(t)
    grade = t._tbl.tblGrid
    for gc, w in zip(grade.findall(qn("w:gridCol")), larguras_cm):
        gc.set(qn("w:w"), str(int(w * 567)))

    for i, linha in enumerate([cabecalho] + linhas):
        row = t.rows[i]
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)
        if i == 0:
            hdr = OxmlElement("w:tblHeader")
            trPr.append(hdr)
        for j, (cell, texto) in enumerate(zip(row.cells, linha)):
            cell.width = Cm(larguras_cm[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.space_before = Pt(0)
            # cabecalho nunca fica orfao no pe da pagina: prende as 2 primeiras linhas
            if i <= 1:
                p.paragraph_format.keep_with_next = True
            p.alignment = (WD_ALIGN_PARAGRAPH.RIGHT if (j in alinh_num and i > 0)
                           else WD_ALIGN_PARAGRAPH.LEFT)
            for trecho, neg, ita in _marcacao(str(texto)):
                r = p.add_run(trecho)
                r.font.size = tam
                r.bold = (i == 0) or neg
                r.italic = ita
            fim = i == len(linhas)
            _borda(cell,
                   top=(8, "444444") if i == 0 else (0, ""),
                   bottom=(8, "444444") if (i == 0 or fim) else (2, "D0CEC7"),
                   left=(0, ""), right=(0, ""))
            if i == 0:
                _sombra(cell, CINZA_CAB)
    return t


def nota_tabela(texto):
    return par(texto, tam=Pt(7.5), alinhar=WD_ALIGN_PARAGRAPH.LEFT, antes=2, depois=6)


# =========================================================================
# CABECALHO
# =========================================================================
C = WD_ALIGN_PARAGRAPH.CENTER
par("UNIVERSIDADE FEDERAL DE GOIÁS · INSTITUTO DE INFORMÁTICA", tam=Pt(9),
    negrito=True, alinhar=C, depois=0)
par("Bacharelado em Inteligência Artificial · Mineração de Dados", tam=Pt(9),
    alinhar=C, depois=6)
par("Pré-processamento de dados: condição cardíaca em crianças e adolescentes",
    tam=Pt(13), negrito=True, alinhar=C, depois=1)
par("Relatório de limpeza e transformação da base UCMF", tam=Pt(10),
    italico=True, alinhar=C, depois=4)
par("João Guilherme Pires de Andrade · Gustavo Henrique Barros · "
    "Gustavo Almeida Leite · Felipe Brito Albernaz", tam=Pt(9), alinhar=C, depois=2)
p = par("", depois=6)
pPr = p._p.get_or_add_pPr()
bdr = OxmlElement("w:pBdr")
b = OxmlElement("w:bottom")
for k, v in (("w:val", "single"), ("w:sz", "6"), ("w:space", "1"), ("w:color", "888888")):
    b.set(qn(k), v)
bdr.append(b)
pPr.append(bdr)
p.paragraph_format.space_after = Pt(6)

# =========================================================================
# INTRODUCAO
# =========================================================================
par("A base UCMF reúne 12.873 atendimentos de uma clínica de cardiologia pediátrica "
    "no Brasil, com 21 variáveis sobre crianças e jovens de 0 a 19 anos: medidas "
    "antropométricas, sinais vitais, achados do exame cardiológico, queixas, motivo do "
    "encaminhamento e o diagnóstico NORMAL X ANORMAL (ausência ou presença de patologia "
    "cardíaca), adotado como variável-alvo. Este relatório descreve o pré-processamento "
    "da base no processo KDD, na ordem pedida pelo enunciado; os itens 2 e 3 e os itens 4 "
    "e 5 aparecem juntos, porque cada problema é seguido da sua solução e cada gráfico da "
    "sua interpretação.")

# =========================================================================
# 1. UNIVARIADA
# =========================================================================
titulo("1. Análise estatística univariada")
par("As 21 variáveis dividem-se em sete quantitativas (peso, altura, IMC, idade, pressão "
    "arterial sistólica e diastólica — PAS e PAD — e frequência cardíaca, FC), duas datas "
    "(atendimento e nascimento), nove qualitativas nominais, uma qualitativa ordinal "
    "(PPA, classe de pressão arterial), o identificador e o alvo. A Tabela 1 resume cada "
    "variável nos dados brutos, sem nenhum tratamento, e a Figura 1 mostra a distribuição "
    "das quantitativas com a frequência em escala logarítmica, para que valores raros "
    "fiquem visíveis; em vermelho, os valores fora das faixas fisiologicamente possíveis "
    "na população do estudo.")

figura("fig1_univariada.png",
       "**Figura 1** — Distribuição das variáveis quantitativas e do alvo nos dados brutos. "
       "Abaixo de cada painel: % de ausentes e o principal sinal de problema.")

legenda("**Tabela 1** — Resumo univariado dos dados brutos (n = 12.873). Quantitativas: "
        "mínimo · mediana · máximo. Qualitativas: nº de categorias distintas · moda "
        "(% do total).", antes=4, depois=2).paragraph_format.keep_with_next = True
T1 = [
    ["Peso (kg)", "quantitativa", "2,5%", "−40 · 12,8 · 157", "2.609 zeros; 1 valor negativo"],
    ["Altura (cm)", "quantitativa", "0,0%", "0 · 69 · 198", "4.460 zeros (35%)"],
    ["IMC (kg/m²)", "derivada", "36,7%", "0 · 17 · 848", "arredondado a inteiro; máx. 848"],
    ["IDADE (anos)", "derivada", "11,6%", "−113,2 · 4,0 · 71,8", "122 negativas; 94 acima de 19 anos"],
    ["PAS (mmHg)", "quantitativa", "60,0%", "60 · 100 · 990", "700 e 990 mmHg"],
    ["PAD (mmHg)", "quantitativa", "60,1%", "6 · 60 · 120", "6 mmHg; PAD igual à PAS"],
    ["FC (bpm)", "quantitativa", "14,5%", "8 · 92 · 9.288", "16 intervalos em texto (“80-90”)"],
    ["Atendimento", "data", "7,6%", "anos 2003–2010", "312 seriais do Excel; 2 após 2010"],
    ["DN (nascimento)", "data", "10,7%", "anos 1988–2009", "124 seriais; 120 textos; 12 no futuro"],
    ["NORMAL X ANORMAL", "alvo", "9,1%", "4 · Normal 52,4%", "rótulos “anormal” e “Normais”"],
    ["SEXO", "nominal", "0,03%", "6 · M 47,1%", "3 grafias de M; 1.417 “Indeterminado”"],
    ["PULSOS", "nominal", "9,3%", "7 · Normais 89,6%", "“NORMAIS”, “AMPLOS”"],
    ["B2 (2ª bulha)", "nominal", "9,2%", "5 · Normal 86,2%", "—"],
    ["SOPRO", "nominal", "9,1%", "7 · ausente* 56,5%", "Sistólico/sistólico; Contínuo/contínuo"],
    ["HDA 1 (queixa)", "nominal", "33,2%", "8 · Assintomático 51,0%", "—"],
    ["HDA2 (2ª queixa)", "nominal", "96,8%", "8 · Dispneia 0,6%", "quase vazia"],
    ["MOTIVO1", "nominal", "8,2%", "5 · Parecer cardiol. 41,6%", "—"],
    ["MOTIVO2", "nominal", "27,2%", "16 · Cirurgia 22,8%", "—"],
    ["Convênio", "nominal", "31,9%", "374 · GS 16,3%", "179 com 1 registro; CAMED/Camed"],
    ["PPA", "ordinal", "1,7%", "9 · Não Calculado 70,5%", "“#VALUE!” em 1.496 = os sem IDADE"],
]
tabela(["Variável", "Tipo", "Ausentes", "Resumo", "Sinal de problema"], T1,
       [3.3, 1.8, 1.7, 4.3, 5.9], alinh_num=(2,))
nota_tabela("* Em SOPRO, “ausente” é achado clínico (sem sopro), não valor faltante. "
            "Derivada: calculada a partir de outras colunas. Datas: percentis 1–99 dos "
            "anos. O identificador (ID) não tem duplicatas.")

par("A análise revela três padrões que orientam a limpeza. O primeiro é a **ausência "
    "disfarçada**: 21% dos pesos e 35% das alturas são exatamente zero, valor impossível "
    "que ocupa o lugar de “não medido” — por isso a altura parece completa sem estar. "
    "O segundo são **valores impossíveis**, por erro de digitação ou de fórmula: idades "
    "negativas de até −113 anos ou acima da população declarada, IMC de 848, PAS de "
    "990 mmHg, PAD de 6 mmHg e FC de 9.288 bpm. O terceiro é a **falta de padronização "
    "das qualitativas**: SEXO tem seis rótulos para duas categorias, SOPRO e PULSOS "
    "repetem categorias com caixa diferente e o convênio tem 374 grafias. Há ainda "
    "variáveis derivadas com erro de origem: a PPA vale “#VALUE!” exatamente nos 1.496 "
    "registros sem idade — fórmula de planilha que falhou onde faltava um insumo — e "
    "“Não Calculado” em 70,5% dos casos. Quanto à forma, peso e FC são assimétricos à "
    "direita, PAS, PAD e FC concentram-se em múltiplos de 5 e 10 (arredondamento na "
    "medição), PAS e PAD faltam em 60% dos registros e HDA2 em 96,8%. O alvo tem 1.168 "
    "registros vazios e dois rótulos fora do padrão.")

# =========================================================================
# 2. PROBLEMAS E LIMPEZA
# =========================================================================
titulo("2. Problemas de qualidade e soluções de limpeza")
par("A limpeza corrige as colunas existentes, sem criar atributos nem imputar valores: "
    "todo valor inválido passa a ser ausente, e a imputação, por depender de estatísticas "
    "da amostra, fica para a transformação, ajustada só no conjunto de treino. "
    "Distinguiu-se valor **implausível** (erro de registro, anulado) de valor **anormal** "
    "(informação clínica, mantido): uma pressão de hipertenso ou uma criança obesa são "
    "exatamente o que o estudo precisa preservar, e valores extremos altos de peso e altura "
    "em crianças costumam ser reais (Freedman et al., 2015). Peso, altura e IMC passaram "
    "por duas barreiras: limites fixos, impossíveis em qualquer idade de 0 a 19 anos, e "
    "plausibilidade por idade e sexo pelo escore-z das curvas da OMS (WHO, 2006; Onis et "
    "al., 2007), com os limites do WHO Anthro (WHO, 2011): peso/idade abaixo de −6 ou acima "
    "de +5, altura/idade fora de ±6 e IMC/idade fora de ±5. Abaixo de um ano aplicou-se só "
    "o limite superior, porque as curvas descrevem bebês a termo e um prematuro pequeno "
    "para a idade é plausível numa clínica de cardiologia. Pressão e frequência cardíaca "
    "usaram limites fixos amplos (PAS 50–200 e PAD 20–150 mmHg; FC 40–220 bpm), muito além "
    "dos percentis pediátricos extremos (NHBPEP, 2004; Fleming et al., 2011). A Tabela 2 "
    "lista os problemas e as soluções, com as contagens na ordem de aplicação.")

legenda("**Tabela 2** — Problemas de qualidade identificados e soluções de limpeza.",
        antes=4, depois=2).paragraph_format.keep_with_next = True
T2 = [
    ["Registro", "Sem alvo (inclui 739 cópias de um registro vazio)", "Removido", "1.168"],
    ["Registro", "Linhas idênticas exceto pelo ID", "Mantida a primeira ocorrência", "242"],
    ["Alvo", "Rótulos fora do padrão (“anormal”, “Normais”)", "Unificados em Normal/Anormal", "2"],
    ["Datas", "Número serial do Excel no lugar da data", "Convertido em data", "425"],
    ["Datas", "Texto livre, data malformada ou ano impossível (“66” → 2066)",
     "Século corrigido se dedutível; senão, ausente", "128"],
    ["IDADE", "Sistematicamente 3,8% acima da diferença entre as datas",
     "Recalculada pelas datas", "9.493"],
    ["IDADE", "Negativa, sem datas ou incoerente com peso e altura",
     "Ausente; registro mantido", "203"],
    ["IDADE", "20 anos ou mais (fora da população)", "Registro removido", "62"],
    ["Peso / Altura", "Zero usado como “não medido”", "Ausente", "1.649 / 3.450"],
    ["Peso / Altura", "Fora dos limites fixos ou implausível para idade e sexo (OMS)",
     "Ausente", "60 / 79"],
    ["IMC", "Arredondado a inteiro; não verificável se peso ou altura inválidos",
     "Recalculado; ausente se implausível", "7.283"],
    ["PAS / PAD / FC", "Digitação impossível (990 e 6 mmHg; 9.288 bpm); PAD ≥ PAS",
     "Ausente", "21"],
    ["FC", "Intervalo digitado como texto (“80-90”)", "Ponto médio", "16"],
    ["PPA", "“#VALUE!” (falta a idade) e “Não Calculado”", "Ausente; mantidas as 7 classes",
     "9.173"],
    ["SEXO", "Seis grafias para dois sexos; “Indeterminado”", "Unificado em M/F; ausente",
     "871 / 523"],
    ["Categóricas", "SOPRO, PULSOS e Convênio com variações de caixa e acento",
     "Grafia mais frequente (Convênio: 364 → 236)", "1.769"],
]
tabela(["Variável", "Problema", "Solução", "Registros"], T2,
       [2.3, 7.4, 5.4, 1.9], alinh_num=(3,))
nota_tabela("B2, HDA 1, HDA2, MOTIVO1 e MOTIVO2 só apresentavam valores ausentes, "
            "mantidos como tais; a decisão sobre essas variáveis ficou para a transformação.")

par("O problema de maior alcance não aparecia na análise univariada: a IDADE informada é, "
    "em mediana, 3,8% maior que a diferença entre as datas de atendimento e de nascimento, "
    "desvio sistemático que indica erro de fórmula na origem; por isso foi recalculada em "
    "9.493 registros. A base passou de 12.873 para **11.401 registros** (6.540 Normal e "
    "4.861 Anormal). A limpeza também tornou visível a ausência que estava disfarçada: a "
    "altura passa de 0% para 30,9% de ausentes e a PPA de 1,7% para 81,8%. Em sentido "
    "contrário, B2, PULSOS e SOPRO caem de cerca de 9% para menos de 0,5%, porque quase "
    "todos os seus vazios estavam nos registros sem alvo.")

# =========================================================================
# 3. COMPARACAO APOS A LIMPEZA
# =========================================================================
titulo("3. Comparação de duas variáveis após a limpeza e interpretação")
figura("fig2_peso_altura.png",
       "**Figura 2** — Peso × altura nos dados brutos (A) e após a limpeza (B), nos mesmos "
       "eixos. Pares com peso ou altura ausente não aparecem.")
par("A Figura 2 compara peso e altura antes e depois da limpeza. Nos dados brutos, 4.610 "
    "dos 12.555 pares têm ao menos um valor inválido ou implausível para a idade: uma "
    "coluna vertical em altura zero (2.500 registros com peso e altura zerados), uma linha "
    "horizontal em peso zero e pontos isolados fora da nuvem, como 175 cm registrados para "
    "um bebê de um mês. Esses erros distorcem a relação entre as variáveis: a correlação "
    "de Pearson é 0,64.")
par("Após a limpeza restam 7.496 pares, e a nuvem descreve a curva de crescimento "
    "infantil: a correlação de Pearson sobe para 0,92 e a de Spearman chega a 0,98, "
    "relação monotônica quase perfeita. A relação não é linear — em escala logarítmica, o "
    "peso cresce aproximadamente com a altura elevada a 2,2 — e a dispersão do peso "
    "aumenta com a altura: entre adolescentes a variação de peso é muito maior que entre "
    "bebês, reflexo de diferenças de composição corporal, incluindo sobrepeso e obesidade, "
    "que a limpeza deliberadamente preservou. Os 41 pares destacados em B são incoerentes "
    "entre si (por exemplo, 150 cm e 11 kg): como não é possível saber qual medida está "
    "errada, ambas foram mantidas e apenas o IMC foi anulado. A limpeza, portanto, não "
    "“alisou” a relação descartando casos extremos: removeu registros impossíveis e "
    "manteve a variabilidade real.")

# =========================================================================
# 4. BIVARIADA
# =========================================================================
titulo("4. Análise bivariada entre as variáveis independentes e o alvo")
par("Cada variável independente da base limpa foi cruzada com o alvo. Como as "
    "quantitativas são assimétricas, usou-se o teste de Mann-Whitney U, que não supõe "
    "normalidade, com a correlação rank-biserial como tamanho de efeito; para as "
    "qualitativas, o qui-quadrado de independência com o V de Cramér, agregando categorias "
    "raras para respeitar a contagem esperada mínima. Com mais de 11 mil registros quase "
    "todo teste é significativo, então a leitura se apoia no tamanho de efeito, "
    "classificado pelos limiares de Cohen (1988). Identificador, convênio, HDA2, PPA e "
    "datas ficaram de fora pelas razões da Seção 5. A Figura 3 ordena as variáveis pela "
    "força da associação.")
par("**O exame físico domina.** O sopro tem associação forte (V = 0,84): 96% dos pacientes "
    "com sopro sistólico são Anormal, contra 10% dos sem sopro. B2 alterada (87–98% de "
    "Anormal) e o motivo do encaminhamento seguem a mesma lógica — “Cardiopatia já "
    "estabelecida” tem 71% de Anormal e “Parecer cardiológico”, 26%. São achados do mesmo "
    "exame e da mesma triagem que produzem o diagnóstico; a associação forte deve ser lida "
    "como parte do rótulo, não como preditor independente, o que é um ponto de atenção "
    "para a modelagem (o modelo pode aprender o diagnóstico em vez de antecipá-lo). PULSOS "
    "ilustra a diferença entre taxa e efeito: pulsos alterados têm 71–92% de Anormal, mas "
    "são só 139 registros (1,2%), e o efeito global é desprezível.")

par("**O perfil do paciente é, sobretudo, a idade.** O grupo Anormal é mais jovem (mediana "
    "de 2,7 contra 4,8 anos), mais leve e mais baixo (Figura 4). Peso e altura, porém, "
    "acompanham a idade quase perfeitamente (Spearman 0,96 e 0,99), e dentro de uma mesma "
    "faixa etária a diferença entre os grupos praticamente desaparece — em pré-escolares, "
    "a mediana de peso é 16 kg nos dois. Peso, altura e idade são, portanto, o mesmo "
    "sinal: cardiopatias congênitas são identificadas e encaminhadas cedo. A frequência "
    "cardíaca é a exceção: continua mais alta no grupo Anormal dentro das faixas "
    "(lactentes, 110 contra 100 bpm; escolares, 88 contra 80 bpm), o que a torna um "
    "sinal próprio.")

figura("fig3_efeitos.png",
       "**Figura 3** — Tamanho de efeito da associação de cada variável com o alvo "
       "(base limpa, n = 11.401). Todas as associações são significativas (p < 0,05), "
       "exceto sexo.")

figura("fig4_quant_alvo.png",
       "**Figura 4** — Idade, peso, frequência cardíaca e PA sistólica por grupo, com a "
       "mediana ao lado de cada caixa e o número de registros válidos.")

par("**Queixas e obesidade contrariam a intuição.** HDA 1 parece ordenar as queixas "
    "(cianose com 55% de Anormal, palpitação com 29%), mas a ordem reflete a idade de quem "
    "as relata: a cianose tem mediana de 0,9 ano; dor precordial e desmaio, perto de 9 e 10 "
    "anos. Dentro de uma mesma faixa etária, sintomáticos e assintomáticos têm a mesma taxa "
    "de Anormal (lactentes, 58% e 57%). Da mesma forma, o baixo peso se associa ao "
    "diagnóstico e a obesidade não: em todas as faixas etárias a taxa de Anormal cai do "
    "baixo peso para a obesidade (escolares: 39%, 35%, 31% e 26%). O alvo mede cardiopatia "
    "estrutural, em geral congênita, que prejudica o ganho de peso — e não a doença "
    "aterosclerótica do adulto, para a qual a obesidade é fator de risco.")

par("**A pressão arterial não discrimina.** PAS e PAD têm a mesma mediana (100/60 mmHg) "
    "nos dois grupos e efeito desprezível, inclusive na classe clínica construída na "
    "Seção 5 (V = 0,08). Sexo não tem associação detectável (p = 0,08).")

# =========================================================================
# 5. TRANSFORMACOES
# =========================================================================
titulo("5. Transformações a serem realizadas e justificativas")
par("As transformações preparam a base para algoritmos que exigem atributos numéricos e "
    "são sensíveis à escala e à distribuição (Tan et al., 2017), em duas camadas. As "
    "**determinísticas** — eliminação de atributos e construção de novos por fórmulas e "
    "tabelas clínicas — não dependem da amostra e são aplicadas à base inteira. As "
    "**estatísticas** — imputação, codificação e padronização — dependem de parâmetros "
    "calculados dos dados (mediana, média, desvio, lista de categorias) e por isso são "
    "ajustadas apenas no conjunto de treino (80%, estratificado pelo alvo, semente fixa) "
    "e reaplicadas sem reajuste ao teste, evitando que informação do teste vaze para a "
    "avaliação.")
par("O eixo das construções é que, em pediatria, o valor bruto de IMC, pressão e "
    "frequência cardíaca não tem significado sem idade e sexo: um IMC de 20 kg/m² é "
    "obesidade aos 2 anos e eutrofia aos 15; uma PAS de 110 mmHg é hipertensão aos 2 anos "
    "e normal aos 10. Por isso as medidas são convertidas em classes clínicas segundo "
    "referências pediátricas (Tabela 3).")

legenda("**Tabela 3** — Transformações propostas e justificativas.",
        antes=4, depois=2).paragraph_format.keep_with_next = True
T3 = [
    ["ID, Convênio, HDA2, PPA, IMC, datas", "Eliminação",
     "ID sem informação; convênio sem relação clínica; HDA2 97% ausente; PPA 82% "
     "inutilizável; IMC redundante; datas já usadas na idade"],
    ["Faixa etária (nova)", "Lactente (< 2), pré-escolar (2–5), escolar (6–11), "
     "adolescente (12–19)",
     "Fisiologia e referências mudam por faixa; cortes do domínio, não de largura "
     "ou frequência iguais"],
    ["Classe de IMC (nova)", "Percentil CDC por idade e sexo: baixo peso (< p5), "
     "eutrofia, excesso (p85–p95), obesidade (≥ p95)",
     "IMC absoluto não tem significado sem idade e sexo (Kuczmarski et al., 2002)"],
    ["Classe de pressão (nova)", "NHBPEP: sexo × idade × percentil de altura, PAS ou "
     "PAD; normal, pré-hipertensão, HAS 1, HAS 2",
     "Substitui a PPA com regra explícita e auditável (NHBPEP, 2004)"],
    ["Classe de FC (nova)", "Bradicardia, normal ou taquicardia por faixa etária",
     "FC normal vai de 60 a 190 bpm conforme a idade (MedlinePlus)"],
    ["Sopro presente; nº de queixas", "Indicador binário; contagem de 0 a 2",
     "Sinal clínico direto; a contagem recupera HDA2 antes da eliminação"],
    ["Alvo", "Normal = 0, Anormal = 1",
     "Positivo = doença: recall e F1 referem-se à classe de interesse"],
    ["Classes e faixa etária", "Ordinal; “não avaliável” = −1",
     "Há ordem clínica; a falta da medida é informativa (ver abaixo)"],
    ["Sexo, pulsos, B2, sopro, HDA 1, motivos", "One-hot; ausente → “não informado”; "
     "categorias < 30 registros agregadas",
     "Nominais sem ordem e de baixa cardinalidade"],
    ["Peso, FC", "Log → imputação pela mediana → z-score",
     "Assimetria à direita (peso: 1,37 → −0,17 após o log)"],
    ["Idade, altura, PAS, PAD", "Imputação pela mediana → z-score",
     "Escalas distintas; exigido por métodos de distância ou gradiente"],
    ["Alvo (balanceamento)", "Ponderação de classes",
     "Desbalanceamento leve (57/43): reamostragem dispensável"],
]
tabela(["Atributo", "Transformação", "Justificativa"], T3, [3.7, 5.9, 7.4])

par(antes=6, texto="A classe de pressão reconstruída foi validada contra a PPA original nos 2.057 "
    "registros em que ambas existem: a concordância exata é de 88,2%, e as divergências "
    "são de critério — 128 casos “Normal” na origem são pré-hipertensão pela regra de "
    "120/80 mmHg em adolescentes, e 43 “HAS-2” são HAS 1 porque o estágio 2 começa no "
    "percentil 99 mais 5 mmHg. As classes têm cobertura desigual, porque dependem de "
    "medidas que faltam: faixa etária existe para 93% dos registros, FC para 88%, IMC "
    "para 37% e pressão para 23%. Os demais recebem a categoria “não avaliável”, mantida "
    "de propósito: sua taxa de Anormal (45–47%) supera a dos avaliados, ou seja, a falta "
    "da medida carrega informação e descartá-la introduziria viés. O pré-processador "
    "ajustado no treino gera a matriz de modelagem com 57 colunas numéricas (9.120 "
    "registros de treino e 2.281 de teste, ambos com 42,6% de Anormal).")

# =========================================================================
# REFERENCIAS
# =========================================================================
titulo("Referências")
REFS = [
    "COHEN, J. *Statistical power analysis for the behavioral sciences*. 2. ed. Hillsdale: "
    "Lawrence Erlbaum, 1988.",
    "FLEMING, S. et al. Normal ranges of heart rate and respiratory rate in children from "
    "birth to 18 years of age: a systematic review of observational studies. *The Lancet*, "
    "v. 377, n. 9770, p. 1011–1018, 2011.",
    "FREEDMAN, D. S. et al. Validity of the WHO cutoffs for biologically implausible values "
    "of weight, height, and BMI in children and adolescents in NHANES from 1999 through "
    "2012. *American Journal of Clinical Nutrition*, v. 102, n. 5, p. 1000–1006, 2015.",
    "KUCZMARSKI, R. J. et al. 2000 CDC growth charts for the United States: methods and "
    "development. *Vital and Health Statistics*, série 11, n. 246, 2002.",
    "NHBPEP — NATIONAL HIGH BLOOD PRESSURE EDUCATION PROGRAM WORKING GROUP ON HIGH BLOOD "
    "PRESSURE IN CHILDREN AND ADOLESCENTS. The fourth report on the diagnosis, evaluation, "
    "and treatment of high blood pressure in children and adolescents. *Pediatrics*, "
    "v. 114, n. 2, supl., p. 555–576, 2004.",
    "ONIS, M. de et al. Development of a WHO growth reference for school-aged children and "
    "adolescents. *Bulletin of the World Health Organization*, v. 85, n. 9, p. 660–667, 2007.",
    "TAN, P.-N.; STEINBACH, M.; KARPATNE, A.; KUMAR, V. *Introdução à mineração de dados*. "
    "2. ed. São Paulo: Pearson, 2017.",
    "U.S. NATIONAL LIBRARY OF MEDICINE. Pulse. *MedlinePlus Medical Encyclopedia*. "
    "Bethesda: NIH. Disponível em: https://medlineplus.gov/ency/article/003399.htm.",
    "WHO MULTICENTRE GROWTH REFERENCE STUDY GROUP. *WHO Child Growth Standards*: methods "
    "and development. Genebra: World Health Organization, 2006.",
    "WORLD HEALTH ORGANIZATION. *WHO Anthro for personal computers, version 3.2.2*. "
    "Genebra: WHO, 2011.",
]
for r in REFS:
    p = par(r, tam=Pt(7.5), alinhar=WD_ALIGN_PARAGRAPH.LEFT, depois=1)

doc.save(SAIDA)
print(SAIDA)
