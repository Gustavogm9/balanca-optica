# Acelerador de dataset: catálogos de leilão + vídeo por lote

Leiloeiras publicam, por animal ou por lote, o peso da pesagem pré-leilão e um vídeo no
YouTube. Verificado em 14/11/2026 no Programa Leilões: um leilão de touros Nelore rende
~90 pares individuais (656 a 926 kg); um leilão virtual de bezerros rende ~77 lotes com peso
médio cobrindo ~2.300 animais (280 a 350 kg).

## Fluxo

```bash
# 1. a partir da página do leilão (baixa o catálogo e lista os vídeos)
bo auction discover --url "https://programaleiloes.com/agenda/leilao/91529-leilao-touros-nelore-mafra-redencao-pa" --out data/leiloes/mafra

# 2. baixar os vídeos (na SUA máquina: YouTube bloqueia datacenter; --cookies chrome se pedir login)
bo auction fetch --csv data/leiloes/mafra/lotes.csv --out data/leiloes/mafra/videos --cookies chrome

# 3. CSV de coleta no formato do importador, e importação
bo auction build --csv data/leiloes/mafra/lotes.csv --videos data/leiloes/mafra/videos --out data/leiloes/mafra/coleta.csv --ref-date 2026-11-14
bo import-phone --csv data/leiloes/mafra/coleta.csv --media data/leiloes/mafra/videos --out data/processed/mafra.csv
```

`bo auction parse --pdf catalogo.pdf --out lotes.csv --hint "touros nelore"` serve para um
PDF avulso. Layouts reconhecidos: individual A ("Peso: 926kg", RGN, Nasc.), individual B
("Nasc: Sexo: CE: ... M22/08/2022 947 40") e lote médio ("QUANTIDADE 31 ... 288 PESO MÉDIO").
Leilões de genética (prenhez, aspiração) não têm peso e saem vazios.

## Regras de uso

- O peso é da pesagem pré-leilão, não do dia do vídeo: ruído de 1 a 3%. Serve para TREINO.
  A validação final continua sendo balança calibrada no mesmo dia (Fazenda Serra).
- Não há marcador de escala no vídeo. `build` preenche `height_cm` com a cernelha típica da
  categoria (idade + sexo) como escala aproximada; marque esses pares com `farm=leilao` e
  nunca os misture no conjunto de teste.
- Lotes com peso médio (`lot_size > 1`) só entram com `--include-lots`; servem para o modo
  lote e para segmentação, não para regressão individual.
- Catálogo é público; vídeo é YouTube padrão. Uso interno para treino é zona cinzenta;
  publicar o dataset ou citar em artigo exige autorização da leiloeira. Pedir permissão
  formal antes de escalar, oferecendo em troca o estimador para o comprador virtual.
