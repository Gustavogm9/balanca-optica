# Balança Óptica — motor de estimativa de peso bovino

Núcleo compartilhado pela câmera fixa (bebedouro/brete) e pelo app de celular: transforma
profundidade + segmentação em geometria métrica e geometria em peso, com intervalo de
confiança por leitura e calibração por rebanho.

```
depth.png ─► segmentação por geometria (piso RANSAC, maior componente, corte de cabeça/cauda)
          ─► grade métrica no piso (área, volume, comprimento, larguras, alturas)
          ─► regressor (linear alométrico | PLS | gradient boosting)
          ─► agregação multi-frame (mediana) ─► IC conformal ─► calibração por rebanho
```

## Instalação

```bash
pip install -e ".[dev]"          # núcleo + testes
pip install -e ".[gbm,yolo]"     # opcional: LightGBM e YOLO-seg
pytest -q
```

## Uso (CLI `bo`)

```bash
# 1. features de um dataset rotulado (CowDatabase2: https://github.com/ruchaya/CowDatabase2)
bo extract --dataset cowdatabase2 --root /caminho/CowDatabase2 --out data/processed/cdb2_top.csv

# 2. validação cruzada por animal (nunca por imagem) + relatório com Bland-Altman
bo validate --features data/processed/cdb2_top.csv --out reports/cdb2 --only-ok

# 3. modelo final + intervalo de confiança conformal (90%)
bo train --features data/processed/cdb2_top.csv --out models/cdb2_linear.joblib --model linear --only-ok

# 4. leitura de um animal a partir de 1+ frames de profundidade (mesmo animal)
bo predict --model models/cdb2_linear.joblib --camera kinect_v2 frame1/depth.png frame2/depth.png
# {"weight_kg": 622.2, "n_frames": 2, "rel_mad": 0.03, "reliable": true, "ci_low_kg": 585.1, "ci_high_kg": 659.3}

# 5. calibração por rebanho com 20-30 pesagens da balança do cliente (csv: weight_kg, pred_kg)
bo calibrate --pairs pares.csv --herd fazenda_serra --out models/calib_fazenda_serra.json
bo predict --model models/cdb2_linear.joblib --calibration models/calib_fazenda_serra.json frame/depth.png
```

Presets de câmera em `balanca_optica/config.py` (`kinect_v2`, `femto_bolt_nfov`, `realsense_d455`,
`oak_d_pro`) ou um YAML como `configs/camera_example.yaml`. Recalibrar intrínsecos na instalação.

## Resultados até agora (datasets públicos)

Todos leave-one-animal-out, nunca por imagem. Relatórios com Bland-Altman em `reports/`.

| caminho | dataset | n | modelo | MAPE | referência |
|---|---|---|---|---|---|
| **Celular, 1 foto lateral, COM escala** (cernelha ≈ LiDAR/marcador) | Bai (Horqin, CC BY 4.0), 341-644 kg | 67 | gbm / linear | **7,4% / 8,1%** (R² 0,69) | fita torácica 5,4%; média 15,3% |
| Celular, 1 foto lateral, SEM escala | idem | 67 | linear | 16,8% (pior que a média) | |
| Celular, 5 ângulos, SEM escala, mediana | Rahman (Dhaka, CC BY 4.0), 40-950 kg | 69 | linear | 54% (rótulos arredondados, distância variável) | |
| Câmera fixa dorsal RGB-D | CowDatabase2 (Angus, Kinect v2), 556-746 kg | 91 | linear | 3,9% (média 4,1%; fita 3,6%) | faixa estreitíssima |
| Câmera fixa dorsal RGB-D | CowDB (Hereford, Kinect v2), bloco consistente, 243-605 kg | 54 | linear | 13,7% (fita 4,9%) | pareamento imagem-rótulo do dataset inconsistente |

O que isso prova e o que não prova:

- **Escala é tudo no celular.** Com escala, uma única foto e um segmentador genérico chegam a 7 a 8%,
  igual à literatura (8,9% Brahman 2025). Sem escala, não funciona. Por isso o app exige LiDAR
  ou marcador impresso, e por isso o marcador de A4 já entra no protocolo de coleta.
- Os datasets RGB-D públicos (Ruchay) servem para validar que o pipeline dorsal extrai geometria
  correta (segmentação sem falha em 249 de 250 frames; teste sintético recupera volume com erro
  < 8%), mas **não** medem a precisão do produto: um tem faixa de peso de 5% de variação, o outro
  tem pareamento imagem-rótulo inconsistente (verificado com as medidas manuais independentes).
- A precisão de produto vem do dataset pareado Nelore da Fase 0/1, com faixa larga, vários frames
  por animal e protocolo de pesagem controlado.

## Estrutura

```
balanca_optica/
  capture/               daemon do ponto fixo: fontes (kinect2|openni2|orbbec|realsense|replay),
                         presença, clipes, fila offline, pareamento com a balança
  phone/                 importador de fotos/vídeos de celular (frames, YOLO, escala, qualidade)
  config.py              intrínsecos e presets de sensores
  io/depth.py            PNG 16 bits -> metros -> nuvem (H,W,3)
  segmentation/
    depth_seg.py         piso por RANSAC, máscara, corte de cabeça/cauda, qualidade do frame
    yolo.py              segmentador neural opcional (RGB / celular)
  geometry/
    features.py          grade métrica no piso: área, volume, comprimento, larguras, alturas
    silhouette.py        features 2D com escala (LiDAR / marcador) para o celular
  model/
    regressor.py         linear | pls | gbm, log-target, metadados categóricos
    uncertainty.py       intervalo conformal por leitura
    calibration.py       calibração por rebanho com encolhimento bayesiano
  aggregate.py           mediana multi-frame + confiabilidade
  eval/                  métricas (MAPE, MAE, Bland-Altman, erro de lote), CV agrupada, relatório
  datasets/              loaders (cowdatabase2, cowdb, mendeley bai/rahman)
  cli.py                 bo extract | validate | train | predict | calibrate | capture | pair | import-phone
tools/coleta.html        formulário de coleta offline para o celular (exporta CSV)
docs/hardware.md         kit de investimento mínimo (celular + Kinect v2 usado)
tests/                   sintético com volume analítico, regressor, conformal, calibração, daemon, importador
```

## Fluxo de campo (Fase 0, custo ~zero)

```bash
# no celular: abra tools/coleta.html, registre brinco + peso + nome do vídeo por animal
# no computador:
bo import-phone --csv coleta.csv --media ./videos --out data/processed/serra_phone.csv
bo validate --features data/processed/serra_phone.csv --out reports/serra_phone --only-ok

# ponto fixo com Kinect v2 usado (ver docs/hardware.md)
bo capture --source kinect2 --out ~/bo_capture --site fazenda_serra_balanca
bo pair --readings ~/bo_capture/readings.csv --scale balanca.csv --out pairs.csv
```

## Regras do projeto

- Split sempre por animal; a partir de várias fazendas, leave-one-farm-out.
- Toda leitura sai com intervalo de confiança e selo de qualidade do frame.
- Modelos simples primeiro (linear/PLS); CNN só com milhares de animais.
- Toda pesagem oficial do cliente vira teste (e depois treino).

## Próximos passos

1. Coleta na Fazenda Serra: 20 a 30 animais pesados + vídeo de celular com marcador A4 (esta semana).
2. Kinect v2 usado na saída da balança; `bo capture` gravando; `bo pair` montando o dataset.
3. Detecção automática do marcador (ArUco/xadrez) no importador e no app.
4. Fine-tuning YOLO-seg com ~300 máscaras de Nelore (cupim, barbela, umbigo).
5. Uploader da fila (Supabase Storage) e dashboard interno de acurácia por fazenda.
6. App Expo: captura guiada, ArUco, LiDAR (ARKit), inferência local ONNX.
