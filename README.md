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

## Resultado atual (validação do pipeline)

CowDatabase2, 96 Angus, Kinect v2, um frame dorsal por animal, leave-one-animal-out
(`reports/cowdatabase2_top/`):

| modelo | MAPE | MAE | viés | lote de 30 |
|---|---|---|---|---|
| linear (log-alométrico) | 3,9% | 24 kg | -0,1% | 0,6% |
| média do rebanho (baseline) | 4,1% | 26 kg | 0 | — |
| fita torácica manual (referência) | 3,6% | — | — | — |

Leitura honesta: esta base tem faixa de peso estreitíssima (556 a 746 kg, CV 5%) e
ground truth ruidoso (a própria circunferência torácica manual só correlaciona r = 0,54
com o peso). Ela valida que o pipeline extrai geometria correta (teste sintético recupera
volume com erro < 8%) e roda ponta a ponta; **não** mede a precisão que o produto terá.
A precisão vem do dataset pareado Nelore da Fase 1 (faixa larga, vários frames por animal,
protocolo de pesagem controlado).

## Estrutura

```
balanca_optica/
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
  datasets/              loaders (cowdatabase2)
  cli.py                 bo extract | validate | train | predict | calibrate
tests/                   sintético com volume analítico, regressor, conformal, calibração
```

## Regras do projeto

- Split sempre por animal; a partir de várias fazendas, leave-one-farm-out.
- Toda leitura sai com intervalo de confiança e selo de qualidade do frame.
- Modelos simples primeiro (linear/PLS); CNN só com milhares de animais.
- Toda pesagem oficial do cliente vira teste (e depois treino).

## Próximos passos (Fase 0/1)

1. Software de captura na borda (Jetson): loop RGB-D, detecção, gravação de clipes, fila offline,
   sincronização com a balança.
2. Loader genérico do formato de captura Guilds (`<sessao>/<animal_id>/<ts>/depth.png|rgb.png|meta.json`).
3. Fine-tuning YOLO-seg com ~300 máscaras de Nelore em curral (cupim e barbela).
4. Vistas laterais (CowDatabase2 tem left/right) e fusão multi-vista.
5. Regressor por categoria com efeito de fazenda; re-treino mensal; monitor de drift.
6. App: captura guiada + LiDAR (ARKit) + `silhouette.py`; inferência local ONNX.
