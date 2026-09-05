"""Loaders dos dois datasets de FOTO DE CELULAR com peso (licença CC BY 4.0).

- Bai (Mendeley h2s22wr5py v3): 72 bovinos Horqin, vista lateral + traseira, com peso,
  comprimento oblíquo, cernelha, perímetro torácico. A cernelha permite simular a
  escala que o LiDAR daria (scale_src = known_height).
- Rahman (Mendeley vf7pxfs7dx v1): 72 vacas, 5 ângulos por animal, peso (arredondado,
  faixa 40 a 1300 kg com outliers), sem escala.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..phone.import_media import PhoneRecord, process_media


def bai_records(root: str | Path, use_height_scale: bool = True) -> list[PhoneRecord]:
    root = Path(root)
    base = next(root.rglob("measurements.xlsx")).parent
    df = pd.read_excel(base / "measurements.xlsx")
    df.columns = [c.strip() for c in df.columns]
    recs = []
    for _, r in df.iterrows():
        n = int(r["Num"])
        meta = {"farm": "horqin", "breed": "horqin", "withers_cm": r["Withers height(cm)"],
                "heart_girth_cm": r["Heart girth(cm)"], "body_length_cm": r["Oblique body length (cm)"]}
        for view, folder in (("lateral", "side view"), ("traseira", "back view")):
            cands = list((base / folder).glob(f"{n}.*"))
            if not cands:
                continue
            recs.append(PhoneRecord(animal_id=f"bai_{n}", weight_kg=float(r["Body weight (kg)"]),
                                    file=cands[0], view=view,
                                    height_cm=float(r["Withers height(cm)"]) if (use_height_scale and view == "lateral") else None,
                                    meta=meta))
    return recs


def rahman_records(root: str | Path) -> list[PhoneRecord]:
    root = Path(root)
    base = next(root.rglob("labels.csv")).parent
    lab = pd.read_csv(base / "labels.csv")
    recs = []
    for _, r in lab.iterrows():
        cid = int(r["cow_id"])
        for a in range(1, 6):
            f = base / f"Angle{a}" / f"cow_{cid:04d}_angle{a}.jpg"
            if f.exists():
                recs.append(PhoneRecord(animal_id=f"rahman_{cid}", weight_kg=float(r["weight_kg"]),
                                        file=f, view=f"angle{a}", meta={"farm": "dhaka", "breed": "mixed"}))
    return recs


def extract(records: list[PhoneRecord], weights: str = "yolov8n-seg.pt", verbose: bool = False) -> pd.DataFrame:
    from ..segmentation.yolo import YoloSegmenter
    seg = YoloSegmenter(weights, conf=0.3)
    return process_media(records, seg, verbose=verbose)
