"""Fine-tuning do segmentador para bovinos (Nelore em curral: cupim, barbela, umbigo).

Fluxo de dados (formato YOLO-seg, uma classe: 0 = bovino):
  1. `pseudo_label(images, out)`   roda o YOLO genérico (classe COCO 'cow') sobre uma pasta
     de fotos/frames e grava as máscaras como labels YOLO-seg + data.yaml. Isso vira o
     ponto de partida para correção humana (CVAT, Roboflow, Label Studio) em vez de anotar
     do zero: corrigir 300 máscaras leva horas; desenhar 300 leva dias.
  2. `merge(sources, out)`         junta datasets YOLO-seg (público + Nelore corrigido),
     remapeando qualquer classe de bovino para 0 e descartando as outras.
  3. `train(data_yaml, ...)`       fine-tune a partir de yolov8n-seg.pt; devolve o caminho do
     best.pt, que entra em `YoloSegmenter(weights=...)`.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _write_yaml(root: Path, splits: dict[str, str]) -> Path:
    d = {"path": str(root.resolve()), "nc": 1, "names": ["bovino"], **splits}
    p = root / "data.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


def mask_to_polygons(mask: np.ndarray, min_area: int = 200) -> list[np.ndarray]:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        eps = 0.002 * cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(c) >= 3:
            polys.append(c)
    return polys


def pseudo_label(images_dir: str | Path, out_dir: str | Path, weights: str = "yolov8n-seg.pt",
                 conf: float = 0.3, val_frac: float = 0.2, max_side: int = 1280,
                 verbose: bool = True) -> Path:
    """Gera labels YOLO-seg com o modelo atual. Todas as instâncias 'cow' viram classe 0."""
    from ultralytics import YOLO  # type: ignore
    from .yolo import COCO_COW

    model = YOLO(weights)
    out = Path(out_dir)
    for s in ("train", "val"):
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "labels").mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(images_dir).rglob("*") if p.suffix.lower() in IMG_EXT)
    n_lab = 0
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None:
            continue
        s = min(1.0, max_side / max(img.shape[:2]))
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        res = model.predict(img, conf=conf, verbose=False)[0]
        lines = []
        if res.masks is not None:
            for m, c in zip(res.masks.data.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                if int(c) != COCO_COW:
                    continue
                m = cv2.resize(m.astype(np.uint8), (w, h)) > 0
                for poly in mask_to_polygons(m):
                    xy = (poly / np.array([w, h])).clip(0, 1).ravel()
                    lines.append("0 " + " ".join(f"{v:.5f}" for v in xy))
        split = "val" if (i % int(round(1 / val_frac)) == 0) else "train"
        stem = f"{f.parent.name}_{f.stem}".replace(" ", "_")
        cv2.imwrite(str(out / split / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / split / "labels" / f"{stem}.txt").write_text("\n".join(lines))
        n_lab += bool(lines)
    if verbose:
        print(f"{len(files)} imagens, {n_lab} com máscara -> {out} (corrigir no CVAT/Roboflow antes de treinar)")
    return _write_yaml(out, {"train": "train/images", "val": "val/images"})


def merge(sources: list[str | Path], out_dir: str | Path, bovine_names=("cow", "cattle", "bovino", "boi", "vaca"),
          verbose: bool = True) -> Path:
    """Junta datasets YOLO-seg; classes cujo nome contém um termo bovino viram 0, o resto é descartado."""
    out = Path(out_dir)
    for s in ("train", "val"):
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "labels").mkdir(parents=True, exist_ok=True)
    total = 0
    for src in sources:
        src = Path(src)
        cfg = yaml.safe_load((src / "data.yaml").read_text())
        names = cfg["names"]
        names = names if isinstance(names, list) else [names[k] for k in sorted(names)]
        keep = {i for i, n in enumerate(names) if any(b in str(n).lower() for b in bovine_names)}
        for split_key, dst in (("train", "train"), ("val", "val"), ("valid", "val"), ("test", "val")):
            rel = cfg.get(split_key)
            cands = [src / split_key / "images"]
            if rel:
                cands = [Path(rel) if Path(rel).is_absolute() else (src / rel), (src / Path(rel).name)] + cands
            img_dir = next((c for c in cands if c.exists()), None)
            if img_dir is None:
                continue
            lab_dir = img_dir.parent / "labels"
            for img in sorted(img_dir.glob("*")):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                lab = lab_dir / (img.stem + ".txt")
                lines = []
                if lab.exists():
                    for ln in lab.read_text().splitlines():
                        parts = ln.split()
                        if parts and int(float(parts[0])) in keep:
                            lines.append("0 " + " ".join(parts[1:]))
                stem = f"{src.name}_{img.stem}"
                shutil.copy(img, out / dst / "images" / (stem + img.suffix.lower()))
                (out / dst / "labels" / (stem + ".txt")).write_text("\n".join(lines))
                total += 1
    if verbose:
        print(f"{total} imagens mescladas em {out}")
    return _write_yaml(out, {"train": "train/images", "val": "val/images"})


def train(data_yaml: str | Path, weights: str = "yolov8n-seg.pt", epochs: int = 50, imgsz: int = 640,
          batch: int = 8, project: str = "runs/seg", name: str = "bovino", device=None, **kw) -> Path:
    from ultralytics import YOLO  # type: ignore
    model = YOLO(weights)
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, project=project,
                name=name, exist_ok=True, device=device, single_cls=True, verbose=False, plots=False, **kw)
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    return best if best.exists() else save_dir / "weights" / "last.pt"


def evaluate(weights: str | Path, data_yaml: str | Path, imgsz: int = 640, device=None) -> dict:
    from ultralytics import YOLO  # type: ignore
    r = YOLO(str(weights)).val(data=str(data_yaml), imgsz=imgsz, device=device, verbose=False, plots=False)
    return {"mask_map50": float(r.seg.map50), "mask_map50_95": float(r.seg.map), "box_map50": float(r.box.map50)}
