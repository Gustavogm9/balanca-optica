"""Acelerador de dataset: catálogos de leilão (PDF) + vídeo por lote.

Leiloeiras publicam, por animal, RGN, peso de pesagem pré-leilão, nascimento e um vídeo
no YouTube (lateral/traseira, filmado de celular ou câmera). O par vídeo + peso já existe;
este módulo o extrai:

  bo auction parse  --pdf catalogo.pdf --out leilao.csv         # lotes, peso, RGN, nasc, URL do vídeo
  bo auction fetch  --csv leilao.csv --out videos/ [--cookies]   # baixa os vídeos (yt-dlp; rodar numa rede
                                                                  # doméstica: datacenter é bloqueado pelo YouTube)
  bo auction build  --csv leilao.csv --videos videos/ --out coleta.csv   # CSV no formato do importador

Depois: bo import-phone --csv coleta.csv --media videos/ --out features.csv

Avisos que ficam gravados no CSV (coluna `notes`): o peso é da pesagem pré-leilão, não do dia
da filmagem (defasagem típica de dias, ruído de 1 a 3%: serve para treino, não para o teste
final); não há marcador de escala no vídeo (scale_src=none ou known_height por categoria);
uso de vídeos de terceiros exige autorização da leiloeira para publicar ou citar.
"""
from __future__ import annotations

import csv
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

RE_LOTE = re.compile(r"LOTE\s*\|?\s*(?:INDIVIDUAL)*\s*\|?\s*(\d{2,4}[A-Z]?)")
RE_PESO = re.compile(r"Peso:\s*([\d.,]+)\s*kg", re.I)
RE_RGN = re.compile(r"RGN:\s*([A-Z0-9\-]+)", re.I)
RE_NASC = re.compile(r"Nasc\.?:\s*(\d{2}/\d{2}/\d{4})")
RE_SERIE = re.compile(r"S[ée]rie:\s*([A-Z0-9]+)")
RE_SEXO = re.compile(r"\b(TOURO|TOUROS|MATRIZ|MATRIZES|NOVILHA|NOVILHAS|BEZERR[OA]S?|GARROTES?|VACAS?|MACHOS?|F[ÊE]MEAS?)\b", re.I)
RE_YT = re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_\-]{11})")


@dataclass
class Lot:
    lot: str
    rgn: str | None
    weight_kg: float | None
    birth: str | None
    serie: str | None
    video_id: str | None
    page: int
    title_hint: str | None = None
    sex: str | None = None  # M | F quando o catálogo diz
    lot_size: int = 1  # >1 = lote de vários animais; weight_kg é a MÉDIA
    layout: str = "individual_a"


def _page_links(page) -> list[str]:
    out = []
    try:
        for a in page.annotations or []:
            o = a.get_object()
            uri = (o.get("/A") or {}).get("/URI")
            if uri:
                out.append(str(uri))
    except Exception:  # noqa: BLE001
        pass
    return out


def parse_catalog(pdf_path: str | Path, verbose: bool = True, hint: str | None = None) -> list[Lot]:
    from pypdf import PdfReader

    r = PdfReader(str(pdf_path))
    lots: list[Lot] = []
    title = " ".join((pg.extract_text() or "")[:600] for pg in r.pages[:3]).replace("\n", " ")
    sex_hint = None
    m = RE_SEXO.search(title + " " + (hint or ""))
    if m:
        sex_hint = m.group(1).upper()
    for i, page in enumerate(r.pages):
        text = page.extract_text() or ""
        flat = re.sub(r"\s+", " ", text)
        links = [RE_YT.search(l).group(1) for l in _page_links(page) if RE_YT.search(l)]
        # Layout C: lotes coletivos ("QUANTIDADE 314 PESO MÉDIO (KG)" ou "29 QUANTIDADE 314 PESO MÉDIO")
        if re.search(r"PESO M[ÉE]DIO", flat, re.I) and "Peso:" not in flat:
            parts = re.split(r"L\s*O\s*T\s*E", flat)
            segs = parts[1:]
            k = 0
            for j, seg in enumerate(segs):
                prev_tail = re.search(r"(\d{1,4}[A-Z]?)\s*$", parts[j])  # número do lote vem ANTES de "LOTE"
                mq = re.search(r"(\d{1,4})\s*QUANTIDADE\s*(\d{2,4})\s*PESO M[ÉE]DIO", seg, re.I) or \
                     re.search(r"QUANTIDADE\s*(\d{1,4})\D{0,40}?(\d{2,4})\s*PESO M[ÉE]DIO", seg, re.I)
                if not mq:
                    continue
                qty, wmean = int(mq.group(1)), float(mq.group(2))
                mc = re.search(r"CATEGORIA\s*([A-ZÇÃÕ ]+?)(?:\s+(?:DESMAMA|BATERIA|SAFRA|\d)|$)", seg, re.I)
                sex = "M" if re.search(r"MACHO", seg, re.I) else ("F" if re.search(r"F[ÊE]MEA", seg, re.I) else None)
                vid = links[k] if k < len(links) else None
                k += 1
                lots.append(Lot(prev_tail.group(1) if prev_tail else f"p{i}_{k}", None, wmean, None, None, vid, i + 1,
                                (mc.group(1).strip() if mc else sex_hint), sex, qty, "lote_medio"))
            continue
        if "Peso" not in text and "RGN" not in text and not re.search(r"Nasc:?\s*Sexo", flat):
            continue
        lot_m = re.search(r"LOTE\s+(?:INDIVIDUAL)*\s*(\d{2,4}[A-Z]?)", flat) or re.search(r"LOTE[^\d]{0,30}(\d{2,4}[A-Z]?)", flat)
        lot_raw = lot_m.group(1) if lot_m else f"p{i}"
        if len(lot_raw) in (4, 6) and lot_raw[: len(lot_raw) // 2] == lot_raw[len(lot_raw) // 2:]:
            lot_raw = lot_raw[: len(lot_raw) // 2]
        peso = RE_PESO.search(text)
        rgn = RE_RGN.search(text)
        nasc = RE_NASC.search(text)
        serie = RE_SERIE.search(text)
        layout = "individual_a"
        w = float(peso.group(1).replace(".", "").replace(",", ".")) if peso else None
        sex = "F" if re.search(r"F[ÊE]MEA", text) else ("M" if re.search(r"\bMACHO\b", text) else None)
        if w is None:
            # Layout B: "Nasc: Sexo: CE: ... M22/08/2022 947 40" (sexo colado na data, peso, CE)
            mb = re.search(r"\b([MF])\s?(\d{2}/\d{2}/\d{4})\s+(\d{3,4})\s+(\d{2})\b", flat)
            if mb:
                sex, w, layout = mb.group(1), float(mb.group(3)), "individual_b"
                nasc = mb.__class__ and re.search(r"(\d{2}/\d{2}/\d{4})", mb.group(0))
        if sex is None and sex_hint:
            sex = "M" if re.search(r"TOURO|MACHO|GARROTE", sex_hint) else ("F" if re.search(r"MATRIZ|NOVILHA|VACA|F[ÊE]MEA", sex_hint) else None)
        vid = links[0] if links else (RE_YT.search(text).group(1) if RE_YT.search(text) else None)
        if rgn is None:
            mr = re.search(r"\bRG:?\s*([A-Z]{2,5}\s?\d{2,6})", flat)
            rgn_val = mr.group(1).replace(" ", "") if mr else None
        else:
            rgn_val = rgn.group(1)
        lots.append(Lot(lot_raw, rgn_val, w, nasc.group(1) if nasc else None,
                        serie.group(1) if serie else None, vid, i + 1, sex_hint, sex, 1, layout))
    if verbose:
        n_w = sum(1 for l in lots if l.weight_kg)
        n_v = sum(1 for l in lots if l.video_id)
        n_both = sum(1 for l in lots if l.weight_kg and l.video_id)
        n_ind = sum(1 for l in lots if l.weight_kg and l.video_id and l.lot_size == 1)
        print(f"{len(lots)} lotes; {n_w} com peso; {n_v} com vídeo; {n_both} pares utilizáveis ({n_ind} individuais, {n_both - n_ind} lotes com peso médio)")
    return lots


def write_lots_csv(lots: list[Lot], out: str | Path, source: str = "") -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [f.name for f in Lot.__dataclass_fields__.values()] + ["video_url", "source"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for l in lots:
            d = asdict(l)
            d["video_url"] = f"https://www.youtube.com/watch?v={l.video_id}" if l.video_id else ""
            d["source"] = source
            w.writerow(d)
    return out


def discover(auction_url: str, out_dir: str | Path, verbose: bool = True) -> dict:
    """Página de leilão do Programa Leilões -> baixa o catálogo PDF e lista os vídeos por lote.
    Devolve {"pdf": caminho, "videos": {lote: video_id}}. Funciona com o HTML público da página."""
    import urllib.request
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(auction_url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
    m = re.search(r'href="((?:https?://[^"/]+)?/leilao/download/\d+-[^"]*)"', html, re.I)
    pdf_path = None
    if m:
        from urllib.parse import urljoin
        pdf_url = urljoin(auction_url, m.group(1))
        pdf_path = out / "catalogo.pdf"
        req2 = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0"})
        pdf_path.write_bytes(urllib.request.urlopen(req2, timeout=120).read())
    videos = {}
    for lm in re.finditer(r'href="(?:https?:)?//(?:www\.)?youtube\.com/watch\?v=([A-Za-z0-9_\-]{11})"[^>]*>(?:(?!</a>).)*?LOTE\s*([0-9]+[A-Z]?)', html, re.S | re.I):
        videos.setdefault(lm.group(2).lstrip("0") or "0", lm.group(1))
    if verbose:
        print(f"catálogo: {pdf_path}; {len(videos)} vídeos por lote na página")
    return {"pdf": str(pdf_path) if pdf_path else None, "videos": videos}


def fetch_videos(csv_path: str | Path, out_dir: str | Path, cookies_from_browser: str | None = None,
                 max_height: int = 720, limit: int | None = None, verbose: bool = True) -> list[Path]:
    """Baixa os vídeos com yt-dlp. Rode na sua máquina (YouTube bloqueia datacenter);
    --cookies-from-browser chrome ajuda se pedir login. Idempotente: pula o que já existe."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    done: list[Path] = []
    n = 0
    for r in rows:
        if not r.get("video_id") or not r.get("weight_kg"):
            continue
        if limit and n >= limit:
            break
        n += 1
        stem = f"lote{r['lot']}_{r['video_id']}"
        existing = list(out.glob(stem + ".*"))
        if existing:
            done.append(existing[0])
            continue
        cmd = ["yt-dlp", "-q", "--no-warnings", "-f", f"bv*[height<={max_height}]+ba/b[height<={max_height}]",
               "--merge-output-format", "mp4", "-o", str(out / (stem + ".%(ext)s")), r["video_url"]]
        if cookies_from_browser:
            cmd += ["--cookies-from-browser", cookies_from_browser]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            got = list(out.glob(stem + ".*"))
            if got:
                done.append(got[0])
                if verbose:
                    print(f"ok  {got[0].name}")
        except subprocess.CalledProcessError as e:  # noqa: PERF203
            msg = e.stderr.decode(errors="ignore").strip().splitlines()[-1:] if e.stderr else ""
            if verbose:
                print(f"falhou lote {r['lot']}: {msg}")
        except subprocess.TimeoutExpired:
            if verbose:
                print(f"timeout lote {r['lot']}")
    return done


def _category(birth: str | None, sex_hint: str | None, ref: str | None = None) -> tuple[str, float | None]:
    """Categoria aproximada e altura de cernelha típica (m) para escala aproximada quando não há marcador.
    Valores de referência para Nelore adulto/jovem; só para treino, nunca para validação."""
    import datetime as dt
    months = None
    if birth:
        try:
            d = dt.datetime.strptime(birth, "%d/%m/%Y").date()
            today = dt.date.fromisoformat(ref) if ref else dt.date.today()
            months = (today.year - d.year) * 12 + today.month - d.month
        except ValueError:
            pass
    male = bool(sex_hint and re.search(r"TOURO|MACHO|GARROTE", sex_hint))
    if months is None:
        return ("desconhecida", None)
    if months < 12:
        return ("bezerro", 1.05)
    if months < 24:
        return ("garrote" if male else "novilha", 1.25 if male else 1.20)
    if months < 36:
        return ("touro_jovem" if male else "novilha", 1.40 if male else 1.30)
    return ("touro" if male else "vaca", 1.48 if male else 1.38)


def build_collection_csv(csv_path: str | Path, videos_dir: str | Path, out: str | Path,
                         farm: str = "leilao", breed: str = "nelore", use_height_scale: bool = True,
                         ref_date: str | None = None, include_lots: bool = False) -> Path:
    """Gera o CSV que `bo import-phone` lê (animal_id, weight_kg, file, view, height_cm, farm, breed, sex, category)."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    vdir = Path(videos_dir)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["animal_id", "weight_kg", "file", "view", "height_cm", "farm", "breed", "sex", "category", "notes"])
        for r in rows:
            if not r.get("weight_kg") or not r.get("video_id"):
                continue
            if int(r.get("lot_size") or 1) > 1 and not include_lots:
                continue
            files = list(vdir.glob(f"lote{r['lot']}_{r['video_id']}.*"))
            if not files:
                continue
            sex = r.get("sex") or ""
            cat, h = _category(r.get("birth") or None, (r.get("title_hint") or "") + (" TOURO" if sex == "M" else " FEMEA" if sex == "F" else ""), ref_date)
            if not sex:
                sex = "M" if cat.startswith(("touro", "garrote")) else ("F" if cat in ("novilha", "vaca") else "")
            aid = f"{farm}_{r.get('rgn') or r['lot']}"
            w.writerow([aid, r["weight_kg"], files[0].name, "volta_completa",
                        round(h * 100) if (use_height_scale and h) else "", farm, breed, sex, cat,
                        ("LOTE de %s animais, peso MEDIO; " % r.get("lot_size") if int(r.get("lot_size") or 1) > 1 else "") +
                        "peso pre-leilao; sem marcador; escala por cernelha tipica (treino apenas)"])
            n += 1
    print(f"{n} registros -> {out}")
    return out
