from pathlib import Path
from balanca_optica.datasets.auction import parse_catalog, write_lots_csv, build_collection_csv

FX = Path(__file__).parent / "fixtures"


def test_layout_individual_a():
    lots = parse_catalog(FX / "cat_individual_a.pdf", verbose=False, hint="touros nelore")
    assert [l.lot for l in lots] == ["01", "02", "03"]
    assert [l.weight_kg for l in lots] == [926.0, 866.0, 819.0]
    assert lots[0].rgn == "11142" and lots[0].birth == "29/06/2024" and lots[0].video_id == "TSJCl4KBmcg"
    assert all(l.sex == "M" and l.lot_size == 1 for l in lots)


def test_layout_lote_medio():
    lots = parse_catalog(FX / "cat_lote_medio.pdf", verbose=False)
    assert len(lots) >= 6 and all(l.layout == "lote_medio" for l in lots)
    assert lots[0].lot == "1" and lots[0].lot_size == 31 and lots[0].weight_kg == 288.0 and lots[0].sex == "M"
    assert all(l.video_id for l in lots)


def test_layout_individual_b():
    lots = parse_catalog(FX / "cat_individual_b.pdf", verbose=False)
    withw = [l for l in lots if l.weight_kg]
    assert withw and withw[0].weight_kg == 947.0 and withw[0].sex == "M" and withw[0].birth == "22/08/2022"
    assert withw[0].layout == "individual_b" and withw[0].video_id


def test_build_collection_csv(tmp_path):
    lots = parse_catalog(FX / "cat_individual_a.pdf", verbose=False, hint="touros nelore")
    csv = write_lots_csv(lots, tmp_path / "lotes.csv", source="fixture")
    vids = tmp_path / "v"; vids.mkdir()
    (vids / "lote02_pV6DB6pU_78.mp4").write_bytes(b"x")
    out = build_collection_csv(csv, vids, tmp_path / "coleta.csv", ref_date="2026-11-14")
    rows = out.read_text().splitlines()
    assert len(rows) == 2 and rows[1].startswith("leilao_11148,866.0,lote02_pV6DB6pU_78.mp4,volta_completa,140")
