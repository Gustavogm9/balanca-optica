"""Testes offline do scripts/smoke_e2e.py: funções puras (vídeo sintético, payloads,
avaliação do resultado, checklist) e um dry-run do fluxo inteiro com um Supabase falso."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import cv2
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_e2e.py"
spec = importlib.util.spec_from_file_location("smoke_e2e", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
sys.modules["smoke_e2e"] = smoke  # dataclasses resolvem anotações via sys.modules
spec.loader.exec_module(smoke)

UUID_A = "11111111-1111-4111-8111-111111111111"
UUID_B = "22222222-2222-4222-8222-222222222222"
UUID_C = "33333333-3333-4333-8333-333333333333"


# ------------------------------------------------------------------ vídeo sintético

def test_synthetic_video_has_marker_and_dark_body(tmp_path):
    from balanca_optica.phone.marker import detect_marker

    p = smoke.make_synthetic_video(tmp_path / "v.mp4", seconds=3.0, fps=30, marker_id=5)
    cap = cv2.VideoCapture(str(p))
    assert cap.isOpened()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert 85 <= n <= 92  # 3 s a 30 fps
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape == (720, 960, 3)

    hit = detect_marker(frame)
    assert hit is not None and hit.marker_id == 5 and hit.px_per_m > 500 and hit.skew > 0.9

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    body = gray[240:480, 240:720]        # tronco
    bg = gray[20:200, 20:200]            # canto sem corpo nem marcador
    assert body.mean() < 60 and bg.mean() > 130
    assert p.stat().st_size > 1000


def test_synthetic_video_short_duration(tmp_path):
    p = smoke.make_synthetic_video(tmp_path / "s.mp4", seconds=0.2, fps=10, size=(320, 240))
    cap = cv2.VideoCapture(str(p))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    cap.release()


# ------------------------------------------------------------------ payloads

def test_identity_helpers():
    assert smoke.smoke_email(1700000000) == "smoke+1700000000@guilds.test"
    pw = smoke.random_password()
    assert len(pw) >= 20 and any(c.isdigit() for c in pw) and any(c.isupper() for c in pw)
    assert smoke.random_password() != pw
    path = smoke.midia_path_for(UUID_A, "abc")
    assert path == f"{UUID_A}/smoke/abc.mp4"
    assert smoke.midia_path_for(UUID_A).startswith(f"{UUID_A}/smoke/")


def test_entity_payloads_match_schema():
    fz = smoke.build_fazenda("user-1", 123)
    assert {"user_id", "nome", "proprietario", "municipio", "uf", "area_ha"} <= set(fz)
    lt = smoke.build_lote(UUID_A)
    assert lt["fazenda_id"] == UUID_A and lt["categoria"] in ("bezerro", "garrote", "novilha", "vaca", "boi", "touro")
    an = smoke.build_animal(UUID_A, UUID_B, 123)
    assert an["identificador"] == "SMOKE-123" and an["lote_id"] == UUID_B and an["categoria"]
    json.dumps([fz, lt, an])


def test_pending_payload_matches_pwa_contract():
    mp = smoke.midia_path_for(UUID_A, "m")
    p = smoke.build_pesagem_pendente(UUID_A, UUID_B, UUID_C, mp, medido_em="2026-11-15T10:00:00+00:00")
    assert p["fonte"] == "celular" and p["peso_estimado_kg"] == 0  # NOT NULL na tabela; worker preenche
    assert p["midia_path"] == mp and p["metadados"]["midia_path"] == mp
    assert p["metadados"]["status"] == "pendente"
    assert p["fazenda_id"] == UUID_A and p["animal_id"] == UUID_B and p["lote_id"] == UUID_C
    json.dumps(p)


def test_ingestion_payloads_match_edge_function():
    j = smoke.build_ingestao_jwt(UUID_A, UUID_B, UUID_C)
    assert j["fazenda_id"] == UUID_A  # obrigatório com JWT de usuário
    assert j["fonte"] in ("celular", "camera_fixa", "balanca", "manual")
    assert 0 < j["peso_estimado_kg"] <= 5000 and isinstance(j["peso_estimado_kg"], float)
    d = smoke.build_ingestao_device(UUID_B)
    assert "fazenda_id" not in d  # vem do dispositivo
    assert d["fonte"] == "camera_fixa" and 0 <= d["confiabilidade"] <= 1 and d["confiabilidade"] >= 0.7
    assert d["ic_inferior_kg"] <= d["peso_estimado_kg"] <= d["ic_superior_kg"]
    assert isinstance(d["n_frames"], int)
    dev = smoke.build_dispositivo(UUID_A, "a" * 64, 1)
    assert dev["token_hash"] == "a" * 64 and dev["ativo"] is True
    json.dumps([j, d, dev])


# ------------------------------------------------------------------ avaliação do resultado

@pytest.mark.parametrize("row, ok", [
    ({"metadados": {"status": "processada", "scale_src": "aruco"}, "peso_estimado_kg": 410.2, "n_frames": 6}, True),
    ({"metadados": {"status": "processada"}, "peso_estimado_kg": 0}, False),
    ({"metadados": {"status": "erro", "reason": "sem_animal"}, "peso_estimado_kg": 0}, True),
    ({"metadados": {"status": "erro", "reason": "sem_escala"}, "peso_estimado_kg": 0}, False),
    ({"metadados": {"status": "erro", "reason": "KeyError: x"}, "peso_estimado_kg": 0}, False),
    ({"metadados": {"status": "pendente"}, "peso_estimado_kg": 0}, False),
    ({"metadados": {"status": "processando"}, "peso_estimado_kg": 0}, False),
    ({"metadados": {}, "peso_estimado_kg": 0}, False),
])
def test_evaluate_processed(row, ok):
    got, detail = smoke.evaluate_processed(row)
    assert got is ok and detail


def test_expected_animal_weight():
    assert smoke.expected_animal_weight(True, True) == smoke.PESO_DISPOSITIVO_KG
    assert smoke.expected_animal_weight(False, True) == smoke.PESO_BALANCA_KG
    assert smoke.expected_animal_weight(False, False) is None


def test_checklist_abort_and_summary(capsys):
    ck = smoke.Checklist()
    ck.run("a", lambda: "ok")
    ck.run("b", lambda: (_ for _ in ()).throw(ValueError("x")), critical=False)
    with pytest.raises(smoke.StepAborted):
        ck.run("c", lambda: 1 / 0)
    ck.skip("d")
    assert [s.status for s in ck.steps] == ["PASS", "FAIL", "FAIL", "SKIP"]
    assert len(ck.failed) == 2
    out = ck.summary()
    assert "1 PASS, 2 FAIL, 1 SKIP" in out and "[FAIL] c" in out


def test_load_env_missing_prints_instructions(monkeypatch, capsys):
    for k in smoke.REQUIRED_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(smoke, "REPO_ROOT", Path("/nonexistent"))
    assert smoke.load_env() is None
    out = capsys.readouterr().out
    assert "SUPABASE_SERVICE_ROLE_KEY" in out and "ingestao-pesagem" in out


def test_main_exits_2_without_env(monkeypatch):
    for k in smoke.REQUIRED_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(smoke, "REPO_ROOT", Path("/nonexistent"))
    assert smoke.main([]) == 2


# ------------------------------------------------------------------ dry-run com Supabase falso

class _Res:
    def __init__(self, data): self.data = data


class _Q:
    def __init__(self, db, name):
        self.db, self.name, self.filters, self.op, self.payload = db, name, [], "select", None

    def select(self, *_): return self
    def eq(self, c, v): self.filters.append((c, v)); return self
    def insert(self, p): self.op, self.payload = "insert", p; return self
    def delete(self): self.op = "delete"; return self

    def _match(self, r): return all(r.get(c) == v for c, v in self.filters)

    def execute(self):
        rows = self.db.tables.setdefault(self.name, [])
        if self.op == "insert":
            r = {"id": f"{self.name}-{len(rows) + 1:04d}", **self.payload}
            rows.append(r); return _Res([r])
        if self.op == "delete":
            keep = [r for r in rows if not self._match(r)]
            self.db.deleted += len(rows) - len(keep); rows[:] = keep; return _Res([])
        if self.name == "vw_gmd_animal":
            pes = [r for r in self.db.tables.get("pesagens_animais", []) if self._match(r)]
            if not pes:
                return _Res([])
            return _Res([{"animal_id": pes[0]["animal_id"], "n_pesagens": len(pes), "primeiro_peso_kg": 412.5,
                          "ultimo_peso_kg": 420.0, "gmd_kg_dia": 0.25}])
        return _Res([r for r in rows if self._match(r)])


class _Bucket:
    def __init__(self, db): self.db = db
    def upload(self, path, data, opts): self.db.files[path] = data
    def remove(self, paths):
        for p in paths: self.db.files.pop(p)


class _Admin:
    def __init__(self, db): self.db = db
    def create_user(self, attrs):
        self.db.users.append(attrs["email"])
        return types.SimpleNamespace(user=types.SimpleNamespace(id="user-smoke"))
    def delete_user(self, uid): self.db.users.clear()


class _Auth:
    def __init__(self, db): self.db, self.admin = db, _Admin(db)
    def sign_in_with_password(self, c):
        assert c["email"] in self.db.users
        return types.SimpleNamespace(session=types.SimpleNamespace(access_token="jwt." + "x" * 40))
    def sign_out(self): pass


class _Rpc:
    def __init__(self, token): self.token = token
    def execute(self):
        import hashlib
        return _Res(hashlib.sha256(self.token.encode()).hexdigest())


class _FakeDB:
    def __init__(self): self.tables, self.files, self.users, self.deleted = {}, {}, [], 0


class _FakeClient:
    def __init__(self, db): self.db, self.auth = db, _Auth(db)
    def table(self, n): return _Q(self.db, n)
    def rpc(self, name, args): assert name == "hash_device_token"; return _Rpc(args["token"])
    @property
    def storage(self): return types.SimpleNamespace(from_=lambda b: _Bucket(self.db))


def _install_fakes(monkeypatch, db, edge_status=201):
    fake_supabase = types.ModuleType("supabase")
    fake_supabase.create_client = lambda url, key: _FakeClient(db)
    monkeypatch.setitem(sys.modules, "supabase", fake_supabase)

    def fake_edge(url, anon, body, *, jwt=None, device_token=None):
        assert (jwt is None) != (device_token is None)
        if edge_status != 201:
            return edge_status, {"error": "falha simulada"}
        row = {"fazenda_id": db.tables["fazendas"][0]["id"], "peso_estimado_kg": body["peso_estimado_kg"],
               "fonte": body["fonte"], "animal_id": body.get("animal_id"),
               "metadados": {**body.get("metadados", {}), **({"dispositivo_registro_id": db.tables["dispositivos"][0]["id"]}
                                                              if device_token else {"enviado_por_user_id": "user-smoke"})}}
        r = _Q(db, "pesagens_animais").insert(row).execute().data[0]
        # trigger: atualiza o peso do animal
        for a in db.tables["animais"]:
            if a["id"] == body.get("animal_id"):
                a["peso_kg"] = body["peso_estimado_kg"]
        return 201, {"id": r["id"]}
    monkeypatch.setattr(smoke, "_call_edge", fake_edge)

    def fake_worker_main(once=True, limit=10):
        for r in db.tables["pesagens_animais"]:
            if r["metadados"].get("status") == "pendente":
                assert r["midia_path"] in db.files  # o worker baixaria este arquivo
                r["metadados"] = {**r["metadados"], "status": "erro", "reason": "sem_animal"}
    fake_mod = types.ModuleType("balanca_optica.worker.supabase_worker")
    fake_mod.main = fake_worker_main
    monkeypatch.setitem(sys.modules, "balanca_optica.worker.supabase_worker", fake_mod)
    import balanca_optica.worker as worker_pkg  # `from pkg import mod` prefere o atributo do pacote
    monkeypatch.setattr(worker_pkg, "supabase_worker", fake_mod, raising=False)

    for k, v in {"SUPABASE_URL": "https://fake.supabase.co", "SUPABASE_ANON_KEY": "anon",
                 "SUPABASE_SERVICE_ROLE_KEY": "service"}.items():
        monkeypatch.setenv(k, v)


def test_dry_run_full_flow_passes_and_cleans_up(monkeypatch, capsys):
    db = _FakeDB()
    _install_fakes(monkeypatch, db)
    rc = smoke.main(["--seconds", "0.3"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "FAIL" not in out.replace("0 FAIL", "")
    assert "erro aceito (sem_animal)" in out
    assert "animais.peso_kg = 420.0" in out
    # limpeza: tabelas vazias, arquivo e usuário removidos
    assert db.files == {} and db.users == []
    assert all(not rows for rows in db.tables.values()), db.tables


def test_dry_run_keep_leaves_data(monkeypatch, capsys):
    db = _FakeDB()
    _install_fakes(monkeypatch, db)
    rc = smoke.main(["--seconds", "0.3", "--keep"])
    out = capsys.readouterr().out
    assert rc == 0 and "--keep" in out
    assert db.users == ["smoke+" + db.users[0][6:]] and db.files and db.tables["fazendas"]


def test_dry_run_edge_failure_is_reported_not_fatal(monkeypatch, capsys):
    db = _FakeDB()
    _install_fakes(monkeypatch, db, edge_status=403)
    rc = smoke.main(["--seconds", "0.3"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL  9." in out and "FAIL  11." in out
    assert "SKIP  15." in out                      # sem ingestão confiável, trigger não é checado
    assert "PASS  13." in out and "PASS  16." in out  # worker e limpeza seguem normalmente
    assert db.users == [] and db.files == {}
