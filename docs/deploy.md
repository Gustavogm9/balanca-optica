# Deploy do worker Balança Óptica

O worker (`bo worker --loop`) lê a fila `pesagens_animais` do Pasto Leve no Supabase, baixa o vídeo do bucket `pesagens-midia`, estima o peso e grava o resultado. Ele precisa de duas variáveis, e só:

| Variável | Valor | Observação |
|---|---|---|
| `SUPABASE_URL` | `https://<ref>.supabase.co` | Project Settings > API |
| `SUPABASE_SERVICE_ROLE_KEY` | chave `service_role` | ignora RLS; só no servidor do worker, nunca no app nem em `VITE_*` |

Opcionais: `BO_MODEL` (padrão `models/phone_v0_bai.joblib`), `BO_SEG_WEIGHTS` (`yolov8n-seg.pt`), `BO_POLL_SECONDS` (30), `BO_LOG_LEVEL` (INFO). Todos já têm padrão no `Dockerfile`/`fly.toml`.

Requisitos de máquina: CPU x86 com 2 GB de RAM (torch CPU + YOLOv8n-seg). Sem GPU: um vídeo de 10 s leva de 5 a 20 s.

Antes de qualquer opção: migrations do Pasto Leve aplicadas no projeto Supabase e Edge Function `ingestao-pesagem` publicada (`docs/supabase-do-zero.md` no repositório `pasto-leve`). Valide o projeto com `python scripts/smoke_e2e.py` (ver fim deste documento).

Três opções, da mais simples para a mais trabalhosa.

## Opção 1: Fly.io (recomendada)

Um processo sempre ligado, sem porta pública, em São Paulo (`gru`). O `fly.toml` já está no repositório com dois processos: `worker` (obrigatório) e `api` (`bo serve --port 8080`, opcional, com serviço HTTP e health check em `/health`).

```bash
# 1. CLI e login
curl -L https://fly.io/install.sh | sh
flyctl auth login

# 2. cria o app a partir do fly.toml, sem publicar ainda
flyctl launch --no-deploy --copy-config --name bo-worker --region gru

# 3. segredos (ficam criptografados no Fly; nunca no fly.toml)
flyctl secrets set SUPABASE_URL="https://<ref>.supabase.co" SUPABASE_SERVICE_ROLE_KEY="<service_role>"

# 4. publica (build remoto do Dockerfile: torch CPU + ultralytics, ~5 min na primeira vez)
flyctl deploy

# 5. confere
flyctl status
flyctl logs -a bo-worker
```

Por padrão o Fly cria uma máquina por processo. O worker não precisa da API; desligue-a se não for usar:

```bash
flyctl scale count worker=1 api=0
```

Para ligar a API depois: `flyctl scale count api=1`. Ela sobe em `https://bo-worker.fly.dev` (`GET /health`, `POST /estimate`), com `auto_stop_machines` para não gastar quando ociosa. Se o nome `bo-worker` estiver tomado, troque em `app = "..."` no `fly.toml` e no `--name`.

Atualizar: `git pull && flyctl deploy`. Trocar de projeto Supabase: repita o passo 3 e `flyctl deploy` (ou `flyctl machine restart`).

Custo: uma `shared-cpu-2x` com 2 GB fica na faixa de US$ 10 a 15/mês.

## Opção 2: VPS com Docker Compose

Qualquer VPS Linux com 2 GB (Hetzner, DigitalOcean, Oracle Free Tier, Contabo). O `docker-compose.yml` do repositório já define `worker` e `api`.

```bash
# na VPS
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin git
git clone <repo> balanca-optica && cd balanca-optica

cp .env.example .env
nano .env                      # SUPABASE_URL e SUPABASE_SERVICE_ROLE_KEY
sed -i 's/env_file: .env.example/env_file: .env/' docker-compose.yml

docker compose up -d --build worker        # só o worker
# docker compose up -d --build             # worker + API na porta 8000
docker compose logs -f worker
```

`restart: unless-stopped` religa o worker após reboot. Atualizar: `git pull && docker compose up -d --build`. Se publicar a API, coloque um proxy com TLS na frente (Caddy resolve em três linhas) e restrinja quem chama.

## Opção 3: GitHub Actions a cada 10 minutos (sem servidor)

Para volume baixo e sem pressa: o workflow `.github/workflows/worker-cron.yml` roda `bo worker --once --limit 20` em um runner do GitHub a cada 10 minutos e também sob demanda (`workflow_dispatch`).

1. No repositório, Settings > Secrets and variables > Actions > New repository secret:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
2. Faça push do workflow para `main`. O cron só é agendado a partir da branch padrão.
3. Teste na hora: Actions > Worker (cron) > Run workflow.

Limitações: latência de até 10 minutos mais a fila do Actions (o GitHub pode atrasar crons em horário de pico); cada execução instala torch CPU (~2 min) antes de processar; repositório privado consome a franquia de minutos (2.000/mês no plano gratuito, 15 min por execução no pior caso). `concurrency.cancel-in-progress: false` garante que um ciclo não seja interrompido no meio, o que deixaria registros presos em `processando`. Se precisar de menos latência, mude para Fly.

## Validar o deploy

Com o worker rodando em qualquer das opções, o smoke test ponta a ponta cria um usuário, fazenda, animal e vídeo de teste, chama a Edge Function pelas duas autenticações e roda o worker em processo contra o mesmo projeto:

```bash
pip install -e ".[worker,yolo,gbm]" requests
export SUPABASE_URL=... SUPABASE_ANON_KEY=... SUPABASE_SERVICE_ROLE_KEY=...
python scripts/smoke_e2e.py          # --keep para inspecionar os registros depois
```

Observação: a etapa 12 roda o worker localmente e consome a fila inteira do projeto (até `--worker-limit` registros), inclusive pesagens reais pendentes. Use um projeto de desenvolvimento ou rode com o worker de produção parado.

## Checklist de produção

- [ ] `SUPABASE_SERVICE_ROLE_KEY` só no Fly/VPS/Secrets do Actions; `grep -r service_role` no front não encontra nada.
- [ ] Projeto Supabase com migrations aplicadas e `ingestao-pesagem` publicada.
- [ ] `scripts/smoke_e2e.py` com todas as etapas PASS.
- [ ] Logs do worker sem `falha ao buscar pendentes` (indica URL/chave errada).
- [ ] Monitorar registros com `metadados->>'status' = 'processando'` há mais de 10 min: worker caiu no meio.
