# INF_WP3

Modular, reproducible research workspace with three cooperating Docker environments:

- **FUNC_Conn** — functional connectivity workflows (current focus)
- **STRUCT_Conn** — structural connectivity workflows (placeholder)
- **BRAIN_Model** — brain modeling / simulation workflows (placeholder)

All services share a common data directory and can be orchestrated together. A GPT interface stub is included to translate natural language tasks into project commands.

---

## Quickstart (on Ubuntu with Docker already installed)

```bash
# Bring up all services (Jupyter in FUNC_Conn)
docker compose -f config/docker-compose.yml up --build

# Open Jupyter in browser at http://localhost:8888 (token shown in logs)

# Run Snakemake inside FUNC_Conn
docker compose -f config/docker-compose.yml exec func_conn snakemake --cores 4
```

> You already have VS Code and Docker on Ubuntu, so no re-install steps are included.

---

## Layout

```
INF_WP3/
  config/
    docker-compose.yml         # 3 services + shared volumes
    config.yaml                # workflow parameters (shared)
    profiles/hpc/slurm/        # Snakemake HPC profile (SLURM example)
  docker/
    FUNC_Conn/                 # Active Docker env (you'll rewrite)
      Dockerfile
      requirements.txt
      entrypoint.sh
    STRUCT_Conn/               # Placeholder
      Dockerfile
    BRAIN_Model/               # Placeholder
      Dockerfile
  src/                         # Project code (shared)
    preprocessing/
    connectivity/
    modeling/
    utils/
  workflows/                   # Snakemake files for each domain
    FUNC_Conn/Snakefile
    STRUCT_Conn/Snakefile
    BRAIN_Model/Snakefile
  shared/                      # Shared data across services
    data/
      raw/
      interim/
      processed/
    results/
      figures/
      metrics/
      logs/
      reports/
  notebooks/                   # Jupyter notebooks (FUNC_Conn uses this)
  agent/                       # GPT interface stub (CLI & planning)
    main.py
    tools/
      exec.py
      snakemake.py
  .vscode/
    settings.json
    extensions.json
  .env.example                 # environment variables template
  .gitignore
```

---

## GPT Interface (stub)

- `agent/main.py "中文或英文问题…" `
- Produces a **plan** and executes mapped commands (e.g., Snakemake targets, Python scripts).
- You can later connect this to an external GPT endpoint; see `agent/README.md`.

---

## HPC

- Example **SLURM** profile at `config/profiles/hpc/slurm/`.
- Run with: `snakemake --profile config/profiles/hpc/slurm` inside the container.
