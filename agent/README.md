# GPT Interface Stub

This folder contains a minimal command-line agent that:
1) Accepts a natural language task (ZH/EN).
2) Produces a simple plan.
3) Maps the plan to concrete project commands (e.g., run a Snakemake target).

You can later connect `agent/main.py` to your GPT endpoint (OpenAI, Azure OpenAI, etc.).
Environment variables can be placed in `.env` (never commit secrets).

Examples:
```bash
# Describe a task to compute ADTF for sub01 and show MCC
python agent/main.py "对 sub01 运行功能连接性流程并给我 MCC 结果"
```
