import argparse, os, subprocess, sys, shlex

from tools.exec import run_cmd
from tools.snakemake import run_snakemake_target

def naive_planner(nl: str):
    nl_lower = nl.lower()
    plan = {"domain":"FUNC_Conn","targets":[],"notes":[]}
    if "adtf" in nl_lower or "功能连接" in nl_lower:
        plan["targets"].append("workflows/FUNC_Conn/Snakefile")
        if "sub01" in nl_lower:
            plan["notes"].append("Focus on subject sub01 (adjust config.yaml if needed).")
    elif "结构" in nl_lower or "struct" in nl_lower:
        plan["domain"] = "STRUCT_Conn"
        plan["targets"].append("workflows/STRUCT_Conn/Snakefile")
    elif "model" in nl_lower or "建模" in nl_lower:
        plan["domain"] = "BRAIN_Model"
        plan["targets"].append("workflows/BRAIN_Model/Snakefile")
    else:
        plan["notes"].append("Defaulting to FUNC_Conn domain.")
        plan["targets"].append("workflows/FUNC_Conn/Snakefile")
    return plan

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="Natural language task description (ZH/EN).")
    parser.add_argument("--dry-run", action="store_true", help="Only print the plan and commands.")
    args = parser.parse_args()

    plan = naive_planner(args.task)
    print("=== PLAN ===")
    print(plan)

    # Example execution: run Snakemake for FUNC_Conn
    if plan["domain"] == "FUNC_Conn":
        cmd = "snakemake -s workflows/FUNC_Conn/Snakefile --cores 4"
        print("\nCommand:", cmd)
        if not args.dry_run:
            run_cmd(cmd)
    else:
        print("\nNo concrete execution mapping for domain:", plan["domain"], "(placeholder)")

if __name__ == "__main__":
    main()
