import subprocess, shlex

def run_cmd(cmd: str) -> int:
    print(f"[exec] {cmd}")
    proc = subprocess.run(shlex.split(cmd), check=False)
    return proc.returncode
