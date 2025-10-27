import subprocess, shlex

def run_snakemake_target(snakefile: str, cores: int = 4, profile: str | None = None) -> int:
    cmd = f"snakemake -s {snakefile} --cores {cores}"
    if profile:
        cmd += f" --profile {profile}"
    print(f"[snakemake] {cmd}")
    return subprocess.run(shlex.split(cmd), check=False).returncode
