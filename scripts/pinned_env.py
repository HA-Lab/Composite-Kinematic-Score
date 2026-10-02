"""
Sets up and runs pipeline steps inside a pinned Python 3.12 /
Does not guarantee closer agreement across environments (for ex. with different OS),
but can be useful for standardizing exactly which package versions a run used.

Two ways this file is used:
  1. Imported from the notebook's own kernel, to create/update the pinned
     env: `from pinned_env import ensure_pinned_env`.
  2. Run directly BY the pinned env's own python interpreter, one pipeline
     step per invocation: `python pinned_env.py <step> <project_root>`.
"""
import subprocess
from pathlib import Path


def ensure_pinned_env(requirements_file: Path) -> str:
    conda_base = Path("/usr/local/miniconda")
    env_dir = conda_base / "envs" / "py312"

    if not conda_base.exists():
        print("Downloading Miniconda (one-time, ~100MB)...")
        subprocess.run(
            ["wget", "-q", "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh", "-O", "/tmp/miniconda.sh"],
            check=True,
        )
        print("Installing Miniconda...")
        subprocess.run(["bash", "/tmp/miniconda.sh", "-b", "-p", str(conda_base)], check=True)
        subprocess.run([str(conda_base / "bin/conda"), "tos", "accept", "--override-channels", "--channel", "https://repo.anaconda.com/pkgs/main"])
        subprocess.run([str(conda_base / "bin/conda"), "tos", "accept", "--override-channels", "--channel", "https://repo.anaconda.com/pkgs/r"])
        print("Miniconda installed.")

    if not env_dir.exists():
        print("Creating py312 conda environment...")
        subprocess.run([str(conda_base / "bin/conda"), "create", "-y", "-n", "py312", "python=3.12"], check=True)
        print("Environment created.")
    else:
        print("py312 environment already exists, reusing it.")

    print(f"Installing packages from {requirements_file.name}...")
    subprocess.run([str(env_dir / "bin/pip"), "install", "-q", "-r", str(requirements_file)], check=True)
    print("Pinned environment ready.")

    return str(env_dir / "bin/python")


if __name__ == "__main__":
    import sys
    import pandas as pd
    import yaml

    step = sys.argv[1]  # "extraction" | "transform" | "recovery"
    PROJECT_ROOT = Path(sys.argv[2])
    sys.path.append(str(PROJECT_ROOT / "scripts"))
    import pipeline_functions as pf

    with open(PROJECT_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    config["paths"]["project_root"] = str(PROJECT_ROOT)
    rp = pf._resolve_project_paths(config)

    key_df = pd.read_excel(PROJECT_ROOT / config["paths"]["animal_key_file"])
    std_df, profile = pf.build_dataset_profile(key_df, config)

    if step == "extraction":
        pf.run_extraction(config, std_df, profile)
    elif step == "transform":
        extracted_df = pd.read_csv(rp["output_dir"] / "extracted_features.csv")
        pf.run_transform_select(config, profile, df=extracted_df, fixed_lambdas=pf.load_fixed_lambdas(config))
    elif step == "recovery":
        transformed_df = pd.read_csv(rp["output_dir"] / "transformed_selected_features.csv")
        pf.run_recovery_scores(config, profile, df=transformed_df)
    else:
        raise ValueError(f"Unknown step: {step!r}")