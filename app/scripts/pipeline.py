from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_step(description: str, module: str):
    print(f"\n=== {description} ===")
    start = time.time()

    result = subprocess.run(
        [PYTHON, "-m", f"app.scripts.{module}"],
        cwd=ROOT.parent.parent,  # go up from app/scripts to project root
        text=True
    )

    if result.returncode != 0:
        print(f"Step failed: {description}")
        sys.exit(1)

    duration = round(time.time() - start, 2)
    print(f"{description} completed in {duration}s")


def main():
    print("  Finance AI Advisor Pipeline")
    print("==============================\n")

    run_step("Fetching OHLCV into database", "fetch_to_db")
    run_step("Computing technical features", "load_features_to_db")
    run_step("Training ML models", "train_model")

    print("\nPipeline complete.\n")


if __name__ == "__main__":
    main()
