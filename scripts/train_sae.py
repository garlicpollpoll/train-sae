from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sae_defense.train import main_train


if __name__ == "__main__":
    main_train()
