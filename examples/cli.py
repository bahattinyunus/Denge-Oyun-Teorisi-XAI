"""
Denge-Oyun-Teorisi-XAI Komut Satırı Arayüzü (CLI)
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_ops.synthetic_coalitions import create_axiomatic_game
from core_theory.shapley_axioms import verify_all_shapley_axioms
from examples.production_snippet import main as run_production
from examples.full_pipeline_demo import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="♟️ Denge-Oyun-Teorisi-XAI: Açıklanabilir Yapay Zekâ Oyun Teorisi Motoru",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["production", "full", "axioms"],
        default="full",
        help="Çalıştırılacak mod: production (README snippet), full (Uçtan uca boru hattı), axioms (Shapley aksiyom testi)",
    )

    args = parser.parse_args()

    if args.mode == "production":
        run_production()
    elif args.mode == "full":
        run_pipeline()
    elif args.mode == "axioms":
        print("\n[Aksiyomatik Denetim] 4 Temel Shapley Aksiyomu Hesaplanıyor...")
        game = create_axiomatic_game(n_players=4)
        rep = verify_all_shapley_axioms(game)
        print(rep["summary"])
        print(f"Toplam Shapley Değerleri: {rep['shapley_values']}")


if __name__ == "__main__":
    main()
