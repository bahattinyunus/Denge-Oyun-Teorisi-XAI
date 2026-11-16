"""
Oyun Teorisi Simülasyonları için Sentetik Veri Jeneratörü (Synthetic Coalitions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Simetri, Etkisiz Eleman (Dummy), Sinerji (Synergy) ve Baskılama (Suppression)
dinamiklerini içeren, zemin gerçekliği (ground-truth) bilinen veri kümeleri üretir.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from core_theory.shapley_axioms import ShapleyGame


@dataclass
class CoalitionDataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    roles: Dict[str, str]
    theoretical_shapley: Optional[np.ndarray] = None


def generate_coalition_dataset(
    n_samples: int = 1000,
    noise_level: float = 0.05,
    random_state: int = 42,
) -> CoalitionDataset:
    """
    Oyun teorisi aksiyomlarını test etmek için 6 özellikli sentetik veri kümesi:
    - X0, X1: Simetrik Çift (Symmetric Pair) -> Eşit ağırlık ve katkı
    - X2, X3: Sinerji Çifti (Synergistic Pair) -> Yalnızca birlikteyken yüksek getiri
    - X4: Doğrusal Ana Sinyal (Linear Signal)
    - X5: Etkisiz Eleman (Dummy Feature) -> Saf gürültü, hedefle sıfır ilişki
    """
    rng = np.random.RandomState(random_state)

    # 1. Simetrik özellikler
    x0 = rng.normal(0, 1, n_samples)
    x1 = rng.normal(0, 1, n_samples)

    # 2. Sinerjik özellikler (Örn. XOR benzeri veya çarpımsal etkileşim)
    x2 = rng.uniform(-1, 1, n_samples)
    x3 = rng.uniform(-1, 1, n_samples)
    synergy_interaction = 3.0 * (x2 * x3)

    # 3. Doğrusal özellik
    x4 = rng.normal(0, 1, n_samples)

    # 4. Etkisiz / Kukla özellik (Dummy player)
    x5 = rng.normal(0, 1, n_samples)

    # Hedef Değişken y
    noise = rng.normal(0, noise_level, n_samples)
    y = (
        2.0 * x0
        + 2.0 * x1
        + synergy_interaction
        + 1.5 * x4
        + 0.0 * x5
        + noise
    )

    X = np.column_stack([x0, x1, x2, x3, x4, x5])
    feature_names = [
        "Symmetric_A",
        "Symmetric_B",
        "Synergy_X",
        "Synergy_Y",
        "Linear_Main",
        "Dummy_Noise",
    ]

    roles = {
        "Symmetric_A": "symmetric",
        "Symmetric_B": "symmetric",
        "Synergy_X": "synergistic",
        "Synergy_Y": "synergistic",
        "Linear_Main": "linear",
        "Dummy_Noise": "dummy",
    }

    return CoalitionDataset(
        X=X,
        y=y,
        feature_names=feature_names,
        roles=roles,
    )


def create_axiomatic_game(n_players: int = 4) -> ShapleyGame:
    """
    Aksiyomların doğrulanması için analitik karakteristik fonksiyon oyunu:
    - Oyuncu 0 ve 1: Simetrik
    - Oyuncu 2: Güçlü koalisyon ortağı
    - Oyuncu 3: Dummy (katkısı her koalisyonda sıfır)
    """
    def char_fn(S: Tuple[int, ...]) -> float:
        if not S:
            return 0.0
        val = 0.0
        # 0 veya 1 var ise eşit oranda katkı
        if 0 in S:
            val += 5.0
        if 1 in S:
            val += 5.0
        # 2 var ise ek katkı
        if 2 in S:
            val += 8.0
        # 0 ve 2 birlikteyse süper-toplamsal sinerji
        if 0 in S and 2 in S:
            val += 4.0
        if 1 in S and 2 in S:
            val += 4.0
        # 3 (dummy) hiçbir koalisyona marjinal değer katmaz
        return val

    names = ["Symmetric_P0", "Symmetric_P1", "Synergist_P2", "Dummy_P3"]
    return ShapleyGame(n_players=4, characteristic_fn=char_fn, player_names=names)
