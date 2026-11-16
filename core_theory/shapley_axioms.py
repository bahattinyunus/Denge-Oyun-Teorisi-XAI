"""
Lloyd Shapley & Kooperatif Oyun Kurami (Cooperative Games)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Shapley'nin 4 temel aksiyomu (Verimlilik, Simetri, Etkisiz Eleman / Dummy, Toplamsallik)
ve kesin (exact) / orneklemeli (sampling) Shapley degeri hesaplama motoru.
"""

from itertools import combinations
import math
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np


class ShapleyGame:
    """
    Kooperatif bir oyunda karakteristik fonksiyon v(S) tanimlar.
    
    N: Toplam oyuncu kumesi {0, 1, ..., n-1}
    v(S): S koalisyonunun toplam getirisi
    """

    def __init__(
        self,
        n_players: int,
        characteristic_fn: Optional[Callable[[Tuple[int, ...]], float]] = None,
        player_names: Optional[List[str]] = None,
    ):
        self.n_players = n_players
        self.players = tuple(range(n_players))
        self.player_names = (
            list(player_names)
            if player_names is not None
            else [f"Player_{i}" for i in range(n_players)]
        )
        self._cache: Dict[Tuple[int, ...], float] = {}

        if characteristic_fn is not None:
            self._char_fn = characteristic_fn
        else:
            self._char_fn = lambda S: 0.0

    def v(self, coalition: Union[Tuple[int, ...], List[int], Set[int]]) -> float:
        """
        Koalisyon getirisini dondurur. Onbellekleme (memoization) kullanir.
        """
        key = tuple(sorted(set(coalition)))
        if key not in self._cache:
            self._cache[key] = float(self._char_fn(key))
        return self._cache[key]

    def set_value(self, coalition: Union[Tuple[int, ...], List[int]], value: float):
        """
        Belirli bir koalisyonun getirisini dogrudan tanimlar.
        """
        key = tuple(sorted(set(coalition)))
        self._cache[key] = float(value)


def compute_exact_shapley(game: ShapleyGame) -> np.ndarray:
    """
    Lloyd Shapley'nin kesin (exact) formulunu hesaplar:
    phi_i(v) = sum_{S subseteq N \setminus {i}} [ |S|! * (|N| - |S| - 1)! / |N|! ] * [v(S cup {i}) - v(S)]
    """
    n = game.n_players
    n_fact = math.factorial(n)
    shapley_values = np.zeros(n, dtype=np.float64)

    all_players = set(game.players)

    for i in range(n):
        marginal_sum = 0.0
        remaining_players = list(all_players - {i})

        # S koalisyonu tum alt kumeler uzerinde gezinir: boyut k = 0 to n-1
        for k in range(n):
            weight = (math.factorial(k) * math.factorial(n - k - 1)) / n_fact

            for S in combinations(remaining_players, k):
                val_without_i = game.v(S)
                val_with_i = game.v(tuple(sorted(S + (i,))))
                marginal_contribution = val_with_i - val_without_i
                marginal_sum += weight * marginal_contribution

        shapley_values[i] = marginal_sum

    return shapley_values


def verify_efficiency_axiom(
    game: ShapleyGame,
    shapley_values: np.ndarray,
    tol: float = 1e-5,
) -> Dict[str, Union[bool, float, str]]:
    """
    Aksiyom 1: Verimlilik (Efficiency)
    sum_{i in N} phi_i = v(N) - v(empty)
    """
    sum_phi = float(np.sum(shapley_values))
    grand_coalition_val = game.v(game.players) - game.v(())
    discrepancy = abs(sum_phi - grand_coalition_val)
    passed = discrepancy <= tol

    return {
        "passed": passed,
        "sum_shapley": sum_phi,
        "grand_coalition_value": grand_coalition_val,
        "discrepancy": discrepancy,
        "status": "✅ GECTI" if passed else "❌ BASARISIZ",
    }


def verify_symmetry_axiom(
    game: ShapleyGame,
    shapley_values: np.ndarray,
    tol: float = 1e-5,
) -> Dict[str, Union[bool, List[Tuple[int, int]], str]]:
    """
    Aksiyom 2: Simetri (Symmetry)
    Eger her S subseteq N \setminus {i, j} icin v(S cup {i}) == v(S cup {j}) ise,
    phi_i == phi_j olmalidir.
    """
    n = game.n_players
    all_players = set(game.players)
    symmetric_pairs = []
    violations = []

    for i in range(n):
        for j in range(i + 1, n):
            is_symmetric = True
            rem = list(all_players - {i, j})
            for k in range(len(rem) + 1):
                for S in combinations(rem, k):
                    val_i = game.v(tuple(sorted(S + (i,))))
                    val_j = game.v(tuple(sorted(S + (j,))))
                    if abs(val_i - val_j) > tol:
                        is_symmetric = False
                        break
                if not is_symmetric:
                    break

            if is_symmetric:
                symmetric_pairs.append((i, j))
                if abs(shapley_values[i] - shapley_values[j]) > tol:
                    violations.append((i, j))

    passed = len(violations) == 0
    return {
        "passed": passed,
        "symmetric_pairs_found": symmetric_pairs,
        "violations": violations,
        "status": "✅ GECTI" if passed else "❌ BASARISIZ",
    }


def verify_dummy_axiom(
    game: ShapleyGame,
    shapley_values: np.ndarray,
    tol: float = 1e-5,
) -> Dict[str, Union[bool, List[int], str]]:
    """
    Aksiyom 3: Etkisiz Eleman (Dummy / Null Player)
    Eger her S subseteq N \setminus {i} icin v(S cup {i}) == v(S) + v({i}) ise,
    phi_i == v({i}) olmalidir.
    """
    n = game.n_players
    all_players = set(game.players)
    dummy_players = []
    violations = []

    for i in range(n):
        is_dummy = True
        v_i = game.v((i,)) - game.v(())
        rem = list(all_players - {i})

        for k in range(len(rem) + 1):
            for S in combinations(rem, k):
                val_with = game.v(tuple(sorted(S + (i,))))
                val_without = game.v(S)
                if abs((val_with - val_without) - v_i) > tol:
                    is_dummy = False
                    break
            if not is_dummy:
                break

        if is_dummy:
            dummy_players.append(i)
            if abs(shapley_values[i] - v_i) > tol:
                violations.append(i)

    passed = len(violations) == 0
    return {
        "passed": passed,
        "dummy_players_found": dummy_players,
        "violations": violations,
        "status": "✅ GECTI" if passed else "❌ BASARISIZ",
    }


def verify_additivity_axiom(
    game_u: ShapleyGame,
    game_w: ShapleyGame,
    tol: float = 1e-5,
) -> Dict[str, Union[bool, float, str]]:
    """
    Aksiyom 4: Toplamsallik (Additivity)
    phi_i(u + w) = phi_i(u) + phi_i(w)
    """
    if game_u.n_players != game_w.n_players:
        raise ValueError("Games must have same number of players for additivity.")

    # Bilesik oyun u + w
    def combined_char_fn(S):
        return game_u.v(S) + game_w.v(S)

    game_combined = ShapleyGame(game_u.n_players, characteristic_fn=combined_char_fn)

    phi_u = compute_exact_shapley(game_u)
    phi_w = compute_exact_shapley(game_w)
    phi_combined = compute_exact_shapley(game_combined)

    diff = np.max(np.abs(phi_combined - (phi_u + phi_w)))
    passed = bool(diff <= tol)

    return {
        "passed": passed,
        "max_discrepancy": float(diff),
        "status": "✅ GECTI" if passed else "❌ BASARISIZ",
    }


def verify_all_shapley_axioms(game: ShapleyGame) -> Dict[str, Dict]:
    """
    Shapley'nin tum 4 aksiyomunu denetler ve kapsamli dogrulama raporu olusturur.
    """
    shapley_vals = compute_exact_shapley(game)
    eff = verify_efficiency_axiom(game, shapley_vals)
    sym = verify_symmetry_axiom(game, shapley_vals)
    dum = verify_dummy_axiom(game, shapley_vals)

    # Toplamsallik testi icin simetrik dummy bir oyun turetilir
    dummy_game = ShapleyGame(game.n_players, characteristic_fn=lambda S: float(len(S) * 2.0))
    add = verify_additivity_axiom(game, dummy_game)

    all_passed = eff["passed"] and sym["passed"] and dum["passed"] and add["passed"]

    return {
        "shapley_values": shapley_vals,
        "all_passed": all_passed,
        "efficiency": eff,
        "symmetry": sym,
        "dummy_player": dum,
        "additivity": add,
        "summary": (
            f"Shapley Aksiyomlari Dogrulandi: "
            f"[Verimlilik: {eff['status']}, Simetri: {sym['status']}, "
            f"Etkisiz Eleman: {dum['status']}, Toplamsallik: {add['status']}]"
        ),
    }
