r"""
John Nash & Rekabetçi Olmayan Oyunlar (Non-Cooperative Equilibrium)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Yapay zeka modellerinde girdiler (features) arasi rekabetin ve stratejik dengenin modellenmesi.
Feature suppression (oznitelik baskilama) ve feature synergy tespiti,
Nash dengesi hesaplamalari ve denge bazli oznitelik siralamasi.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linprog


class FeaturePayoffMatrix:
    """
    Oznitelikler arasi rekabet ve isbirligi odul matrisini insa eder.
    
    Her oznitelik bir oyuncudur (player).
    Iki oyuncu arasindaki getiri (payoff):
    - A[i, j]: j ozniteligi varken i ozniteliginin marjinal katkisi / getirisi.
    """

    def __init__(
        self,
        payoff_matrix: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ):
        self.matrix = np.asarray(payoff_matrix, dtype=np.float64)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Payoff matrix must be a square 2D numpy array.")

        self.n_features = self.matrix.shape[0]
        if feature_names is not None:
            if len(feature_names) != self.n_features:
                raise ValueError("Length of feature_names must match matrix dimensions.")
            self.feature_names = list(feature_names)
        else:
            self.feature_names = [f"Feature_{i}" for i in range(self.n_features)]

    @classmethod
    def from_correlations(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "FeaturePayoffMatrix":
        """
        Korelasyon ve kismi korelasyon / mutual information tabanli odul matrisi uretir.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_features = X.shape[1]

        r_iy = np.zeros(n_features)
        std_y = np.std(y) + 1e-12
        for i in range(n_features):
            std_xi = np.std(X[:, i]) + 1e-12
            r_iy[i] = np.abs(np.corrcoef(X[:, i], y)[0, 1])
            if np.isnan(r_iy[i]):
                r_iy[i] = 0.0

        R = np.corrcoef(X, rowvar=False)
        R = np.nan_to_num(R, nan=0.0)

        matrix = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    matrix[i, j] = r_iy[i]
                else:
                    matrix[i, j] = r_iy[i] * (1.0 - 0.5 * np.abs(R[i, j]))

        return cls(matrix, feature_names)

    @classmethod
    def from_model_loss(
        cls,
        model,
        X: np.ndarray,
        y: np.ndarray,
        loss_fn,
        feature_names: Optional[List[str]] = None,
    ) -> "FeaturePayoffMatrix":
        """
        Model kaybi (loss reduction) uzerinden marjinal getiri matrisi olusturur.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]
        baseline_pred = model.predict(X)
        base_loss = loss_fn(y, baseline_pred)

        matrix = np.zeros((n_features, n_features))
        means = np.mean(X, axis=0)

        for j in range(n_features):
            X_mod_j = X.copy()
            X_mod_j[:, j] = means[j]
            pred_no_j = model.predict(X_mod_j)
            loss_no_j = loss_fn(y, pred_no_j)

            for i in range(n_features):
                if i == j:
                    matrix[i, j] = max(0.0, loss_no_j - base_loss)
                else:
                    X_mod_ij = X_mod_j.copy()
                    X_mod_ij[:, i] = means[i]
                    pred_no_ij = model.predict(X_mod_ij)
                    loss_no_ij = loss_fn(y, pred_no_ij)
                    matrix[i, j] = max(0.0, loss_no_ij - loss_no_j)

        return cls(matrix, feature_names)


def compute_nash_equilibrium(
    payoff_matrix: Union[np.ndarray, FeaturePayoffMatrix]
) -> Dict[str, Union[np.ndarray, List[Tuple[int, int]], float]]:
    """
    Symmetric ve 2-oyuncu / n-oznitelik stratejik oyununda Nash dengesini hesaplar.
    
    1. Saf Strateji Nash Dengeleri (Pure Strategy Nash Equilibria)
    2. Karma Strateji Nash Dengesi (Linear Programming Minimax / Support Enumeration)
    """
    if isinstance(payoff_matrix, FeaturePayoffMatrix):
        A = payoff_matrix.matrix
    else:
        A = np.asarray(payoff_matrix, dtype=np.float64)

    n = A.shape[0]

    # 1. Saf Stratejiler Kontrolu:
    pure_equilibria: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            is_best_row = bool(A[i, j] >= np.max(A[:, j]) - 1e-9)
            is_best_col = bool(A[j, i] >= np.max(A[:, i]) - 1e-9)
            if is_best_row and is_best_col:
                pure_equilibria.append((i, j))

    # 2. Saf Simetrik Denge Varsa
    symmetric_pure = [i for (i, j) in pure_equilibria if i == j]
    if len(symmetric_pure) == 1:
        i_star = symmetric_pure[0]
        mixed_eq = np.zeros(n)
        mixed_eq[i_star] = 1.0
        expected_payoff = float(A[i_star, i_star])
        return {
            "pure_equilibria": pure_equilibria,
            "mixed_equilibrium": mixed_eq,
            "expected_payoff": expected_payoff,
        }

    # 3. Dogrusal Programlama ile Minimax / Nash Stratejisi
    c = np.zeros(n + 1)
    c[-1] = -1.0

    A_ub = np.hstack([-A, np.ones((n, 1))])
    b_ub = np.zeros(n)

    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, 1.0) for _ in range(n)] + [(None, None)]

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res.success and np.all(res.x[:n] >= -1e-7):
            mixed_eq = np.maximum(0.0, res.x[:n])
            mixed_eq = mixed_eq / np.sum(mixed_eq)
        else:
            raise RuntimeError("Linprog did not converge")
    except Exception:
        row_means = np.mean(A, axis=1)
        exp_vals = np.exp(row_means - np.max(row_means))
        mixed_eq = exp_vals / np.sum(exp_vals)

    expected_payoff = float(mixed_eq.T @ A @ mixed_eq)

    return {
        "pure_equilibria": pure_equilibria,
        "mixed_equilibrium": mixed_eq,
        "expected_payoff": expected_payoff,
    }


def detect_feature_suppression(
    payoff_matrix: Union[np.ndarray, FeaturePayoffMatrix],
    threshold: float = 0.15,
) -> List[Dict[str, Union[str, int, float]]]:
    """
    Oznitelikler arasi baskilama (suppression / cannibalization) tespiti.
    Bir oznitelik digerinin marjinal getirisini belirgin olcude dusuruyorsa tespit edilir.
    """
    if isinstance(payoff_matrix, FeaturePayoffMatrix):
        A = payoff_matrix.matrix
        names = payoff_matrix.feature_names
    else:
        A = np.asarray(payoff_matrix, dtype=np.float64)
        names = [f"Feature_{i}" for i in range(A.shape[0])]

    n = A.shape[0]
    suppressions = []

    for i in range(n):
        isolated_payoff = A[i, i]
        for j in range(n):
            if i != j:
                coupled_payoff = A[i, j]
                drop = isolated_payoff - coupled_payoff
                relative_drop = drop / (isolated_payoff + 1e-12)
                if relative_drop > threshold:
                    suppressions.append({
                        "suppressed_feature": names[i],
                        "suppressor_feature": names[j],
                        "suppressed_index": i,
                        "suppressor_index": j,
                        "isolated_payoff": float(isolated_payoff),
                        "coupled_payoff": float(coupled_payoff),
                        "relative_drop": float(relative_drop),
                    })

    suppressions.sort(key=lambda x: x["relative_drop"], reverse=True)
    return suppressions


def compute_feature_synergy(
    payoff_matrix: Union[np.ndarray, FeaturePayoffMatrix],
    threshold: float = 0.05,
) -> List[Dict[str, Union[str, int, float]]]:
    """
    Super-additive sinerji tespiti.
    """
    if isinstance(payoff_matrix, FeaturePayoffMatrix):
        A = payoff_matrix.matrix
        names = payoff_matrix.feature_names
    else:
        A = np.asarray(payoff_matrix, dtype=np.float64)
        names = [f"Feature_{i}" for i in range(A.shape[0])]

    n = A.shape[0]
    synergies = []

    for i in range(n):
        for j in range(i + 1, n):
            joint_gain = (A[i, j] + A[j, i]) / 2.0
            indiv_gain = (A[i, i] + A[j, j]) / 2.0
            synergy_delta = joint_gain - indiv_gain
            if synergy_delta > threshold:
                synergies.append({
                    "feature_1": names[i],
                    "feature_2": names[j],
                    "index_1": i,
                    "index_2": j,
                    "synergy_gain": float(synergy_delta),
                    "joint_payoff": float(joint_gain),
                    "individual_payoff": float(indiv_gain),
                })

    synergies.sort(key=lambda x: x["synergy_gain"], reverse=True)
    return synergies


class NashFeatureRanker:
    """
    Nash Denge agirliklari uzerinden ozniteliklerin onem ve kararlilik siralamasini yapar.
    """

    def __init__(self, payoff_matrix: FeaturePayoffMatrix):
        self.payoff_matrix = payoff_matrix
        self.equilibrium_result = compute_nash_equilibrium(payoff_matrix)

    def rank(self) -> List[Dict[str, Union[str, int, float]]]:
        """
        Oznitelikleri Nash dengesindeki olasilik agirligina ve izole getiriye gore siralar.
        """
        weights = self.equilibrium_result["mixed_equilibrium"]
        names = self.payoff_matrix.feature_names
        
        ranked = []
        for idx, (name, weight) in enumerate(zip(names, weights)):
            ranked.append({
                "index": idx,
                "feature": name,
                "nash_weight": float(weight),
                "isolated_payoff": float(self.payoff_matrix.matrix[idx, idx]),
            })

        ranked.sort(key=lambda x: (x["nash_weight"], x["isolated_payoff"]), reverse=True)
        return ranked
