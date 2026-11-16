"""
John Maynard Smith & Evrimsel Oyun Teorisi (Evolutionary Game Theory)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Evrimsel Olarak Kararli Strateji (ESS - Evolutionarily Stable Strategy) ve 
Replicator Dynamics uzerinden saglam (robust) oznitelik secim filtresi.
Modelin gurultulu verilere, dagilim kaymalarina (distribution drift) ve
adversarial mutasyonlara karsi direncini olcer.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class EvolutionaryGame:
    """
    Evrimsel oyun matrisi ve popülasyon uygunluk (fitness) hesaplayicisi.
    """

    def __init__(self, payoff_matrix: np.ndarray, feature_names: Optional[List[str]] = None):
        self.A = np.asarray(payoff_matrix, dtype=np.float64)
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Payoff matrix must be square.")
        self.n_strategies = self.A.shape[0]
        if feature_names is not None:
            if len(feature_names) != self.n_strategies:
                raise ValueError("feature_names length must match matrix dimension.")
            self.feature_names = list(feature_names)
        else:
            self.feature_names = [f"Strategy_{i}" for i in range(self.n_strategies)]

    def fitness(self, population_shares: np.ndarray) -> np.ndarray:
        """
        Popülasyondaki her bir stratejinin getirisini (fitness) dondurur: (A x)_i
        """
        x = np.asarray(population_shares, dtype=np.float64)
        return self.A @ x

    def average_fitness(self, population_shares: np.ndarray) -> float:
        """
        Popülasyonun ortalama getirisi: x^T A x
        """
        x = np.asarray(population_shares, dtype=np.float64)
        return float(x.T @ self.A @ x)


def verify_ess_condition(
    payoff_matrix: np.ndarray,
    candidate_strategy: np.ndarray,
    mutants: Optional[List[np.ndarray]] = None,
    tolerance: float = 1e-6,
) -> Dict[str, Union[bool, str, List[Dict[str, float]]]]:
    """
    Maynard Smith ESS Kriterlerinin Dogrulanmasi:
    Bir S stratejisinin ESS olmasi icin:
    1. u(S, S) >= u(T, S)   (Nash dengesi kosulu)
    2. Eger u(S, S) == u(T, S) ise, u(S, T) > u(T, T) (Mutanta karsi ustunluk)
    """
    A = np.asarray(payoff_matrix, dtype=np.float64)
    n = A.shape[0]
    S = np.asarray(candidate_strategy, dtype=np.float64)
    S = S / np.sum(S)

    if mutants is None:
        # Standart saf mutasyon vektorleri (temel bazlar)
        mutants = []
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            if not np.allclose(e_i, S, atol=tolerance):
                mutants.append(e_i)

    # u(P, Q) = P^T A Q
    u_S_S = float(S.T @ A @ S)
    is_ess = True
    reasons = []
    mutant_evals = []

    for idx, T in enumerate(mutants):
        T = np.asarray(T, dtype=np.float64)
        T = T / np.sum(T)
        u_T_S = float(T.T @ A @ S)
        u_S_T = float(S.T @ A @ T)
        u_T_T = float(T.T @ A @ T)

        # 1. Kriter testi:
        if u_T_S > u_S_S + tolerance:
            is_ess = False
            reasons.append(f"Mutant {idx} can invade: u(T,S)={u_T_S:.4f} > u(S,S)={u_S_S:.4f}")
        # 2. Kriter testi (esitlik durumunda):
        elif abs(u_S_S - u_T_S) <= tolerance:
            if u_S_T <= u_T_T + tolerance:
                is_ess = False
                reasons.append(f"Weak against mutant {idx}: u(S,T)={u_S_T:.4f} <= u(T,T)={u_T_T:.4f}")

        mutant_evals.append({
            "mutant_id": idx,
            "u_S_S": u_S_S,
            "u_T_S": u_T_S,
            "u_S_T": u_S_T,
            "u_T_T": u_T_T,
            "invadable": not is_ess,
        })

    return {
        "is_ess": is_ess,
        "explanation": "; ".join(reasons) if reasons else "Strictly Evolutionarily Stable Strategy (ESS)",
        "mutant_evaluations": mutant_evals,
    }


def simulate_replicator_dynamics(
    payoff_matrix: np.ndarray,
    initial_shares: Optional[np.ndarray] = None,
    generations: int = 200,
    dt: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replicator Dynamics diferansiyel denklemini simule eder:
    dx_i / dt = x_i * [ (A x)_i - x^T A x ]
    
    Donus:
        (trajectory, time_points)
        trajectory.shape == (generations, n_strategies)
    """
    A = np.asarray(payoff_matrix, dtype=np.float64)
    n = A.shape[0]

    if initial_shares is None:
        x = np.ones(n, dtype=np.float64) / n
    else:
        x = np.asarray(initial_shares, dtype=np.float64)
        x = x / np.sum(x)

    trajectory = np.zeros((generations, n), dtype=np.float64)
    trajectory[0] = x

    for t in range(1, generations):
        # Runge-Kutta 4. Derece (RK4) adimi
        def deriv(pop):
            fit = A @ pop
            avg_fit = pop @ fit
            return pop * (fit - avg_fit)

        k1 = deriv(x)
        k2 = deriv(x + 0.5 * dt * k1)
        k3 = deriv(x + 0.5 * dt * k2)
        k4 = deriv(x + dt * k3)

        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x = np.maximum(x, 1e-12)  # Yok olma siniri
        x = x / np.sum(x)  # Normalizasyon
        trajectory[t] = x

    time_points = np.arange(generations) * dt
    return trajectory, time_points


class MaynardESSFilter:
    """
    Evrimsel Oyun Teorisi tabanli Oznitelik Secim ve Filtreleme Motoru.
    
    Veri kumesindeki ozniteliklere cesitli gurultuler ve mutasyonlar (adversarial shift,
    Gaussian noise, zero-masking) enjekte ederek hangi ozniteliklerin evrimsel olarak
    kararli (ESS) kaldigini, hangilerinin yok oldugunu tespit eder.
    """

    def __init__(
        self,
        n_generations: int = 150,
        mutation_rate: float = 0.15,
        extinction_threshold: float = 0.02,
        dt: float = 0.05,
        random_state: int = 42,
    ):
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.extinction_threshold = extinction_threshold
        self.dt = dt
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def fit_filter(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        noise_scenarios: int = 10,
    ) -> Dict[str, Union[List[str], np.ndarray, Dict]]:
        """
        Oznitelikleri evrimsel saglamlik (robustness) testine tabi tutar.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape

        if feature_names is not None:
            names = list(feature_names)
        else:
            names = [f"Feature_{i}" for i in range(n_features)]

        # 1. Taban getiri matrisi (Temiz veride)
        base_corr = np.zeros(n_features)
        std_y = np.std(y) + 1e-12
        for i in range(n_features):
            std_xi = np.std(X[:, i]) + 1e-12
            base_corr[i] = np.abs(np.corrcoef(X[:, i], y)[0, 1])

        # 2. Mutasyon ve gurultu senaryolari altinda getiri matrisleri
        payoff_accum = np.zeros((n_features, n_features))

        for scenario in range(noise_scenarios):
            # Adversarial / gurultu mutasyonu
            noise = self.rng.normal(0, self.mutation_rate * np.std(X, axis=0), size=X.shape)
            # Rastgele oznitelik maskeleme (mutant invasion)
            mask = self.rng.binomial(1, 0.9, size=X.shape)
            X_mutated = (X + noise) * mask

            R_mut = np.nan_to_num(np.corrcoef(X_mutated, rowvar=False), nan=0.0)
            corr_mut = np.zeros(n_features)
            for i in range(n_features):
                c = np.abs(np.corrcoef(X_mutated[:, i], y)[0, 1])
                corr_mut[i] = 0.0 if np.isnan(c) else c

            # Senaryo matrisi
            scen_matrix = np.zeros((n_features, n_features))
            for i in range(n_features):
                for j in range(n_features):
                    if i == j:
                        scen_matrix[i, j] = corr_mut[i]
                    else:
                        scen_matrix[i, j] = corr_mut[i] * (1.0 - 0.5 * np.abs(R_mut[i, j]))

            payoff_accum += scen_matrix

        avg_payoff = payoff_accum / noise_scenarios

        # 3. Replicator Dynamics uzerinden simulasyon
        trajectory, times = simulate_replicator_dynamics(
            payoff_matrix=avg_payoff,
            generations=self.n_generations,
            dt=self.dt,
        )

        final_shares = trajectory[-1]
        
        # 4. ESS Kriteri ve Saglamlik Analizi
        selected_indices = []
        selected_features = []
        eliminated_features = []

        feature_scores = []
        for idx in range(n_features):
            share = final_shares[idx]
            is_survivor = share >= self.extinction_threshold
            
            score_dict = {
                "index": idx,
                "feature": names[idx],
                "final_share": float(share),
                "is_ess_survivor": bool(is_survivor),
                "clean_corr": float(base_corr[idx]),
            }
            feature_scores.append(score_dict)

            if is_survivor:
                selected_indices.append(idx)
                selected_features.append(names[idx])
            else:
                eliminated_features.append(names[idx])

        feature_scores.sort(key=lambda s: s["final_share"], reverse=True)

        return {
            "selected_features": selected_features,
            "selected_indices": selected_indices,
            "eliminated_features": eliminated_features,
            "feature_scores": feature_scores,
            "final_shares": final_shares,
            "trajectory": trajectory,
            "payoff_matrix": avg_payoff,
            "generations": self.n_generations,
        }
