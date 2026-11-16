"""
Mikro-Simülasyon ve Lokal Karar Açıklayıcısı (Waterfall Local)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tekil tahminlerin ortalama beklentiden (Baseline / E[f(x)]) başlayarak
adım adım hangi özniteliklerin koalisyondaki marjinal katkılarıyla
nihai karara (f(x)) dönüştüğünü görselleştirir ve analiz eder.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for all environments
import matplotlib.pyplot as plt


class WaterfallExplainer:
    """
    Lokal karar simulasyon motoru.
    """

    def __init__(
        self,
        base_value: float,
        values: np.ndarray,
        data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.base_value = float(base_value)
        self.values = np.asarray(values, dtype=np.float64)
        self.data = np.asarray(data) if data is not None else None
        self.n_features = len(self.values)

        if feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = [f"Feature_{i}" for i in range(self.n_features)]

    @classmethod
    def from_explanation(cls, exp: Any) -> "WaterfallExplainer":
        """
        shap.Explanation veya ozel nesnelerden WaterfallExplainer olusturur.
        """
        # shap.Explanation instance
        if hasattr(exp, "values") and hasattr(exp, "base_values"):
            vals = exp.values
            if hasattr(vals, "ndim") and vals.ndim > 1:
                vals = vals[0]
            
            b_val = exp.base_values
            if hasattr(b_val, "__len__") and not isinstance(b_val, (str, bytes)):
                try:
                    b_val = float(b_val[0])
                except Exception:
                    b_val = float(b_val)
            else:
                b_val = float(b_val)

            d = getattr(exp, "data", None)
            f_names = getattr(exp, "feature_names", None)
            return cls(base_value=b_val, values=vals, data=d, feature_names=f_names)

        elif isinstance(exp, dict):
            return cls(
                base_value=exp.get("base_value", 0.0),
                values=exp["values"],
                data=exp.get("data", None),
                feature_names=exp.get("feature_names", None),
            )
        elif isinstance(exp, np.ndarray):
            return cls(base_value=0.0, values=exp)
        else:
            raise TypeError(f"Unsupported explanation object of type {type(exp)}")

    def text_summary(self, max_display: int = 10) -> str:
        """
        Lokal kararin adim adim metinsel dokumunu olusturur.
        """
        indices = np.argsort(np.abs(self.values))[::-1]
        lines = [
            "=" * 68,
            " ♟️  DENGE-XAI: LOKAL KARAR ADIMLARI (WATERFALL SIMULATION)",
            "=" * 68,
            f" 🎯 Taban Deger (E[f(x)] / Baseline): {self.base_value:+.4f}",
            "-" * 68,
            f"{'Oznitelik':<25} | {'Deger':<12} | {'Katki (Phi)':<14} | {'Yon'}",
            "-" * 68,
        ]

        curr = self.base_value
        displayed = 0
        for idx in indices:
            if displayed >= max_display:
                break
            name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feat_{idx}"
            val_str = f"{self.data[idx]:.2f}" if self.data is not None and idx < len(self.data) else "-"
            phi = self.values[idx]
            direction = "🔺 Pozitif (+)" if phi >= 0 else "🔻 Negatif (-)"
            lines.append(f"{name:<25} | {val_str:<12} | {phi:+.4f}         | {direction}")
            curr += phi
            displayed += 1

        lines.append("-" * 68)
        lines.append(f" 🏁 Nihai Model Tahmini f(x)       : {self.base_value + np.sum(self.values):+.4f}")
        lines.append("=" * 68)
        return "\n".join(lines)

    def plot(
        self,
        max_display: int = 10,
        save_path: Optional[str] = None,
        show: bool = False,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Waterfall grafigini profesyonel bicimde cizer.
        """
        # Onem sirasina gore sirala
        indices = np.argsort(np.abs(self.values))[::-1][:max_display]
        # Alttan yukariya cizmek icin ters cevir
        indices = indices[::-1]

        names = [
            (
                f"{self.feature_names[i]} = {self.data[i]:.2f}"
                if self.data is not None and i < len(self.data)
                else self.feature_names[i]
            )
            for i in indices
        ]
        vals = self.values[indices]

        fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.5 + 2)))

        # Kümülatif baslangic ve bitisler
        running = self.base_value
        bottoms = []
        for v in vals:
            bottoms.append(running if v >= 0 else running + v)
            running += v

        colors = ["#e74c3c" if v >= 0 else "#3498db" for v in vals]

        y_positions = np.arange(len(names))
        bars = ax.barh(
            y_positions,
            np.abs(vals),
            left=bottoms,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            height=0.6,
        )

        # Bar deger etiketleri
        for y, val, bottom in zip(y_positions, vals, bottoms):
            x_pos = bottom + np.abs(val) / 2.0
            sign = "+" if val >= 0 else ""
            ax.text(
                x_pos,
                y,
                f"{sign}{val:.2f}",
                va="center",
                ha="center",
                color="white",
                fontweight="bold",
                fontsize=9,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(names, fontsize=10)
        ax.axvline(self.base_value, color="gray", linestyle="--", alpha=0.8, label=f"E[f(X)] = {self.base_value:.2f}")

        final_pred = self.base_value + np.sum(self.values)
        ax.axvline(final_pred, color="#2ecc71", linestyle="-", linewidth=1.5, label=f"f(x) = {final_pred:.2f}")

        ax.set_xlabel("Model Tahmini Ekseninde Katkı", fontsize=11)
        ax.set_title(
            title or f"Denge-XAI: Lokal Waterfall Analizi (f(x) = {final_pred:.2f})",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(loc="lower right")
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()

        return fig


def render_waterfall(
    shap_value_object: Any,
    max_display: int = 10,
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
) -> WaterfallExplainer:
    """
    README API ile tam uyumlu fonksiyonel waterfall cizici:
    render_waterfall(shap_values[0])
    """
    explainer = WaterfallExplainer.from_explanation(shap_value_object)
    print(explainer.text_summary(max_display=max_display))
    explainer.plot(max_display=max_display, save_path=save_path, show=show, title=title)
    return explainer
