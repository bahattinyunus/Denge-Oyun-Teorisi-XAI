"""
Model Makro Strateji ve Önem Haritası (Beeswarm Global)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Modelin tüm veri kümesindeki evrensel kararlarında özniteliklerin
makro baskınlıklarını, yönlü etkilerini (+ / -) ve genel stratejisini
özetleyen global açıklanabilirlik motoru.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class BeeswarmExplainer:
    """
    Global SHAP / Oyun Teorisi makro strateji analizcisi.
    """

    def __init__(
        self,
        values: np.ndarray,
        data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.values = np.asarray(values, dtype=np.float64)
        if self.values.ndim == 1:
            self.values = self.values.reshape(1, -1)

        self.n_samples, self.n_features = self.values.shape
        self.data = np.asarray(data) if data is not None else None

        if feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = [f"Feature_{i}" for i in range(self.n_features)]

    @classmethod
    def from_explanation(cls, exp: Any) -> "BeeswarmExplainer":
        """
        shap.Explanation veya ozel nesnelerden BeeswarmExplainer olusturur.
        """
        if hasattr(exp, "values"):
            vals = exp.values
            d = getattr(exp, "data", None)
            f_names = getattr(exp, "feature_names", None)
            return cls(values=vals, data=d, feature_names=f_names)
        elif isinstance(exp, dict):
            return cls(
                values=exp["values"],
                data=exp.get("data", None),
                feature_names=exp.get("feature_names", None),
            )
        elif isinstance(exp, np.ndarray):
            return cls(values=exp)
        else:
            raise TypeError(f"Unsupported explanation object of type {type(exp)}")

    def compute_global_importance(self) -> pd.DataFrame:
        """
        Her oznitelik icin ortalama mutlak SHAP degeri: (1/N) * sum |phi_i|
        ve yonlu etki korelasyonunu hesaplar.
        """
        mean_abs = np.mean(np.abs(self.values), axis=0)
        mean_signed = np.mean(self.values, axis=0)
        std_val = np.std(self.values, axis=0)

        df = pd.DataFrame({
            "Feature": self.feature_names,
            "Mean_Abs_SHAP": mean_abs,
            "Mean_Signed_SHAP": mean_signed,
            "Std_SHAP": std_val,
        })
        df = df.sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
        return df

    def text_summary(self, max_display: int = 15) -> str:
        """
        Konsol icin makro onem tablosunu uretir.
        """
        df = self.compute_global_importance().head(max_display)
        lines = [
            "=" * 72,
            " ♟️  DENGE-XAI: MAKRO STRATEJI VE GLOBAL ONEM HARITASI (BEESWARM)",
            "=" * 72,
            f"{'Sira':<5} | {'Oznitelik':<25} | {'Ort. |SHAP|':<14} | {'Net Yon':<12}",
            "-" * 72,
        ]

        for i, row in df.iterrows():
            net_dir = "🔺 Pozitif" if row["Mean_Signed_SHAP"] >= 0 else "🔻 Negatif"
            lines.append(
                f"{i+1:<5} | {row['Feature']:<25} | {row['Mean_Abs_SHAP']:<14.4f} | {net_dir}"
            )

        lines.append("=" * 72)
        return "\n".join(lines)

    def plot(
        self,
        max_display: int = 15,
        save_path: Optional[str] = None,
        show: bool = False,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Global beeswarm / summary grafigini cizer.
        """
        df = self.compute_global_importance().head(max_display)
        # Alttan yukari gosterim
        top_indices = [self.feature_names.index(f) for f in df["Feature"][::-1]]
        top_names = [self.feature_names[i] for i in top_indices]

        fig, ax = plt.subplots(figsize=(10, max(6, len(top_names) * 0.45 + 2)))

        y_positions = np.arange(len(top_names))

        for y_idx, f_idx in enumerate(top_indices):
            f_vals = self.values[:, f_idx]
            
            # Gorsel jitter ekleme (beeswarm etkisi)
            jitter = np.random.normal(0, 0.08, size=len(f_vals))
            y_scatter = y_idx + jitter

            if self.data is not None and f_idx < self.data.shape[1]:
                raw = self.data[:, f_idx]
                norm_raw = (raw - np.min(raw)) / (np.ptp(raw) + 1e-9)
                scatter = ax.scatter(
                    f_vals,
                    y_scatter,
                    c=norm_raw,
                    cmap="coolwarm",
                    alpha=0.6,
                    s=18,
                    edgecolors="none",
                )
            else:
                ax.scatter(
                    f_vals,
                    y_scatter,
                    color="#3498db",
                    alpha=0.6,
                    s=18,
                    edgecolors="none",
                )

        ax.axvline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(top_names, fontsize=10)
        ax.set_xlabel("SHAP Değeri (Model Çıktısı Üzerindeki Etki)", fontsize=11)
        ax.set_title(
            title or "Denge-XAI: Makro Strateji Özeti (Global Beeswarm Dağılımı)",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(axis="x", linestyle=":", alpha=0.5)

        if self.data is not None:
            cbar = plt.colorbar(scatter, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
            cbar.set_label("Öznitelik Değeri (Düşükten Yükseğe)", fontsize=9)
            cbar.set_ticks([0.05, 0.95])
            cbar.set_ticklabels(["Düşük", "Yüksek"])

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()

        return fig


def render_summary(
    shap_values: Any,
    max_display: int = 15,
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
) -> BeeswarmExplainer:
    """
    README API ile tam uyumlu fonksiyonel summary cizici:
    render_summary(shap_values)
    """
    explainer = BeeswarmExplainer.from_explanation(shap_values)
    print(explainer.text_summary(max_display=max_display))
    explainer.plot(max_display=max_display, save_path=save_path, show=show, title=title)
    return explainer
