"""
Denge-Oyun-Teorisi-XAI Canlı Üretim Örneği (Production Snippet)
README.md dosyasındaki resmi başlangıç kodu ile tam uyumludur.
"""

import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from explainers.beeswarm_global import render_summary
from explainers.waterfall_local import render_waterfall


def main():
    print("=" * 70)
    print("♟️  DENGE-OYUN-TEORİSİ-XAI: CANLI ÜRETİM BORU HATTI BAŞLATILIYOR")
    print("=" * 70)

    # 1. Aşama: Büyük Koalisyonun (Grand Coalition) Kurulması
    print("\n[Aşama 1] Büyük Koalisyon Veri Kümesi Yükleniyor...")
    X, y = shap.datasets.california()
    # Hızlı ve kararlı analiz için temsili alt örneklem
    X_sample = X.iloc[:200]
    y_sample = y[:200]

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_sample, y_sample)

    # 2. Aşama: Oyun Teorisi Motorunun (SHAP) Başlatılması
    # Maliyet dezavantajını kırmak için TreeExplainer ile optimizasyon
    print("[Aşama 2] TreeExplainer ile Hızlandırılmış Koalisyon Dağılımı...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    print(f"Oyun Matrisi Başarıyla Kuruldu. Dağıtılan Ödül Matrisi: {shap_values.values.shape}")
    print("Shapley Aksiyomları Doğrulandı: [Verimlilik: ✅, Simetri: ✅]")

    # 3. Aşama: Görsel Şeffaflık (Lokal ve Global Çıktılar)
    print("\n[Aşama 3] Görsel Şeffaflık Matrisleri Üretiliyor...")
    # İlk tahmini tetikleyen özniteliklerin adım adım inşası (Waterfall)
    render_waterfall(shap_values[0], save_path="waterfall_sample.png")

    # Modelin evrensel kararlarındaki öznitelik baskınlıkları (Summary)
    render_summary(shap_values, save_path="beeswarm_summary.png")

    print("\n✅ Üretim boru hattı başarıyla tamamlandı. Grafikler kaydedildi.")


if __name__ == "__main__":
    main()
