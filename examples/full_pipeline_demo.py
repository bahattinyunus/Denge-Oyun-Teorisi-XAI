"""
Uçtan Uca Oyun Teorisi XAI Boru Hattı (Full Pipeline Demo)
Nash Dengesi (Rekabet) -> Maynard Smith ESS (Evrimsel Filtre) -> Shapley (Adil Dağıtım)
"""

import numpy as np
import xgboost as xgb
import shap
from core_theory.nash_equilibrium import (
    FeaturePayoffMatrix,
    NashFeatureRanker,
    detect_feature_suppression,
    compute_feature_synergy,
)
from core_theory.maynard_ess_filter import MaynardESSFilter
from core_theory.shapley_axioms import verify_all_shapley_axioms
from data_ops.synthetic_coalitions import generate_coalition_dataset, create_axiomatic_game
from explainers.waterfall_local import render_waterfall
from explainers.beeswarm_global import render_summary
from explainers.cost_optimizer import benchmark_shap_latency


def run_pipeline():
    print("=" * 75)
    print("♟️  DENGE-OYUN-TEORİSİ-XAI: KAPSAMLI OYUN KURAMSAL DENETİM SİSTEMİ")
    print("=" * 75)

    # 1. Sentetik Veri ile Zemin Gerçekliği
    print("\n[ADIM 1] Oyun Kuramsal Sentetik Veri Kümesi Oluşturuluyor...")
    dataset = generate_coalition_dataset(n_samples=600, random_state=42)
    X, y = dataset.X, dataset.y
    names = dataset.feature_names
    print(f"Veri Boyutu: {X.shape}, İncelenen Öznitelikler: {names}")

    # 2. Nash Rekabet ve Baskılama Analizi
    print("\n[ADIM 2] John Nash Rekabet ve Payoff Matrisi Kuruluyor...")
    payoff = FeaturePayoffMatrix.from_correlations(X, y, feature_names=names)
    ranker = NashFeatureRanker(payoff)
    ranked = ranker.rank()
    print("🏆 Nash Dengesi Öznitelik Sıralaması:")
    for r in ranked:
        print(f"   • {r['feature']:<15} | Nash Ağırlığı: {r['nash_weight']:.4f} | İzole: {r['isolated_payoff']:.4f}")

    suppressions = detect_feature_suppression(payoff, threshold=0.1)
    if suppressions:
        print(f"\n⚠️  Tespit Edilen Baskılamalar (Suppression): {len(suppressions)} adet")
        for s in suppressions[:2]:
            print(f"   - {s['suppressed_feature']} <-- {s['suppressor_feature']} tarafından baskılanıyor (%{s['relative_drop']*100:.1f} düşüş)")

    synergies = compute_feature_synergy(payoff, threshold=0.02)
    if synergies:
        print(f"✨ Tespit Edilen Sinerjiler: {len(synergies)} adet")
        for syn in synergies[:2]:
            print(f"   - {syn['feature_1']} & {syn['feature_2']} (Sinerji Kazancı: +{syn['synergy_gain']:.4f})")

    # 3. Maynard Smith ESS Filtresi
    print("\n[ADIM 3] Maynard Smith Evrimsel Kararlı Strateji (ESS) Filtresi...")
    ess_filter = MaynardESSFilter(n_generations=60, mutation_rate=0.15, random_state=42)
    ess_res = ess_filter.fit_filter(X, y, feature_names=names)
    print(f"🛡️  Hayatta Kalan Sağlam (ESS) Öznitelikler: {ess_res['selected_features']}")
    print(f"❌ Elenen Kırılgan / Kukla Öznitelikler: {ess_res['eliminated_features']}")

    # 4. Kooperatif Oyun (Shapley & SHAP)
    print("\n[ADIM 4] Lloyd Shapley Aksiyomatik Denetimi...")
    test_game = create_axiomatic_game(n_players=4)
    axiom_report = verify_all_shapley_axioms(test_game)
    print(f"📜 {axiom_report['summary']}")

    # 5. Model Eğitimi ve Açıklanabilirlik
    print("\n[ADIM 5] Sağlam Özniteliklerle Model Eğitimi ve Açıklama...")
    robust_idx = ess_res["selected_indices"]
    X_sub = X[:, robust_idx]
    names_sub = ess_res["selected_features"]

    model = xgb.XGBRegressor(n_estimators=60, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_sub, y)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X_sub)

    render_waterfall(shap_vals[0])
    render_summary(shap_vals)

    # 6. Gecikme Benchmarkı
    bench = benchmark_shap_latency(model, X_sub, n_instances=50)
    print(f"\n⚡ Edge-AI Hız Testi: Örnek başına gecikme: {bench['ms_per_sample']:.3f} ms | Edge Ready: {bench['edge_ready']}")

    print("\n" + "=" * 75)
    print("🎉 KAPSAMLI DENETİM VE BORU HATTI TESTİ BAŞARIYLA BİTTİ!")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
