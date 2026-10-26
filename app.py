import sys
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from parse import build_parsed_df                              
from features import (clean_name, combined_text, numeric_features,  
                      severity_features, therapeutic_features)
from train import aggregate_compositions                         

TARGET = "net_score"
REVIEW_COLS = ["excellent_review", "average_review", "poor_review"]
VARIANT = "composition_net_score_noextremes"   

st.set_page_config(page_title="Medicine Satisfaction", page_icon="💊", layout="wide")

@st.cache_resource(show_spinner="Loading model...")
def load_model(variant):
    return joblib.load(ROOT / "models" / f"gbm_{variant}.pkl")


@st.cache_data(show_spinner="Loading data...")
def load_data(variant, drop_extremes):
    bundle = pd.read_pickle(ROOT / "data" / "processed" / "features.pkl")
    brands = bundle["df"]
    if drop_extremes:
        brands = brands[~(brands[REVIEW_COLS] == 100).any(axis=1)]
    brands = brands.reset_index(drop=True)
    comp, feats = aggregate_compositions(brands, bundle["feature_cols"], TARGET)

    oof = pd.read_csv(ROOT / "reports" / f"oof_{variant}.csv")
    comp = comp.merge(oof[["composition_clean", "pred_gbm_grouped"]],
                      on="composition_clean", how="left")

    parsed = pd.read_pickle(ROOT / "data" / "processed" / "medicines_parsed.pkl")
    strength = pd.to_numeric(parsed["main_strength_mg"], errors="coerce")
    median_log_strength = float(np.log1p(strength.median()))
    return brands, comp, feats, median_log_strength


@st.cache_resource(show_spinner=False)
def load_explainer(variant):
    try:
        import shap
        return shap.TreeExplainer(load_model(variant)["model"])
    except Exception:  # shap missing or model type unsupported
        return None

def features_for_new_medicine(name, composition, uses, side_effects, feats, median_log_strength):
    raw = pd.DataFrame([{
        "medicine_name": name, "composition": composition, "uses": uses,
        "side_effects": side_effects, "manufacturer": "", "excellent_review": 0,
        "average_review": 0, "poor_review": 0,
    }])
    p = build_parsed_df(raw)

    X = pd.DataFrame(0.0, index=[0], columns=feats)
    for block in (numeric_features(p), therapeutic_features(p), severity_features(p)):
        for c in block.columns:
            if c in X.columns:
                X.loc[0, c] = float(block[c].iloc[0])

    if p["main_strength_mg"].isna().iloc[0] or pd.isna(p["main_strength_mg"].iloc[0]):
        X.loc[0, "strength_missing"] = 1.0
        X.loc[0, "log_main_strength_mg"] = median_log_strength

    form_col = f"form_{p['dosage_form'].iloc[0]}"
    if form_col in X.columns:
        X.loc[0, form_col] = 1.0
    elif "form_other" in X.columns:
        X.loc[0, "form_other"] = 1.0

    for item in p["side_effect_list"].iloc[0]:
        c = f"se_{clean_name(item)}"
        if c in X.columns and c not in ("se_serious_count", "se_mild_count", "se_has_serious"):
            X.loc[0, c] = 1.0
    for salt in p["salt_names"].iloc[0]:
        c = f"salt_{clean_name(salt)}"
        if c in X.columns:
            X.loc[0, c] = 1.0

    X["text"] = combined_text(p).values
    return X, p


def predict(X, gbm):
    return float(gbm["model"].predict(X[gbm["feature_cols"]].astype("float64"))[0])


def describe_score(score, all_preds):
    pct = (all_preds < score).mean() * 100
    if pct >= 75:
        label = "High predicted satisfaction"
    elif pct >= 40:
        label = "Average predicted satisfaction"
    else:
        label = "Low predicted satisfaction"
    return pct, label


def shap_chart(explainer, X, feats, max_display=10):
    import matplotlib.pyplot as plt
    import shap
    values = explainer.shap_values(X[feats].astype("float64"))
    expl = shap.Explanation(values=values[0], base_values=float(np.ravel(explainer.expected_value)[0]),
                            data=X[feats].astype("float64").values[0], feature_names=feats)
    fig = plt.figure()
    shap.plots.waterfall(expl, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 5)
    fig.subplots_adjust(left=0.38, right=0.95, top=0.95, bottom=0.12)
    return fig

if not (ROOT / "models" / f"gbm_{VARIANT}.pkl").exists():
    st.error("Model not found. Run first:\n\n"
             "`python src/train.py --level composition --drop-extremes`")
    st.stop()

with st.sidebar:
    st.header("💊 Medicine Satisfaction")
    st.caption("Predicts how satisfied users on 1mg are with a medicine (net score = "
               "excellent % − poor %), from its composition, uses and side effects.")
    st.warning("Ratings reflect user perception, not clinical effectiveness. "
               "Not medical advice. Never use this to choose a treatment.", icon="⚠️")

gbm = load_model(VARIANT)
brands, comp, feats, median_log_strength = load_data(VARIANT, gbm.get("drop_extremes", False))
all_preds = gbm["model"].predict(comp[feats].astype("float64"))
explainer = load_explainer(VARIANT)

st.title("What drives satisfaction with medicines sold in India?")
st.caption(f"{len(brands):,} medicines · {len(comp):,} compositions · data scraped from 1mg (Kaggle)")

tab_explore, tab_predict = st.tabs(["🔎 Explore a composition", "🧪 Predict a new medicine"])

with tab_explore:
    st.subheader("Same composition, different brands")
    query = st.text_input("Search a salt (e.g. telmisartan, paracetamol, luliconazole)", "telmisartan")
    matches = comp[comp["composition_clean"].str.contains(query.strip(), case=False, regex=False)] \
        if query.strip() else comp.iloc[0:0]
    matches = matches.sort_values("n_brands", ascending=False)

    if matches.empty:
        st.info("No composition matches that search.")
    else:
        choice = st.selectbox(
            "Composition", matches["composition_clean"],
            format_func=lambda c: f"{c}  ({int(matches.set_index('composition_clean').loc[c, 'n_brands'])} brands)")
        row = comp[comp["composition_clean"] == choice].iloc[0]
        sub = brands[brands["composition_clean"] == choice].sort_values("net_score", ascending=False)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Brands", int(row["n_brands"]))
        c2.metric("Actual mean net score", f"{row[TARGET]:.1f}")
        oof_pred = row.get("pred_gbm_grouped")
        c3.metric("Model prediction*", f"{oof_pred:.1f}" if pd.notna(oof_pred) else "n/a")
        spread = sub["excellent_review"].max() - sub["excellent_review"].min() if len(sub) > 1 else 0
        c4.metric("Spread in 'excellent %'", f"{spread:.0f} pts")
        st.caption("*Out-of-fold prediction: made by a model that never saw this salt combination "
                   "during training, so it is an honest test.")

        if row["n_brands"] < 3:
            st.warning("Fewer than 3 brands: this average rests on very few reviews and is unreliable.")

        chart = sub.set_index("medicine_name")[REVIEW_COLS].rename(columns={
            "excellent_review": "Excellent %", "average_review": "Average %", "poor_review": "Poor %"})
        st.bar_chart(chart.head(30), horizontal=True, stack=True, height=max(250, 22 * min(len(chart), 30)),
                     color=["#2a9d8f", "#e9c46a", "#e76f51"])
        st.dataframe(sub[["medicine_name", "manufacturer", "dosage_form"] + REVIEW_COLS + ["net_score"]],
                     hide_index=True, width="stretch")

with tab_predict:
    st.subheader("Describe a medicine and get a predicted satisfaction score")
    st.caption("Use the same style as 1mg listings. Side effects: separate each one with a "
               "capital letter or comma.")

    examples = {
        "Blood pressure tablet": ("Telmisart 40 Tablet", "Telmisartan (40mg)",
                                  "Treatment of Hypertension (high blood pressure) Prevention of Heart attack and stroke",
                                  "Dizziness Back pain Diarrhea Upper respiratory tract infection"),
        "Antifungal cream": ("Fungiclear Cream", "Luliconazole (1% w/w)",
                             "Treatment of Fungal skin infections",
                             "Skin peeling Dry skin Itching Application site irritation"),
        "Cough syrup": ("Coughfree Syrup", "Ambroxol (30mg/5ml) + Guaifenesin (50mg/5ml) + Levosalbutamol (1mg/5ml)",
                        "Treatment of Cough with mucus",
                        "Nausea Vomiting Headache Tremors Palpitations"),
    }
    pick = st.radio("Start from an example", list(examples), horizontal=True)
    ex = examples[pick]

    with st.form("predict"):
        a, b = st.columns(2)
        name = a.text_input("Medicine name (include the form: Tablet, Syrup, Cream...)", ex[0])
        composition = b.text_input("Composition", ex[1], help="e.g. Amoxycillin (500mg) + Clavulanic Acid (125mg)")
        uses = st.text_area("Uses", ex[2], height=70)
        side_effects = st.text_area("Side effects", ex[3].replace(",", ""), height=70)
        st.form_submit_button("Predict", type="primary")

    # A prediction is shown on first load too; the button re-runs it with your edits
    X, parsed_row = features_for_new_medicine(name, composition, uses, side_effects.replace(",", " "),
                                              feats, median_log_strength)
    score = predict(X, gbm)
    pct, label = describe_score(score, all_preds)

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted net score", f"{score:.1f}", help="Excellent % minus poor %, predicted")
    m2.metric("Percentile vs. all compositions", f"{pct:.0f}th")
    m3.metric("Verdict", label.split(" predicted")[0])
    st.caption(f"Dataset average: {comp[TARGET].mean():.1f}")

    with st.expander("How the app read your input"):
        st.write({
            "salts": parsed_row["salt_names"].iloc[0],
            "dosage form": parsed_row["dosage_form"].iloc[0],
            "side effects": parsed_row["side_effect_list"].iloc[0],
            "therapeutic areas": [c[3:] for c in feats
                                  if c.startswith("tx_") and c not in ("tx_n_categories", "tx_unmatched")
                                  and X.loc[0, c] == 1],
        })

    if explainer is not None:
        st.markdown("**Why this score?** Each bar shows how a feature pushed the "
                    "prediction up (red) or down (blue) from the average.")
        st.pyplot(shap_chart(explainer, X, gbm["feature_cols"]), clear_figure=True)
    else:
        st.info("Install `shap` to see an explanation of each prediction: python -m pip install shap")