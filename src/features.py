import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "medicines_parsed.pkl"
OUT_PATH = ROOT / "data" / "processed" / "features.pkl"

TOP_N_SIDE_EFFECTS = 60
TOP_N_SALTS = 60
MIN_FORM_COUNT = 30        
N_EMBED_COMPONENTS = 20

THERAPEUTIC_MAP = {
    "antifungal":    [r"fungal"],
    "antiviral":     [r"viral", r"hiv", r"herpes", r"hepatitis", r"influenza", r"covid"],
    "antiparasitic": [r"parasitic", r"worms?", r"malaria", r"amoebiasis", r"lice"],
    "antibiotic":    [r"bacterial", r"tuberculosis", r"typhoid", r"urinary tract",
                      r"(?<!fungal )(?<!viral )(?<!parasitic )(?<!worm )infections?"],
    "pain_fever":    [r"pain", r"fever", r"inflammation", r"arthritis", r"migraine", r"gout"],
    "cardio":        [r"blood pressure", r"hypertension", r"heart", r"angina", r"cholesterol",
                      r"stroke", r"blood clots?"],
    "diabetes":      [r"diabetes", r"blood sugar"],
    "respiratory":   [r"cough", r"asthma", r"copd", r"allerg\w*", r"cold", r"rhinitis",
                      r"bronch\w*"],
    "gastro":        [r"acidity", r"peptic", r"ulcers?", r"reflux", r"constipation",
                      r"diarrh(?:o)?ea", r"vomiting", r"nausea", r"indigestion",
                      r"irritable bowel"],
    "mental_neuro":  [r"depression", r"anxiety", r"epilepsy", r"seizures?", r"schizophrenia",
                      r"insomnia", r"parkinson\w*", r"bipolar", r"alzheimer\w*", r"panic"],
    "skin":          [r"skin", r"acne", r"eczema", r"psoriasis", r"dermatitis", r"scabies"],
    "nutrition":     [r"deficiency", r"vitamins?", r"an(?:a)?emia", r"supplements?"],
    "hormonal":      [r"thyroid", r"contracep\w*", r"infertility", r"menopause", r"hormon\w*",
                      r"osteoporosis"],
    "cancer":        [r"cancer", r"tumou?r", r"leuk(?:a)?emia", r"lymphoma"],
    "eye_ear":       [r"eyes?", r"glaucoma", r"ears?"],
}
THERAPEUTIC_RE = {cat: re.compile(r"\b(?:" + "|".join(p) + r")\b", re.IGNORECASE)
                  for cat, p in THERAPEUTIC_MAP.items()}

SERIOUS_TERMS = [
    "liver", "kidney", "renal", "seizure", "convulsion", "anaphyla", "bleeding",
    "arrhythmia", "irregular heart", "suicid", "stevens-johnson", "hypoglycemia",
    "low blood sugar", "pancreatitis", "jaundice", "angioedema", "blood dyscrasia",
]
MILD_TERMS = [
    "nausea", "headache", "dizziness", "vomiting", "diarrhea", "constipation",
    "rash", "dry mouth", "sleepiness", "drowsiness", "fatigue", "stomach pain",
    "abdominal pain", "itching", "indigestion", "flatulence",
]


def clean_name(s: str) -> str:
    """Make a safe column name (LightGBM dislikes special characters)."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()


def numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["n_salts"] = df["n_salts"]
    out["is_combination"] = df["is_combination"]
    out["n_side_effects"] = df["n_side_effects"]
    out["n_uses"] = df["n_uses"]
    out["is_modified_release"] = df["is_modified_release"]
    out["log_manufacturer_n_products"] = np.log1p(df["manufacturer_n_products"])

    strength = pd.to_numeric(df["main_strength_mg"], errors="coerce")
    out["strength_missing"] = strength.isna().astype(int)
    out["log_main_strength_mg"] = np.log1p(strength.fillna(strength.median()))
    return out


def dosage_form_features(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["dosage_form"].value_counts()
    keep = counts[counts >= MIN_FORM_COUNT].index
    form = df["dosage_form"].where(df["dosage_form"].isin(keep), "other")
    return pd.get_dummies(form, prefix="form", dtype=int)


def therapeutic_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df["uses"].str.lower()
    out = pd.DataFrame(index=df.index)
    for cat, rx in THERAPEUTIC_RE.items():
        out[f"tx_{cat}"] = text.apply(lambda t, r=rx: int(bool(r.search(t))))
    out["tx_n_categories"] = out.sum(axis=1)
    out["tx_unmatched"] = (out["tx_n_categories"] == 0).astype(int)
    return out


def primary_category(df: pd.DataFrame) -> pd.Series:
    """First matching category, kept as a readable label for EDA (not a feature)."""
    text = df["uses"].str.lower()

    def first_match(t: str) -> str:
        for cat, rx in THERAPEUTIC_RE.items():
            if rx.search(t):
                return cat
        return "other"

    return text.apply(first_match)


def severity_features(df: pd.DataFrame) -> pd.DataFrame:
    def count_terms(items: list[str], terms: list[str]) -> int:
        return sum(any(t in item for t in terms) for item in items)

    out = pd.DataFrame(index=df.index)
    out["se_serious_count"] = df["side_effect_list"].apply(lambda x: count_terms(x, SERIOUS_TERMS))
    out["se_mild_count"] = df["side_effect_list"].apply(lambda x: count_terms(x, MILD_TERMS))
    out["se_has_serious"] = (out["se_serious_count"] > 0).astype(int)
    return out


def multi_hot(lists: pd.Series, top_n: int, prefix: str) -> pd.DataFrame:
    """Flags for the top_n most common items. Uses no target, so no leakage."""
    counts = lists.explode().dropna().value_counts()
    vocab = counts.head(top_n).index.tolist()
    out = pd.DataFrame(index=lists.index)
    sets = lists.apply(set)
    for item in vocab:
        out[f"{prefix}_{clean_name(item)}"] = sets.apply(lambda s, i=item: int(i in s))
    # Columns can collide after clean_name; merge duplicates
    out = out.T.groupby(level=0).max().T
    return out


def combined_text(df: pd.DataFrame) -> pd.Series:
    """One text field for TF-IDF / embeddings."""
    return (
        "uses: " + df["uses"].str.lower()
        + " | side effects: " + df["side_effect_list"].apply(", ".join)
        + " | salts: " + df["salt_names"].apply(", ".join)
    )


def embedding_features(text: pd.Series, n_components: int = N_EMBED_COMPONENTS) -> pd.DataFrame:
    """Sentence embeddings reduced with PCA. Optional (slow on first run)."""
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(text.tolist(), batch_size=64, show_progress_bar=True)
    reduced = PCA(n_components=n_components, random_state=42).fit_transform(emb)
    cols = [f"emb_{i}" for i in range(n_components)]
    return pd.DataFrame(reduced, index=text.index, columns=cols)



def target_encode(train_col: pd.Series, train_y: pd.Series, apply_col: pd.Series,
                  smoothing: float = 20.0) -> pd.Series:
    """
    Smoothed mean of the target per category, learned on train rows only.
    Small categories are pulled toward the global mean.
    """
    global_mean = train_y.mean()
    stats = pd.DataFrame({"cat": train_col, "y": train_y}).groupby("cat")["y"].agg(["mean", "count"])
    smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    return apply_col.map(smooth).fillna(global_mean)


def build_features(df: pd.DataFrame, use_embeddings: bool = False) -> dict:
    df = df.dropna(subset=["excellent_review", "poor_review"]).reset_index(drop=True)

    blocks = [
        numeric_features(df),
        dosage_form_features(df),
        therapeutic_features(df),
        severity_features(df),
        multi_hot(df["side_effect_list"], TOP_N_SIDE_EFFECTS, "se"),
        multi_hot(df["salt_names"], TOP_N_SALTS, "salt"),
    ]

    text = combined_text(df)
    if use_embeddings:
        blocks.append(embedding_features(text))

    X = pd.concat(blocks, axis=1)
    feature_cols = X.columns.tolist()

    tier = pd.qcut(df["net_score"], q=3, labels=["low", "medium", "high"], duplicates="drop")

    meta = pd.DataFrame({
        "medicine_name": df["medicine_name"],
        "manufacturer": df["manufacturer"],
        "composition_clean": df["composition_clean"],
        "dosage_form": df["dosage_form"],
        "primary_category": primary_category(df),
        "text": text,
        "excellent_review": df["excellent_review"],
        "average_review": df["average_review"],
        "poor_review": df["poor_review"],
        "net_score": df["net_score"],
        "satisfaction_tier": tier.astype(str),
    })

    full = pd.concat([meta, X], axis=1)

    return {
        "df": full,
        "feature_cols": feature_cols,
        "target_cols": ["excellent_review", "net_score", "satisfaction_tier"],
        "group_col": "composition_clean",
        "text_col": "text",
    }


def print_summary(bundle: dict) -> None:
    df, cols = bundle["df"], bundle["feature_cols"]
    print(f"Rows: {len(df):,}")
    print(f"Features: {len(cols)}")
    groups = {}
    for c in cols:
        groups.setdefault(c.split("_")[0], 0)
        groups[c.split("_")[0]] += 1
    print("Feature groups:", ", ".join(f"{k}={v}" for k, v in sorted(groups.items())))
    print(f"Composition groups (for GroupKFold): {df[bundle['group_col']].nunique():,}")
    print("\nPrimary therapeutic category:")
    print(df["primary_category"].value_counts().to_string())
    print("\nSatisfaction tiers:")
    print(df["satisfaction_tier"].value_counts().to_string())
    print(f"\nnet_score: mean={df['net_score'].mean():.1f}, "
          f"std={df['net_score'].std():.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", action="store_true",
                        help="add sentence-embedding features (needs sentence-transformers)")
    args = parser.parse_args()

    parsed = pd.read_pickle(IN_PATH)
    bundle = build_features(parsed, use_embeddings=args.embeddings)
    print_summary(bundle)

    pd.to_pickle(bundle, OUT_PATH)
    print(f"\nSaved -> {OUT_PATH.relative_to(ROOT)}")