import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "Medicine_Details.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_PATH = OUT_DIR / "medicines_parsed.pkl"

# The Kaggle CSV uses headers like "Medicine Name" and "Excellent Review %".
# We map them to simple snake_case names.
COLUMN_MAP = {
    "medicine name": "medicine_name",
    "composition": "composition",
    "uses": "uses",
    "side_effects": "side_effects",
    "side effects": "side_effects",
    "image url": "image_url",
    "manufacturer": "manufacturer",
    "excellent review %": "excellent_review",
    "average review %": "average_review",
    "poor review %": "poor_review",
}

DOSAGE_FORMS = [
    "oral suspension", "oral solution", "dry syrup", "eye drop", "ear drop",
    "nasal spray", "tablet", "capsule", "syrup", "suspension", "injection",
    "infusion", "cream", "ointment", "gel", "lotion", "drop", "inhaler",
    "rotacap", "respules", "spray", "powder", "sachet", "solution",
    "shampoo", "soap", "patch", "suppository", "granules", "lozenge",
]

# Modified / extended release markers that appear in brand names
RELEASE_FLAGS = ["sr", "er", "xr", "cr", "mr", "pr", "od", "xl", "la"]

USE_STARTERS = r"(?:Treatment|Prevention|Relief|Management|Control|Reduction|Improvement)"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the CSV, standardise column names, drop the image column and duplicates."""
    df = pd.read_csv(path)
    df.columns = [COLUMN_MAP.get(c.strip().lower(), c.strip().lower()) for c in df.columns]

    missing = {"medicine_name", "composition", "uses", "side_effects",
               "manufacturer", "excellent_review", "average_review", "poor_review"} - set(df.columns)
    if missing:
        raise ValueError(f"Expected columns not found: {missing}. Found: {list(df.columns)}")

    df = df.drop(columns=["image_url"], errors="ignore")
    df = df.drop_duplicates().reset_index(drop=True)

    for col in ["medicine_name", "composition", "uses", "side_effects", "manufacturer"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    for col in ["excellent_review", "average_review", "poor_review"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Composition:  "Amoxycillin (500mg) + Clavulanic Acid (125mg)"
# ---------------------------------------------------------------------------
_SALT_RE = re.compile(r"^(?P<salt>.+?)\s*\((?P<strength>[^)]*)\)\s*$")
_NUM_UNIT_RE = re.compile(r"([\d.]+)\s*(mcg|mg|g|iu|%|ml)", re.IGNORECASE)
_TO_MG = {"mg": 1.0, "mcg": 0.001, "g": 1000.0}


def parse_composition(text: str) -> list[dict]:
    """
    Split a composition string into salts.
    Returns a list like:
        [{"salt": "amoxycillin", "strength": "500mg", "value": 500.0,
          "unit": "mg", "strength_mg": 500.0}, ...]
    """
    if not text:
        return []

    salts = []
    for part in text.split("+"):
        part = part.strip()
        if not part:
            continue

        m = _SALT_RE.match(part)
        if m:
            salt, strength = m.group("salt"), m.group("strength").strip()
        else:
            salt, strength = part, ""

        value, unit, strength_mg = None, None, None
        nm = _NUM_UNIT_RE.search(strength)
        if nm:
            value = float(nm.group(1))
            unit = nm.group(2).lower()
            if unit in _TO_MG:
                strength_mg = value * _TO_MG[unit]

        salts.append({
            "salt": re.sub(r"\s+", " ", salt).strip().lower(),
            "strength": strength.lower(),
            "value": value,
            "unit": unit,
            "strength_mg": strength_mg,
        })
    return salts


def canonical_composition(salts: list[dict]) -> str:
    """
    A stable key for 'same medicine, different brand'.
    Sorted so 'A + B' and 'B + A' match. Used later for GroupKFold.
    """
    return " + ".join(sorted(f"{s['salt']} ({s['strength']})" for s in salts))


# ---------------------------------------------------------------------------
# Side effects:  "Nausea Vomiting Dry mouth Headache"
# Each new side effect starts with a capital letter, so we split before capitals.
# Multi-word items like "Dry mouth" stay together because "mouth" is lowercase.
# ---------------------------------------------------------------------------
_SPLIT_BEFORE_CAPITAL = re.compile(r"\s+(?=[A-Z])")


def split_side_effects(text: str) -> list[str]:
    if not text:
        return []
    items = [s.strip(" ,.;").lower() for s in _SPLIT_BEFORE_CAPITAL.split(text)]
    # Drop empties and keep order while removing repeats
    seen, out = set(), []
    for s in items:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Uses:  "Treatment of Bacterial infections Prevention of Heart attack"
# ---------------------------------------------------------------------------
_SPLIT_USES = re.compile(rf"\s+(?={USE_STARTERS}\b)")


def split_uses(text: str) -> list[str]:
    if not text:
        return []
    return [u.strip(" ,.;").lower() for u in _SPLIT_USES.split(text) if u.strip()]


# ---------------------------------------------------------------------------
# Medicine name:  "Augmentin 625 Duo Tablet", "Dolo 650 Tablet SR"
# ---------------------------------------------------------------------------
def extract_dosage_form(name: str) -> str:
    low = name.lower()
    for form in DOSAGE_FORMS:  # multi-word forms are listed first
        if re.search(rf"\b{re.escape(form)}s?\b", low):
            return form
    return "other"


def is_modified_release(name: str) -> int:
    tokens = re.findall(r"[a-z]+", name.lower())
    return int(any(t in RELEASE_FLAGS for t in tokens))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_parsed_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Composition
    out["salts"] = out["composition"].apply(parse_composition)
    out["salt_names"] = out["salts"].apply(lambda s: [x["salt"] for x in s])
    out["n_salts"] = out["salts"].apply(len)
    out["is_combination"] = (out["n_salts"] > 1).astype(int)
    out["main_salt"] = out["salt_names"].apply(lambda s: s[0] if s else "unknown")
    out["main_strength_mg"] = out["salts"].apply(
        lambda s: s[0]["strength_mg"] if s and s[0]["strength_mg"] is not None else None
    )
    out["composition_clean"] = out["salts"].apply(canonical_composition)

    # Side effects and uses
    out["side_effect_list"] = out["side_effects"].apply(split_side_effects)
    out["n_side_effects"] = out["side_effect_list"].apply(len)
    out["use_list"] = out["uses"].apply(split_uses)
    out["n_uses"] = out["use_list"].apply(len)

    # Name-based fields
    out["dosage_form"] = out["medicine_name"].apply(extract_dosage_form)
    out["is_modified_release"] = out["medicine_name"].apply(is_modified_release)

    # Manufacturer portfolio size
    out["manufacturer"] = out["manufacturer"].str.strip()
    out["manufacturer_n_products"] = out.groupby("manufacturer")["medicine_name"].transform("count")

    # Targets (the three review % columns are complements: never use one as a feature)
    out["net_score"] = out["excellent_review"] - out["poor_review"]
    out["review_sum"] = out[["excellent_review", "average_review", "poor_review"]].sum(axis=1)

    return out


def print_summary(df: pd.DataFrame) -> None:
    print(f"Rows after cleaning: {len(df):,}")
    print(f"Unique compositions: {df['composition_clean'].nunique():,}")
    print(f"Unique manufacturers: {df['manufacturer'].nunique():,}")
    print(f"Review % sum (should be ~100): mean={df['review_sum'].mean():.1f}, "
          f"min={df['review_sum'].min():.0f}, max={df['review_sum'].max():.0f}")
    print("\nDosage forms:")
    print(df["dosage_form"].value_counts().head(10).to_string())
    print("\nSide effects per medicine:")
    print(df["n_side_effects"].describe().round(1).to_string())
    print("\nSample parsed rows:")
    cols = ["medicine_name", "composition_clean", "dosage_form", "n_side_effects", "net_score"]
    print(df[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    raw = load_raw()
    parsed = build_parsed_df(raw)
    print_summary(parsed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parsed.to_pickle(OUT_PATH)  # pickle keeps the list columns intact
    print(f"\nSaved -> {OUT_PATH.relative_to(ROOT)}")