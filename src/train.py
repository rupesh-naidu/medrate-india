import argparse
import os
import re
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from features import target_encode

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "features.pkl"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

N_SPLITS = 5
SEED = 42
MAX_WEIGHT = 20                     
RIDGE_ALPHAS = [1, 3, 10, 30, 100, 300]

LGBM_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                   min_child_samples=30, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.7, reg_lambda=1.0, random_state=SEED, verbose=-1)


def metrics(y_true, y_pred, weight=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred, sample_weight=weight),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=weight)),
        "R2": r2_score(y_true, y_pred, sample_weight=weight),
        "Spearman": spearmanr(y_true, y_pred).correlation,
    }


_LGBM_OK = HAS_LGBM


def make_gbm(use_lgbm: bool, min_leaf: int = 30):
    if use_lgbm:
        return lgb.LGBMRegressor(**{**LGBM_PARAMS, "min_child_samples": min_leaf})
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03,
                                         min_samples_leaf=min_leaf, random_state=SEED)


def fit_gbm(X: pd.DataFrame, y, sample_weight=None, min_leaf: int = 30):
    """
    LightGBM can crash on some Windows setups ("access violation"), so we pass clean
    float64 data and contiguous numpy arrays. If it still fails, we switch to
    sklearn's HistGradientBoosting for the rest of the run.
    """
    global _LGBM_OK
    X = X.astype("float64")
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    w = None if sample_weight is None else np.ascontiguousarray(
        np.asarray(sample_weight, dtype=np.float64))

    if _LGBM_OK:
        try:
            return make_gbm(True, min_leaf).fit(X, y, sample_weight=w)
        except Exception as e:  # noqa: BLE001
            print(f"  ! LightGBM failed ({type(e).__name__}: {e}). "
                  f"Switching to sklearn HistGradientBoosting for this run.")
            _LGBM_OK = False
    return make_gbm(False, min_leaf).fit(X, y, sample_weight=w)


def fit_ridge(tr_df, y_tr, feature_cols, sample_weight=None, tune=False):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000,
                            sublinear_tf=True)
    scaler = StandardScaler()
    X = sparse.hstack([tfidf.fit_transform(tr_df["text"]),
                       sparse.csr_matrix(scaler.fit_transform(tr_df[feature_cols]))]).tocsr()
    model = RidgeCV(alphas=RIDGE_ALPHAS) if tune else Ridge(alpha=10.0)
    model.fit(X, np.asarray(y_tr, dtype=float), sample_weight=sample_weight)
    return {"tfidf": tfidf, "scaler": scaler, "model": model, "feature_cols": feature_cols}


def predict_ridge(bundle, df):
    X = sparse.hstack([bundle["tfidf"].transform(df["text"]),
                       sparse.csr_matrix(bundle["scaler"].transform(df[bundle["feature_cols"]]))]).tocsr()
    return bundle["model"].predict(X)


def add_manufacturer_te(train_df, val_df, y_train, inner_splits=5):
    val_te = target_encode(train_df["manufacturer"], y_train, val_df["manufacturer"])
    train_te = pd.Series(np.nan, index=train_df.index)
    for in_tr, in_va in KFold(inner_splits, shuffle=True, random_state=SEED).split(train_df):
        tr_idx, va_idx = train_df.index[in_tr], train_df.index[in_va]
        train_te.loc[va_idx] = target_encode(train_df.loc[tr_idx, "manufacturer"],
                                             y_train.loc[tr_idx],
                                             train_df.loc[va_idx, "manufacturer"])
    return train_te.values, val_te.values

def salt_key(composition: str) -> str:
    """'amoxycillin (500mg) + clavulanic acid (125mg)' -> 'amoxycillin + clavulanic acid'"""
    return re.sub(r"\s*\([^)]*\)", "", composition).strip()


def aggregate_compositions(df, feature_cols, target):
    keep_feats = [c for c in feature_cols if c != "log_manufacturer_n_products"]

    agg = {c: "mean" for c in keep_feats}
    agg.update({target: "mean", "text": "first", "primary_category": "first",
                "medicine_name": "count"})
    comp = (df.groupby("composition_clean").agg(agg)
              .rename(columns={"medicine_name": "n_brands"}).reset_index())
    comp["target_std"] = df.groupby("composition_clean")[target].std().values
    comp["salt_key"] = comp["composition_clean"].apply(salt_key)
    comp["weight"] = np.minimum(comp["n_brands"], MAX_WEIGHT)
    return comp, keep_feats

def cv_brand(df, y, feature_cols, splits, name):
    oof = np.zeros(len(df))
    for tr, va in splits:
        tr_df, va_df, y_tr = df.iloc[tr], df.iloc[va], y.iloc[tr]
        if name == "baseline_mean":
            oof[va] = y_tr.mean()
        elif name == "ridge_tfidf":
            oof[va] = predict_ridge(fit_ridge(tr_df, y_tr, feature_cols), va_df)
        else:
            X_tr, X_va = tr_df[feature_cols].copy(), va_df[feature_cols].copy()
            if name == "gbm":
                X_tr["manufacturer_te"], X_va["manufacturer_te"] = \
                    add_manufacturer_te(tr_df, va_df, y_tr)
            oof[va] = fit_gbm(X_tr, y_tr).predict(X_va.astype("float64"))
    return oof


def cv_composition(comp, y, feature_cols, splits, name):
    oof = np.zeros(len(comp))
    w = comp["weight"].values
    for tr, va in splits:
        tr_df, va_df, y_tr = comp.iloc[tr], comp.iloc[va], y.iloc[tr]
        if name == "baseline_mean":
            oof[va] = np.average(y_tr, weights=w[tr])
        elif name == "ridge_tfidf":
            oof[va] = predict_ridge(fit_ridge(tr_df, y_tr, feature_cols,
                                              sample_weight=w[tr], tune=True), va_df)
        elif name == "gbm":
            oof[va] = fit_gbm(tr_df[feature_cols], y_tr, sample_weight=w[tr],
                              min_leaf=20).predict(va_df[feature_cols].astype("float64"))
    return oof


def print_row(name, split, m):
    print(f"  {name:<22} {split:<8} MAE={m['MAE']:6.2f}  RMSE={m['RMSE']:6.2f}  "
          f"R2={m['R2']:6.3f}  Spearman={m['Spearman']:6.3f}")


def run_brand(df, feature_cols, target, suffix):
    y = df[target].astype(float)
    groups = df["composition_clean"]
    print(f"\nLevel: brand | target: {target} | rows: {len(df):,} | "
          f"features: {len(feature_cols)} | composition groups: {groups.nunique():,}")

    grouped = list(GroupKFold(N_SPLITS).split(df, y, groups))
    random_ = list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(df))
    experiments = [("baseline_mean", "grouped"), ("ridge_tfidf", "grouped"),
                   ("gbm_no_manufacturer", "grouped"), ("gbm", "grouped"), ("gbm", "random")]

    rows, oofs = [], {}
    for name, split in experiments:
        t0 = time.time()
        oof = cv_brand(df, y, feature_cols, grouped if split == "grouped" else random_, name)
        m = metrics(y, oof)
        rows.append({"model": name, "split": split, **m, "seconds": round(time.time() - t0, 1)})
        oofs[f"pred_{name}_{split}"] = oof
        print_row(name, split, m)

    results = pd.DataFrame(rows)
    oof_df = df[["medicine_name", "manufacturer", "composition_clean",
                 "primary_category", target]].assign(**oofs)

    X = df[feature_cols].copy()
    te = pd.Series(np.nan, index=df.index)
    for tr, va in KFold(N_SPLITS, shuffle=True, random_state=SEED).split(df):
        te.iloc[va] = target_encode(df["manufacturer"].iloc[tr], y.iloc[tr],
                                    df["manufacturer"].iloc[va]).values
    X["manufacturer_te"] = te.values
    gm = y.mean()
    st = pd.DataFrame({"m": df["manufacturer"], "y": y}).groupby("m")["y"].agg(["mean", "count"])
    te_map = ((st["mean"] * st["count"] + gm * 20) / (st["count"] + 20)).to_dict()

    gbm_bundle = {"model": fit_gbm(X, y), "feature_cols": feature_cols + ["manufacturer_te"],
                  "manufacturer_te_map": te_map, "global_mean": gm}
    return results, oof_df, gbm_bundle, None

def run_composition(df, feature_cols, target, suffix):
    comp, feats = aggregate_compositions(df, feature_cols, target)
    y = comp[target].astype(float)
    w = comp["weight"].values

    print(f"\nLevel: composition | target: mean {target} across brands | "
          f"compositions: {len(comp):,} | salt groups: {comp['salt_key'].nunique():,} | "
          f"features: {len(feats)}")
    print(f"Brands per composition: median {comp['n_brands'].median():.0f}, "
          f"{(comp['n_brands'] >= 3).mean():.0%} have 3+ brands, "
          f"{(comp['n_brands'] >= 5).mean():.0%} have 5+")

    grouped = list(GroupKFold(N_SPLITS).split(comp, y, comp["salt_key"]))
    random_ = list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(comp))
    experiments = [("baseline_mean", "grouped"), ("ridge_tfidf", "grouped"),
                   ("gbm", "grouped"), ("ridge_tfidf", "random")]

    rows, oofs = [], {}
    for name, split in experiments:
        t0 = time.time()
        oof = cv_composition(comp, y, feats, grouped if split == "grouped" else random_, name)
        oofs[f"pred_{name}_{split}"] = oof

        for subset, mask in [("all", np.ones(len(comp), bool)),
                             ("3+ brands", comp["n_brands"].values >= 3),
                             ("5+ brands", comp["n_brands"].values >= 5)]:
            m = metrics(y[mask], oof[mask])
            mw = metrics(y[mask], oof[mask], weight=w[mask])
            rows.append({"model": name, "split": split, "subset": subset, "n": int(mask.sum()),
                         **m, "R2_weighted": mw["R2"], "MAE_weighted": mw["MAE"],
                         "seconds": round(time.time() - t0, 1)})
        print_row(name, split, metrics(y, oof, weight=w))

    results = pd.DataFrame(rows)
    oof_df = comp[["composition_clean", "salt_key", "primary_category", "n_brands",
                   target]].assign(**oofs)

    gbm_bundle = {"model": fit_gbm(comp[feats], y, sample_weight=w, min_leaf=20),
                  "feature_cols": feats}
    ridge_bundle = fit_ridge(comp, y, feats, sample_weight=w, tune=True)
    print(f"Ridge alpha chosen on full data: {ridge_bundle['model'].alpha_}")
    return results, oof_df, gbm_bundle, ridge_bundle

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="brand", choices=["brand", "composition"])
    p.add_argument("--target", default="net_score", choices=["net_score", "excellent_review"])
    p.add_argument("--drop-extremes", action="store_true",
                   help="drop brand rows where any review %% is exactly 100 "
                        "(likely very few reviews)")
    args = p.parse_args()

    bundle = pd.read_pickle(IN_PATH)
    df = bundle["df"].reset_index(drop=True)
    feature_cols = bundle["feature_cols"]

    review_cols = ["excellent_review", "average_review", "poor_review"]
    extreme = (df[review_cols] == 100).any(axis=1)
    print(f"Rows with a 100% review value (likely very few reviews): "
          f"{extreme.sum():,} of {len(df):,} ({extreme.mean():.1%})")
    if args.drop_extremes:
        df = df[~extreme].reset_index(drop=True)
        print(f"Dropped them -> {len(df):,} rows remain")

    suffix = f"{args.level}_{args.target}" + ("_noextremes" if args.drop_extremes else "")
    runner = run_brand if args.level == "brand" else run_composition
    results, oof_df, gbm_bundle, ridge_bundle = runner(df, feature_cols, args.target, suffix)

    REPORT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    results.round(4).to_csv(REPORT_DIR / f"results_{suffix}.csv", index=False)
    oof_df.to_csv(REPORT_DIR / f"oof_{suffix}.csv", index=False)

    model_type = "lightgbm" if _LGBM_OK else "sklearn_hist_gb"
    meta = {"target": args.target, "level": args.level, "model_type": model_type,
            "drop_extremes": args.drop_extremes, "cv_results": results}
    joblib.dump({**gbm_bundle, **meta}, MODEL_DIR / f"gbm_{suffix}.pkl")
    if ridge_bundle is not None:
        joblib.dump({**ridge_bundle, **meta}, MODEL_DIR / f"ridge_{suffix}.pkl")

    # Plain-language summary
    view = results if args.level == "brand" else results.query("subset == 'all'")
    show_cols = [c for c in ["model", "split", "subset", "n", "MAE", "RMSE", "R2",
                             "R2_weighted", "Spearman"] if c in results.columns]
    print("\nResults (grouped = honest estimate on unseen medicines):")
    print(results[show_cols].round(3).to_string(index=False))

    grp = view.query("split == 'grouped'")
    base, best = grp.query("model == 'baseline_mean'").iloc[0], grp.sort_values("MAE").iloc[0]
    print(f"\nBest model: {best['model']} -> MAE {best['MAE']:.2f} vs baseline "
          f"{base['MAE']:.2f} ({1 - best['MAE'] / base['MAE']:.1%} better), "
          f"R2 {best['R2']:.3f}, Spearman {best['Spearman']:.3f}")
    print(f"Tree model used: {model_type}")

    saved = [f"reports/results_{suffix}.csv", f"reports/oof_{suffix}.csv",
             f"models/gbm_{suffix}.pkl"] + ([f"models/ridge_{suffix}.pkl"] if ridge_bundle else [])
    print("Saved -> " + ", ".join(saved))


if __name__ == "__main__":
    main()