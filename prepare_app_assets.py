"""Run once: builds everything the Streamlit app reads from outputs/ (metrics, predictions, final model)."""
import warnings, joblib
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import forecast_utils as fu

warnings.filterwarnings("ignore")
OUT = fu.BASE / "outputs"; OUT.mkdir(exist_ok=True)
RS = 42

data = fu.load_clean()
feats = fu.build_features(data)
cols = fu.feature_columns(feats)
train_all = fu.training_rows(feats)
num = [c for c in cols if c != "category"]

def preproc():
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["category"])])

dates = np.array(sorted(train_all.year_month.unique()))
split = dates[int(len(dates) * 0.8)]
tr, te = train_all[train_all.year_month < split], train_all[train_all.year_month >= split]
print("split", pd.Timestamp(split).date(), len(tr), len(te))

def metrics(name, y, p):
    return {"Model": name, "MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** .5, "R2": r2_score(y, p)}

TUNED_RF = dict(n_estimators=400, min_samples_leaf=1, max_features=0.3, max_depth=None, random_state=RS, n_jobs=-1)
def models():
    return {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=10.0),
            "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=RS, n_jobs=-1),
            "Random Forest (tuned)": RandomForestRegressor(**TUNED_RF),
            "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=150, random_state=RS)}

def run(target, cur, extra=None):
    rows = [metrics("Baseline: last month", te[target], te[cur])]
    if extra: rows += [metrics(n, te[target], te[c]) for n, c in extra.items()]
    fitted = {}
    for n, m in models().items():
        p = Pipeline([("preprocessing", preproc()), ("model", m)]).fit(tr[cols], tr[target])
        fitted[n] = p; rows.append(metrics(n, te[target], p.predict(te[cols])))
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True), fitted

sales_tbl, sales_fit = run("target_next_sales", "dc_total_sales", {"Baseline: avg last 3 months": "sales_roll3"})
late_tbl, _ = run("target_next_late_rate", "dc_late_rate")
sales_tbl.round(4).to_csv(OUT / "model_comparison_sales.csv", index=False)
late_tbl.round(4).to_csv(OUT / "model_comparison_late_rate.csv", index=False)

# test-period predictions (train-only model) for backtests/plots
best = sales_fit["Random Forest (tuned)"]
tp = te[["category", "year_month", "target_next_sales"]].copy()
tp["target_month"] = tp.year_month + pd.offsets.MonthBegin(1)
tp["predicted"] = np.clip(best.predict(te[cols]), 0, None)
tp["baseline_last_month"] = te["dc_total_sales"].values
tp.rename(columns={"target_next_sales": "actual"}).drop(columns="year_month").to_csv(OUT / "test_predictions.csv", index=False)

# error by category
ce = tp.groupby("category").apply(lambda g: pd.Series({
    "test_months": len(g), "avg_actual_sales": g.target_next_sales.mean(),
    "MAE": mean_absolute_error(g.target_next_sales, g.predicted),
    "MAPE_pct": float(np.mean(np.abs(g.target_next_sales - g.predicted) / g.target_next_sales) * 100)})).reset_index()
ce.round(2).to_csv(OUT / "category_errors.csv", index=False)

# permutation importance
pi = permutation_importance(best, te[cols], te["target_next_sales"], n_repeats=15, random_state=RS,
                            scoring="neg_mean_absolute_error")
pd.Series(pi.importances_mean, index=cols, name="mae_increase").sort_values(ascending=False).round(2)\
  .to_csv(OUT / "feature_importance.csv", header=True, index_label="feature")

# FINAL model: same tuned RF but trained on ALL available rows (used only for the app forecast)
final = Pipeline([("preprocessing", preproc()), ("model", RandomForestRegressor(**TUNED_RF))])
final.fit(train_all[cols], train_all["target_next_sales"])
joblib.dump({"model": final, "feature_columns": cols, "name": "Random Forest (tuned), trained on all rows",
             "trained_until": str(train_all.year_month.max().date()),
             "forecast_month": str(fu.FORECAST_MONTH.date())}, OUT / "app_model.joblib")
print("assets saved"); print(sales_tbl.round(1).to_string(index=False))
