import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import forecast_utils as fu

st.set_page_config(page_title="Market Signals — Sales Forecast", page_icon="📈", layout="wide")
OUT = fu.BASE / "outputs"
SMALL_SALES, HIGH_MAPE = 10_000, 25     # warn when a category is small / historically hard to forecast


# ---------- cached loaders ----------
@st.cache_data
def load_data():
    data = fu.load_clean()
    feats = fu.build_features(data)
    return data, feats

@st.cache_resource
def load_bundle():
    try:
        return joblib.load(OUT / "app_model.joblib")
    except Exception:
        # Model file missing or saved with a different scikit-learn version -> rebuild it here (one-time, ~1 min)
        import runpy
        with st.spinner("Rebuilding the model for this environment (one-time)..."):
            runpy.run_path(str(fu.BASE / "prepare_app_assets.py"))
        return joblib.load(OUT / "app_model.joblib")

@st.cache_data
def load_csv(name, **kw):
    return pd.read_csv(OUT / name, **kw)

@st.cache_data
def make_forecasts():
    _, feats = load_data()
    b = load_bundle()
    rows = fu.forecast_rows(feats, b["feature_columns"])
    rows["forecast"] = b["model"].predict(rows[b["feature_columns"]]).clip(min=0)
    return rows[["category", "dc_total_sales", "forecast"]].rename(columns={"dc_total_sales": "last_actual"})


data, feats = load_data()
bundle = load_bundle()
forecasts = make_forecasts()
errors = load_csv("category_errors.csv").set_index("category")
test_pred = load_csv("test_predictions.csv", parse_dates=["target_month"])

# ---------- sidebar ----------
st.sidebar.title("📈 Market Signals")
page = st.sidebar.radio("Page", ["Forecast", "Category overview", "Model performance", "Data notes (عربي)"])
st.sidebar.caption(f"Data: {data.year_month.min():%Y-%m} → {fu.LAST_COMPLETE_MONTH:%Y-%m}  \n"
                   f"Forecast month: **{fu.FORECAST_MONTH:%Y-%m}**  \n"
                   f"Model: {bundle['name']}")


def fmt(x):
    return f"{x:,.0f}"


# =========================================================
if page == "Forecast":
    st.title("Next-month sales forecast")
    st.caption(f"Predicts total sales for **{fu.FORECAST_MONTH:%B %Y}** — the month right after the last complete month in the data. "
               "It is not a live forecast of today.")

    cats = sorted(data.category.unique())
    cat = st.selectbox("Category", cats, index=cats.index("Fishing") if "Fishing" in cats else 0)

    hist = data[data.category == cat].set_index("year_month").dc_total_sales
    f = forecasts[forecasts.category == cat]
    err = errors.loc[cat] if cat in errors.index else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Last actual ({fu.LAST_COMPLETE_MONTH:%Y-%m})", fmt(hist.iloc[-1]) if hist.index[-1] == fu.LAST_COMPLETE_MONTH else "n/a")
    if len(f):
        pred, last = f.forecast.iloc[0], f.last_actual.iloc[0]
        c2.metric(f"Forecast ({fu.FORECAST_MONTH:%Y-%m})", fmt(pred), f"{(pred - last) / last:+.1%} vs last month")
    else:
        c2.metric("Forecast", "n/a")
    if err is not None:
        c3.metric("Typical error (backtest)", f"±{err.MAPE_pct:.0f}%")
        c4.metric("Backtest MAE", fmt(err.MAE))

    if not len(f):
        st.warning("No forecast: this category does not have 3 consecutive months of data ending in "
                   f"{fu.LAST_COMPLETE_MONTH:%Y-%m}, which the model needs.")
    elif err is not None and (err.MAPE_pct > HIGH_MAPE or hist.mean() < SMALL_SALES):
        st.warning(f"Low-confidence forecast: this is a small category (avg monthly sales ≈ {fmt(hist.mean())}) "
                   f"and the backtest error was about {err.MAPE_pct:.0f}%. Treat the number as indicative only.")
    else:
        st.success("Reasonable-confidence forecast: this category's backtest error was small.")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hist.index, hist.values, "o-", label="Actual", color="#1f4e79")
    bt = test_pred[test_pred.category == cat].sort_values("target_month")
    if len(bt):
        ax.plot(bt.target_month, bt.predicted, "s--", label="Backtest prediction (train-only model)", color="#d98c00")
    if len(f):
        ax.plot([fu.LAST_COMPLETE_MONTH, fu.FORECAST_MONTH], [f.last_actual.iloc[0], f.forecast.iloc[0]], ":", color="#c0392b")
        ax.scatter([fu.FORECAST_MONTH], [f.forecast.iloc[0]], s=90, color="#c0392b", zorder=5, label="Forecast")
    ax.set_title(cat); ax.set_ylabel("Total sales"); ax.legend(); ax.grid(alpha=.25)
    st.pyplot(fig); plt.close(fig)

    with st.expander("How is this forecast made?"):
        st.write("The model uses this category's sales, orders, discount, late rate, profit and benefit for the last three months, "
                 "plus the 3-month average of sales and the month of the year. It was trained on all available history. "
                 "The backtest line comes from a separate model trained only on data before 2017-03, so it is an honest test.")

# =========================================================
elif page == "Category overview":
    st.title("Category overview")
    ov = (data.groupby("category").agg(avg_monthly_sales=("dc_total_sales", "mean"), months=("year_month", "count"))
          .join(errors[["MAPE_pct"]], how="left")
          .join(forecasts.set_index("category")[["last_actual", "forecast"]], how="left")
          .sort_values("avg_monthly_sales", ascending=False).reset_index())
    ov["forecast_change_pct"] = (ov.forecast / ov.last_actual - 1) * 100
    ov["confidence"] = ov.apply(lambda r: "low" if (pd.notna(r.MAPE_pct) and r.MAPE_pct > HIGH_MAPE) or r.avg_monthly_sales < SMALL_SALES else "ok", axis=1)
    st.dataframe(ov.round(1).rename(columns={"MAPE_pct": "backtest_error_%", "forecast": f"forecast_{fu.FORECAST_MONTH:%Y-%m}",
                                             "last_actual": f"actual_{fu.LAST_COMPLETE_MONTH:%Y-%m}"}),
                 use_container_width=True, hide_index=True)
    top = ov.dropna(subset=["forecast"]).head(10)
    fig, ax = plt.subplots(figsize=(11, 4))
    x = range(len(top))
    ax.bar([i - .2 for i in x], top.last_actual, width=.4, label="Last actual", color="#1f4e79")
    ax.bar([i + .2 for i in x], top.forecast, width=.4, label="Forecast", color="#d98c00")
    ax.set_xticks(list(x)); ax.set_xticklabels(top.category, rotation=30, ha="right"); ax.legend()
    ax.set_title("Top categories: last actual vs forecast"); st.pyplot(fig); plt.close(fig)
    st.caption("Categories with `confidence = low` are small and had a large backtest error (40–90%).")

# =========================================================
elif page == "Model performance":
    st.title("Model performance")
    st.caption("Test period: 2017-03 → 2017-09 (109 rows). Lower MAE/RMSE is better.")
    sales = load_csv("model_comparison_sales.csv")
    st.subheader("Sales forecast — model comparison")
    st.dataframe(sales, use_container_width=True, hide_index=True)
    base = sales[sales.Model == "Baseline: last month"].MAE.iloc[0]
    best = sales[~sales.Model.str.startswith("Baseline")].iloc[0]
    st.info(f"Best ML model: **{best.Model}** — MAE {best.MAE:,.0f} vs {base:,.0f} for the “last month” baseline "
            f"(≈{(1 - best.MAE / base):.0%} better). Sales are almost flat over time, so the baseline is already strong "
            "and the gain is small and within the noise of a 109-row test.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Actual vs predicted")
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.scatter(test_pred.actual, test_pred.predicted, alpha=.6)
        m = max(test_pred.actual.max(), test_pred.predicted.max()) * 1.05
        ax.plot([0, m], [0, m], "r--"); ax.set_xlim(0, m); ax.set_ylim(0, m)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); st.pyplot(fig); plt.close(fig)
    with col2:
        st.subheader("Feature importance")
        imp = load_csv("feature_importance.csv").head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.barh(imp.feature, imp.mae_increase, color="#1f4e79"); ax.set_xlabel("Increase in MAE when shuffled")
        st.pyplot(fig); plt.close(fig)
    st.caption("Permutation importance on the test set. Recent sales dominate; discount, late rate and profit add almost nothing.")

    st.subheader("Late-rate forecast (negative result)")
    late = load_csv("model_comparison_late_rate.csv")
    st.dataframe(late, use_container_width=True, hide_index=True)
    st.warning("R² is ≤ 0 for every model: no model beats simply predicting the average late rate (~0.55). "
               "Late rate is not predictable from the signals in this file, so the app does not offer a late-rate forecast.")

# =========================================================
else:
    st.title("Data notes — ملاحظات على الداتا")
    st.markdown("""
<div dir="rtl" style="text-align:right">

### إيه اللي اتعمل في الداتا قبل التدريب
- **استبعاد صفوف الـ Logs (30 صف):** فئاتها (apparel وfan shop وgolf...) مختلفة عن فئات DataCo، والـ outer join ملأ مبيعاتها بصفر.
- **استبعاد الشهور بعد 2017-09:** البيانات ناقصة (مثلاً Fishing نزلت من حوالي 490 أوردر لـ 16)، وده انهيار وهمي وليس سلوك سوق.
- **الإبقاء على 24 فئة** ليها 24 شهر أو أكتر من التاريخ.
- **استبعاد `conversion_rate`** (كله صفر) وأعمدة `lg_*` و`high_*` من الـ features.

### ليه مفيش تصنيف (Classification)؟
- `high_interest_low_sales`: كل الحالات الإيجابية (30) هي صفوف الـ Logs، وفترة التدريب مفيهاش ولا حالة.
- `high_late_risk`: قيمته 1 في 95% من الصفوف، والـ 0 أغلبه صفوف الـ Logs (late_rate=0 بسبب `fillna(0)`). أي دقة عالية فيه هي مجرد نسبة الأغلبية.

### حدود التطبيق
- المبيعات شبه ثابتة، فالـ baseline قوي والتحسن من الـ ML صغير.
- الفئات الصغيرة خطأها 40–90%، فالتوقع فيها إرشادي فقط.
- اختبار الأداء على 109 صف فقط (2017-03 → 2017-09).
- التوقع للشهر **{month}** مش لـ "الشهر الجاي الحقيقي"، لأن آخر بيانات مكتملة هي 2017-09.
- لازم mapping صحيح بين فئات الـ Logs وDataCo قبل استخدام بيانات الزيارات.

</div>
""".replace("{month}", f"{fu.FORECAST_MONTH:%Y-%m}"), unsafe_allow_html=True)
