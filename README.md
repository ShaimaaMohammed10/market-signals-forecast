# Market Signals — ML Project

توقع مبيعات الشهر القادم ونسبة التأخير لكل Category من `market_signals.csv`.

## الملفات
| الملف | الوصف |
|---|---|
| `Market_Signals_ML_fixed.ipynb` | الـ notebook الكامل (Cleaning → Features → Models → Tuning → Importance → Save) |
| `market_signals.csv` | الداتا (883 صف × 16 عمود) |
| `README_data_market_signals.md` | وصف أعمدة الداتا |
| `outputs/project_summary.json` | ملخص النتايج |
| `outputs/best_sales_model.joblib` | أحسن موديل مبيعات (Pipeline كامل) |
| `app.py` | تطبيق Streamlit (4 صفحات: Forecast / Category overview / Model performance / Data notes) |
| `forecast_utils.py` | دوال تنضيف الداتا والـ features (مشتركة بين التطبيق والسكريبت) |
| `prepare_app_assets.py` | سكريبت بيبني ملفات التطبيق في `outputs/` (النتايج والـ predictions والموديل النهائي) |
| `outputs/app_model.joblib` | الموديل اللي التطبيق بيستخدمه (Random Forest بعد الـ tuning، متدرّب على كل الداتا) |
| `requirements.txt` | المكتبات |

## التشغيل
```
pip install -r requirements.txt
jupyter notebook Market_Signals_ML_fixed.ipynb
```

## تشغيل تطبيق Streamlit
```
pip install -r requirements.txt
streamlit run app.py
```
(شغّلي الأمر من جوه الفولدر. لو `outputs/` اتمسح، شغّلي `python prepare_app_assets.py` الأول.)
التطبيق بيتوقع مبيعات **2017-10** لأن آخر بيانات مكتملة هي 2017-09، ومش بيعرض توقع للتأخير لأنه مش قابل للتنبؤ.

## استخدام الموديل المحفوظ
```python
import joblib
bundle = joblib.load("outputs/best_sales_model.joblib")
pred = bundle["model"].predict(new_df[bundle["feature_columns"]])
```
`new_df` لازم يكون فيه نفس أعمدة الـ features (قيم الشهر الحالي + lag1 + lag2 + sales_roll3 + month_sin/cos + category).

## النتايج (Test: 2017-03 → 2017-09)
| المهمة | أحسن موديل | MAE | R² | Baseline (الشهر اللي فات) MAE |
|---|---|---|---|---|
| مبيعات الشهر الجاي | Random Forest (tuned) | ~3.6k (3619 في الـ notebook، 3659 في سكريبت التطبيق؛ الفرق من الـ randomness) | 0.993 | 3816 |
| نسبة التأخير | Random Forest | 0.057 | -0.03 | 0.074 |

## قرارات تنضيف الداتا
- استبعاد صفوف الـ Logs (30 صف): مبيعاتها صفر بحكم الـ outer join ولا تتطابق مع فئات DataCo.
- استبعاد الشهور بعد 2017-09: بيانات ناقصة (مثلاً Fishing نزلت من ~490 أوردر لـ 16).
- الإبقاء على الفئات اللي ليها 24 شهر أو أكتر (24 فئة).
- استبعاد `conversion_rate` (كله صفر) وأعمدة `lg_*` و`high_*`.
- حذف مهام التصنيف (`high_late_risk`, `high_interest_low_sales`): الـ labels ناتجة عن `fillna(0)` بعد الـ outer join (التفاصيل في قسم 9 بالـ notebook).

## حدود المشروع
- المبيعات شبه ثابتة، فالـ baseline قوي والتحسين من الـ ML صغير (حوالي 5%).
- نسبة التأخير مش قابلة للتنبؤ بالمتغيرات المتاحة (R² سالب).
- الاختبار على 109 صف فقط؛ الفروق الصغيرة بين الموديلات في حدود الضوضاء.
- الموديل المحفوظ اتختار بناءً على الـ test، فيفضل إعادة تدريبه على الداتا كلها قبل الاستخدام الفعلي.
