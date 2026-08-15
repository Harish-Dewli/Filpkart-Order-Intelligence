from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "orders_dataset.csv")
X = df.drop(columns=["returned"]); y = df["returned"]
cat = ["product_category", "payment_method"]
num = [c for c in X.columns if c not in cat]

pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Baseline
baseline = Pipeline([("pre", pre), ("model", DummyClassifier(strategy="most_frequent"))])
baseline.fit(X_train, y_train)
bpred = baseline.predict(X_test)

# Logistic regression
lr = Pipeline([("pre", pre), ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))])
lr.fit(X_train, y_train)
lrp = lr.predict_proba(X_test)[:, 1]

def metrics_at(y_true, p, t):
    pred = (p >= t).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, p),
    }

thresholds = np.arange(0.10, 0.901, 0.01)
lr_sweep = []
for t in thresholds:
    m = metrics_at(y_test, lrp, t); m["threshold"] = float(t); lr_sweep.append(m)
lr_best = max(lr_sweep, key=lambda x: x["f1"])

# Random forest grid search
rf = Pipeline([("pre", pre), ("model", RandomForestClassifier(
    class_weight="balanced", random_state=42, n_jobs=-1
))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    rf,
    {"model__n_estimators": [100, 200], "model__max_depth": [6, 10, None]},
    scoring="roc_auc", cv=cv, n_jobs=-1, refit=True,
)
grid.fit(X_train, y_train)
rf_best = grid.best_estimator_
rfp = rf_best.predict_proba(X_test)[:, 1]
rf_sweep = []
for t in thresholds:
    m = metrics_at(y_test, rfp, t); m["threshold"] = float(t); rf_sweep.append(m)
rf_best_threshold = max(rf_sweep, key=lambda x: x["f1"])

# Feature importance
fitted_pre = rf_best.named_steps["pre"]
fitted_rf = rf_best.named_steps["model"]
feature_names = fitted_pre.get_feature_names_out()
imp = fitted_rf.feature_importances_
idx = np.argsort(imp)[::-1]
top5 = [{"feature": feature_names[i], "importance": float(imp[i])} for i in idx[:5]]

perm = permutation_importance(rf_best, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1)
perm_rows = sorted(
    [{"feature": c, "importance_mean": float(v)} for c, v in zip(X.columns, perm.importances_mean)],
    key=lambda x: x["importance_mean"], reverse=True
)

# Subgroup metrics at the RF F1-optimal threshold.
rf_pred = (rfp >= rf_best_threshold["threshold"]).astype(int)
subgroups = {}
for col in ["product_category", "payment_method"]:
    subgroups[col] = []
    for value, ix in X_test.groupby(col).groups.items():
        yt = y_test.loc[ix]; yp = rf_pred[X_test.index.get_indexer(ix)]
        subgroups[col].append({
            "group": value, "n": int(len(ix)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
        })

joblib.dump(rf_best, MODEL_DIR / "return_risk_model.pkl")
(MODEL_DIR / "return_risk_threshold.json").write_text(json.dumps({"t_rf": rf_best_threshold["threshold"]}, indent=2))

report = {
    "rows": int(len(df)), "columns": int(df.shape[1]),
    "return_rate": float(y.mean()), "rating_missing_rate": float(df["rating_given"].isna().mean()),
    "rating_missing_rate_by_payment": df.groupby("payment_method")["rating_given"].apply(lambda s: float(s.isna().mean())).to_dict(),
    "baseline": {"accuracy": float(accuracy_score(y_test,bpred)), "f1_class_1": float(f1_score(y_test,bpred,zero_division=0))},
    "logistic_default_0.5": metrics_at(y_test, lrp, 0.5),
    "logistic_best_threshold": lr_best,
    "logistic_sweep": lr_sweep,
    "rf_best_params": grid.best_params_, "rf_best_cv_auc": float(grid.best_score_),
    "rf_test_auc": float(roc_auc_score(y_test, rfp)), "rf_best_threshold": rf_best_threshold,
    "rf_sweep": rf_sweep, "top5_impurity_features": top5, "permutation_importance": perm_rows,
    "subgroups": subgroups,
}
(OUT_DIR / "part1_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ["rows","columns","return_rate","rating_missing_rate","baseline","logistic_default_0.5","logistic_best_threshold","rf_best_params","rf_best_cv_auc","rf_test_auc","rf_best_threshold","top5_impurity_features"]}, indent=2))
