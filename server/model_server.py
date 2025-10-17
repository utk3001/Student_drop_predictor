import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.utils import resample

MODEL_PATH = "./assets/model.pkl"
SCALER_PATH = "./assets/scaler.pkl"
MODEL_FE_PATH = "./assets/model_fe.pkl"
SCALER_FE_PATH = "./assets/scaler_fe.pkl"
DATA_PATH = "./assets/original_data.pkl"

app = Flask(__name__)

def load_obj(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

model = load_obj(MODEL_PATH)
scaler = load_obj(SCALER_PATH)
model_fe = load_obj(MODEL_FE_PATH)
scaler_fe = load_obj(SCALER_FE_PATH)
data_for_metrics = load_obj(DATA_PATH)  

if model is None or scaler is None:
    raise RuntimeError("Please save and place model.pkl and scaler.pkl next to this script.")

def explain_linear(m, feature_names, top_k=6):
    coefs = np.asarray(m.coef_).ravel()
    idx = np.argsort(np.abs(coefs))[::-1][:top_k]
    return [{"feature": feature_names[i], "coef": float(coefs[i])} for i in idx]

@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json()
    features = body.get("features")
    which = body.get("model", "baseline")
    if features is None:
        return jsonify({"error": "No features provided"}), 400
    
    X_row = pd.DataFrame([features])
    feat_cols_path = "feature_columns.pkl"
    if os.path.exists(feat_cols_path):
        import pickle
        with open(feat_cols_path, "rb") as f:
            feat_cols = pickle.load(f)
        X_row = X_row.reindex(columns=feat_cols, fill_value=0)

    if which == "mitigated":
        X_scaled = scaler_fe.transform(X_row)
        prob = model_fe.predict_proba(X_scaled)[0, 1]
        pred = int(model_fe.predict(X_scaled)[0])
        explanation = explain_linear(model_fe, feat_cols)
    else:
        X_scaled = scaler.transform(X_row)
        prob = model.predict_proba(X_scaled)[0, 1]
        pred = int(model.predict(X_scaled)[0])
        explanation = explain_linear(model, feat_cols)

    return jsonify({
        "prediction": pred,
        "probability_dropout": float(prob),
        "explanation": explanation,
        "model": which
    })

@app.route("/metrics", methods=["GET"])
def metrics():
    which = request.args.get("model", "baseline")
    group_col = request.args.get("group", None)

    if data_for_metrics is None:
        return jsonify({"error": "No metrics data available. Save a sample data_for_metrics.pkl"}), 400

    df = data_for_metrics.copy()
    feat_cols_path = "feature_columns.pkl"
    with open(feat_cols_path, "rb") as f:
        feat_cols = pickle.load(f)
    X_all = df[feat_cols].reindex(columns=feat_cols, fill_value=0)
    y_all = df["Target"]

    if which == "mitigated":
        X_scaled = scaler_fe.transform(X_all)
        preds = model_fe.predict(X_scaled)
    else:
        X_scaled = scaler.transform(X_all)
        preds = model.predict(X_scaled)

    overall = {}
    overall["accuracy"] = float(accuracy_score(y_all, preds))
    overall["precision"] = float(precision_score(y_all, preds, zero_division=0))
    overall["recall"] = float(recall_score(y_all, preds, zero_division=0))
    cm = confusion_matrix(y_all, preds).tolist()

    group_results = []
    spd = None
    eod = None
    if group_col and group_col in df.columns:
        groups = df[group_col].astype(str).unique()
        sel_rates = []
        recalls = []
        for g in groups:
            mask = df[group_col].astype(str) == g
            if mask.sum() == 0:
                continue
            y_g = y_all[mask]
            Xg = X_all[mask]
            if which == "mitigated":
                yhat = model_fe.predict(scaler_fe.transform(Xg))
            else:
                yhat = model.predict(scaler.transform(Xg))
            acc = accuracy_score(y_g, yhat)
            rec = recall_score(y_g, yhat, zero_division=0)
            prec = precision_score(y_g, yhat, zero_division=0)
            sel = float(np.mean(yhat))
            group_results.append({
                "group": g,
                "size": int(mask.sum()),
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "selection_rate": sel
            })
            sel_rates.append(sel)
            recalls.append(rec)
        if len(sel_rates) > 0:
            spd = float(max(sel_rates) - min(sel_rates))
        if len(recalls) > 0:
            eod = float(max(recalls) - min(recalls))

    return jsonify({
        "model": which,
        "overall": overall,
        "confusion_matrix": cm,
        "group_metrics": group_results,
        "SPD": spd,
        "EOD": eod
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
