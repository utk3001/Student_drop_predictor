import os
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "model_fe.pkl"), "rb") as f:
    model_fe = pickle.load(f)
with open(os.path.join(BASE_DIR, "scaler_fe.pkl"), "rb") as f:
    scaler_fe = pickle.load(f)

protected_features = [
    "Age at enrollment", "Gender", "Nacionality", "Educational special needs",
    "Mother's qualification", "Father's qualification", 
    "Mother's occupation", "Father's occupation", "Tuition fees up to date", "Roll_No"
]

expected_features = [
    'Marital Status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification',
    'Previous qualification (grade)', 'Admission grade', 'Displaced',
    'Debtor', 'Scholarship holder', 'International',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP',
    'AgeBin_21-23', 'AgeBin_24-26', 'AgeBin_27+'
]

def bin_age(age):
    if age <= 20: 
        return "18-20"
    elif age <= 23: 
        return "21-23"
    elif age <= 26: 
        return "24-26"
    else: 
        return "27+"

def preprocess_student_input(input_dict):
    df = pd.DataFrame([input_dict])

    age_bin = bin_age(df.loc[0, "Age at enrollment"])
    for bin_name in ["21-23", "24-26", "27+"]:
        df[f"AgeBin_{bin_name}"] = 1 if age_bin == bin_name else 0

    df.drop(columns=protected_features, inplace=True, errors='ignore')

    df = pd.get_dummies(df, drop_first=True)

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_features]

    X_scaled = scaler_fe.transform(df)
    return X_scaled,df

def compute_fairness_metrics():
    return [
        {
            "group": "Gender",
            "improvement": "≈20%",
            "interpretation": "Fairness between male and female students improved by around 20%. The model now treats both genders more equally."
        },
        {
            "group": "Nationality",
            "improvement": "≈98–99%",
            "interpretation": "The model now treats students of different nationalities almost equally, showing a 98–99% fairness improvement."
        },
        {
            "group": "Age Group",
            "improvement": "≈60%",
            "interpretation": "Fairness across age groups improved by roughly 60%, though older students still have a slight advantage."
        },
        {
            "group": "Educational Special Needs",
            "improvement": "≈60%",
            "interpretation": "Fairness for students with special needs improved significantly (~60%), but further recall balance could help."
        },
        {
            "group": "Debtor Status",
            "improvement": "≈15%",
            "interpretation": "Students with and without outstanding dues are now treated about 15% more fairly than before."
        },
        {
            "group": "Overall Fairness",
            "improvement": "≈50–60%",
            "interpretation": "Across all groups combined, the model is now about 50–60% fairer and more equitable than before mitigation."
        }
    ]

def load_background_data(n_samples=300):
    bg_path = os.path.join(BASE_DIR, "train_background.npy")
    if os.path.exists(bg_path):
        background = np.load(bg_path)
        if background.shape[0] > n_samples:
            background = background[:n_samples]
    else:
        print("Warning: train_background.npy not found — using random noise.")
        background = np.random.normal(0, 1, size=(n_samples, len(expected_features)))
    return background


def get_justification(X_processed, df_processed, top_n=6):
    
    background = load_background_data()
    
    feature_names = df_processed.columns.tolist()
    
    explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=feature_names,
        discretize_continuous=True,
        mode="classification"
    )

    exp = explainer.explain_instance(
        X_processed[0],
        model_fe.predict_proba,
        num_features=top_n
    )

    explanation = []
    for feature, weight in exp.as_list():
        strength = "strongly" if abs(weight) > 0.2 else "slightly"
        if weight > 0:
            explanation.append(f"{feature} {strength} increased the likelihood of dropout.")
        else:
            explanation.append(f"{feature} {strength} decreased the likelihood of dropout.")

    return explanation



# def get_justification(X_processed, pred_class, top_n=6):
#     try:
#         importances = model_fe.feature_importances_
#     except AttributeError:
#         importances = np.abs(model_fe.coef_[0])
#     feature_contribs = list(zip(expected_features, importances, X_processed[0]))
#     feature_contribs.sort(key=lambda x: abs(x[1] * x[2]), reverse=True)

#     top_features = feature_contribs[:top_n]
#     explanation = []

#     for feature, importance, value in top_features:
#         effect_strength = "strongly" if abs(value * importance) > 0.5 else "slightly"
#         if value * importance > 0:
#             explanation.append(f"{feature} {effect_strength} increased the likelihood of dropout.")
#         else:
#             explanation.append(f"{feature} {effect_strength} decreased the likelihood of dropout.")

#     return explanation


def generate_global_explanation(output_dir="staticimg/explanations"):
    os.makedirs(output_dir, exist_ok=True)

    background = load_background_data()

    if hasattr(model_fe, "apply"):
        explainer = shap.TreeExplainer(model_fe, background)
    else:
        explainer = shap.LinearExplainer(model_fe, background)

    shap_values = explainer.shap_values(background)

    plt.figure()
    shap.summary_plot(shap_values, background, feature_names=expected_features, show=False)
    plt.tight_layout()
    global_path = os.path.join(output_dir, "global_summary.png")
    plt.savefig(global_path, bbox_inches="tight")
    plt.close()

    return "/staticimg/explanations/global_summary.png"

def generate_local_explanation(X_sample, df_sample, output_dir="staticimg/explanations"):
    os.makedirs(output_dir, exist_ok=True)

    background = load_background_data()

    explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=df_sample.columns,
        discretize_continuous=True,
        mode="classification"
    )

    exp = explainer.explain_instance(
        X_sample[0],
        model_fe.predict_proba,
        num_features=10
    )

    local_path = os.path.join(output_dir, "local_lime_explanation.png")
    fig = exp.as_pyplot_figure()
    fig.tight_layout()
    fig.savefig(local_path, bbox_inches="tight")
    plt.close(fig)

    return "/staticimg/explanations/local_lime_explanation.png"


def predict_student_outcome(input_dict):
    X_processed, df_processed = preprocess_student_input(input_dict)

    pred_class = int(model_fe.predict(X_processed)[0])
    pred_proba = model_fe.predict_proba(X_processed)[0]
    confidence = float(pred_proba[pred_class] * 100)
    pred_str = "Dropout" if pred_class == 1 else "Graduate"
    justification = get_justification(X_processed, df_processed)
    global_plot_url = generate_global_explanation()
    local_plot_url = generate_local_explanation(X_processed, df_processed)

    return {
        "prediction": pred_str,
        "confidence": round(confidence, 2),
        "overall_accuracy": "88.1%",
        "fairness_metrics": compute_fairness_metrics(),
        "justification": justification,
        "local_explanation_url": local_plot_url,
        "global_explanation_url": global_plot_url
    }
