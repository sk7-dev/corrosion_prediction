import math
from pathlib import Path
from typing import Dict, List

import gradio as gr
import numpy as np
import pandas as pd
import xgboost as xgb

# ------------------ Load Model (XGBoost) ------------------
MODEL_JSON_PATH = Path(r"D:\CSULB\Research\Corrosion\Analysis\Model\xgb_model.json")
MODEL_JOBLIB_PATH = Path(r"D:\CSULB\Research\Corrosion\Analysis\Model\xgb_model.joblib")

model = None
if MODEL_JSON_PATH.exists():
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_JSON_PATH))
elif MODEL_JOBLIB_PATH.exists():
    from joblib import load
    model = load(str(MODEL_JOBLIB_PATH))
else:
    raise FileNotFoundError(
        f"Could not find model file. Checked:\n - {MODEL_JSON_PATH}\n - {MODEL_JOBLIB_PATH}"
    )

# ------------------ Feature List in Training Set (ORDER MATTERS) ------------------
features_in_order: List[str] = [
    'texture_Clay', 'texture_Clay Loam', 'texture_Loam', 'texture_Loamy Sand',
    'texture_Marsh', 'texture_Muck', 'texture_Peat', 'texture_Sand',
    'texture_Sandy Clay Loam', 'texture_Sandy Loam', 'texture_Silt Loam',
    'texture_Silty Clay', 'texture_Silty Clay Loam',
    'internal_drainage_code', 'pH', 'annual_precipitation',
    'log_resistivity', 'mean_temp',
    # Composition percentages (mandatory; can be zero) — keep this order
    'C', 'Si', 'Mn', 'S', 'P',
    'Cr', 'Ni', 'Cu', 'Mo',
    'log_days'
]

available_textures = [f.split("_", 1)[1] for f in features_in_order if f.startswith("texture_")]
composition_features = ['C', 'Si', 'Mn', 'S', 'P', 'Cr', 'Ni', 'Cu', 'Mo']
DRAINAGE_CHOICES = ["0 - Very Poor", "1 - Poor", "2 - Fair", "3 - Good"]

# ------------------ Core predict ------------------
def predict_once(
    texture: str,
    internal_drainage_label: str,
    pH: float,
    annual_precipitation: float,
    resistivity: float,
    mean_temp: float,
    days: float,
    C: float, Si: float, Mn: float, S: float, P: float, Cr: float, Ni: float, Cu: float, Mo: float
):
    try:
        internal_drainage_code = float(internal_drainage_label.split(" - ")[0])
    except Exception:
        internal_drainage_code = 2.0  # default to "Fair"
    """
    Build the feature vector in EXACT training order and run inference.
    return: (log_corrosion_rate, corrosion_rate)
    """
    # Build composition dict in the exact order
    composition_values: Dict[str, float] = {
        'C': C, 'Si': Si, 'Mn': Mn, 'S': S, 'P': P, 'Cr': Cr, 'Ni': Ni, 'Cu': Cu, 'Mo': Mo
    }

    # Optional sanity check (just a warning string shown below)
    total_pct = sum(composition_values.values())
    warning = ""
    if not (0 <= total_pct <= 100):
        warning = f"Warning: Composition adds to {total_pct:.2f}% (outside 0–100)."

    input_dict = {}
    for feature in features_in_order:
        if feature.startswith("texture_"):
            input_dict[feature] = 1.0 if feature == f"texture_{texture}" else 0.0
        elif feature == "internal_drainage_code":
            input_dict[feature] = float(internal_drainage_code)
        elif feature == "pH":
            input_dict[feature] = float(pH)
        elif feature == "annual_precipitation":
            input_dict[feature] = float(annual_precipitation)
        elif feature == "log_resistivity":
            input_dict[feature] = math.log(float(resistivity) + 1e-8)
        elif feature == "mean_temp":
            input_dict[feature] = float(mean_temp)
        elif feature in composition_features:
            input_dict[feature] = float(composition_values[feature])
        elif feature == "log_days":
            input_dict[feature] = math.log(float(days) + 1e-8)
        else:
            input_dict[feature] = 0.0

    input_df = pd.DataFrame([input_dict])[features_in_order]
    input_mat = input_df.astype(np.float32)

    log_corrosion_rate = float(model.predict(input_mat)[0])
    corrosion_rate = math.exp(log_corrosion_rate)
    # Return values + optional warning
    return round(log_corrosion_rate, 6), round(corrosion_rate, 6), warning

# ------------------ UI ------------------
with gr.Blocks(title="Corrosion Model") as demo:
    gr.Markdown("# Corrosion Rate Predictor")
    gr.Markdown(
        #"Provide inputs below. The app computes `log_resistivity = log(resistivity + 1e-8)` "
        #"and `log_days = log(days + 1e-8)` and preserves the exact feature order your model expects."
    )

    with gr.Row():
        texture_dd = gr.Dropdown(
            choices=available_textures, value=available_textures[0], label="Texture"
        )
        idc = gr.Dropdown(choices=DRAINAGE_CHOICES, value="2 - Fair", label="Internal Drainage Code")
        #ph = gr.Number(label="pH", value=7.0)
        ph = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=7.0, label="pH")
        precip = gr.Number(label="Annual Precipitation (Inches)", value=500.0)

    with gr.Row():
        resist = gr.Number(label="Soil Resistivity (at 60°F)", value=1000.0)
        mean_temp = gr.Number(label="Mean Temperature (°F)", value=15.0)
        days = gr.Number(label="Exposure Days", value=365.0)

    gr.Markdown("### Composition in % — values may be 0")
    with gr.Row():
        C = gr.Number(label="C", value=0.2)
        Si = gr.Number(label="Si", value=0.2)
        Mn = gr.Number(label="Mn", value=0.4)
        S = gr.Number(label="S", value=0.02)
        P = gr.Number(label="P", value=0.02)
    with gr.Row():
        Cr = gr.Number(label="Cr", value=0.0)
        Ni = gr.Number(label="Ni", value=0.0)
        Cu = gr.Number(label="Cu", value=0.0)
        Mo = gr.Number(label="Mo", value=0.0)

    btn = gr.Button("Predict", variant="primary")

    with gr.Row():
        out_log = gr.Number(label="Predicted log(corrosion rate)")
        out_rate = gr.Number(label="Predicted corrosion rate")
    warn = gr.Markdown()

    btn.click(
        predict_once,
        inputs=[texture_dd, idc, ph, precip, resist, mean_temp, days, C, Si, Mn, S, P, Cr, Ni, Cu, Mo],
        outputs=[out_log, out_rate, warn]
    )

if __name__ == "__main__":
    demo.launch()  # add share=True for a temporary public URL
