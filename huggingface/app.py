import gradio as gr
import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path

# ---------------- Rutas de archivos ----------------
MODEL_PATH = Path("rf_default.pkl")
COLS_PATH = Path("columnas_features.pkl")
PLACE_FREQ_PATH = Path("place_name_freq.pkl")
PROP_OHE_PATH = Path("property_type_ohe")      # sin .pkl (tal cual compartiste)
STATE_OHE_PATH = Path("state_name_ohe.pkl")

# ---------------- Carga robusta ----------------
def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def error_card(message: str) -> str:
    return f"""
<div style="
    background:#FFECEC;
    border:1px solid #FFB3B3;
    border-radius:14px;
    padding:18px;
    color:#7A0B0B;
    text-align:center;
    font-weight:700;
">
  ⚠️ {message}
</div>
"""

def result_card(formatted_ars: str) -> str:
    return f"""
<div style="
    background:#E9FFF8;
    border:1px solid #B6F3E2;
    border-radius:16px;
    padding:24px;
    text-align:center;
    box-shadow: 0 6px 18px rgba(10, 92, 79, 0.10);
">
  <div style="font-size:14px;color:#0A5C4F;opacity:0.85;margin-bottom:8px;">
    Precio estimado (ARS)
  </div>
  <div style="font-size:40px;font-weight:800;color:#0A5C4F;letter-spacing:0.5px;">
    {formatted_ars}
  </div>
  <div style="font-size:12px;color:#0A5C4F;opacity:0.7;margin-top:10px;">
    Estimación basada en los datos ingresados.
  </div>
</div>
"""

# Cargas
missing = [p.name for p in [MODEL_PATH, COLS_PATH, PLACE_FREQ_PATH, PROP_OHE_PATH, STATE_OHE_PATH] if not p.exists()]
if missing:
    raise FileNotFoundError(f"Faltan archivos: {', '.join(missing)}. Sube todos a la raíz del Space.")

# Modelo (asegurá scikit-learn==1.5.2 en requirements.txt)
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(
        f"No se pudo cargar rf_default.pkl. "
        f"Sugerencia: fija scikit-learn==1.5.2. Detalle: {type(e).__name__}: {e}"
    )

# Columnas finales esperadas por el modelo (orden exacto)
cols_expected = load_pickle(COLS_PATH)
if not isinstance(cols_expected, (list, pd.Index)):
    raise ValueError("columnas_features.pkl debe contener una lista/Index de columnas.")
cols_expected = list(cols_expected)

# OHEs para conocer categorías disponibles
prop_ohe = load_pickle(PROP_OHE_PATH)   # sklearn OneHotEncoder
state_ohe = load_pickle(STATE_OHE_PATH) # sklearn OneHotEncoder

prop_categories = list(map(str, prop_ohe.categories_[0]))
state_categories = list(map(str, state_ohe.categories_[0]))

# Deducción de columnas OHE presentes (drop='first' típico): base = categoría NO presente en columnas
prop_ohe_cols = [c for c in cols_expected if c.startswith("property_type_")]
state_ohe_cols = [c for c in cols_expected if c.startswith("state_name_")]

# Extraemos los "sufijos" reales que quedaron como columnas:
prop_columns_suffix = [c.replace("property_type_", "") for c in prop_ohe_cols]
state_columns_suffix = [c.replace("state_name_", "") for c in state_ohe_cols]

# Base implícita (categoría droppeda): la que está en categories pero NO en _columns_suffix
prop_base = [c for c in prop_categories if c not in prop_columns_suffix]
prop_base = prop_base[0] if prop_base else None
state_base = [c for c in state_categories if c not in state_columns_suffix]
state_base = state_base[0] if state_base else None

# place_name → frecuencia
place_freq_series = load_pickle(PLACE_FREQ_PATH)  # pandas Series index=place_name, values=freq
place_name_choices = list(map(str, place_freq_series.index))

# ---------------- Definición de inputs crudos (UI) ----------------
FEATURES = [
    {"name": "surface_total",   "label": "Superficie total (m²)",    "type": "number", "value": 60.0, "min": 10},
    {"name": "surface_covered", "label": "Superficie cubierta (m²)", "type": "number", "value": 50.0, "min": 0},
    {"name": "rooms",           "label": "Ambientes",                "type": "slider", "value": 2,    "min": 1, "max": 10, "step": 1},
    {"name": "bedrooms",        "label": "Dormitorios",              "type": "slider", "value": 1,    "min": 0, "max": 8,  "step": 1},
    {"name": "bathrooms",       "label": "Baños",                    "type": "slider", "value": 1,    "min": 0, "max": 5,  "step": 1},
    {"name": "property_type",   "label": "Tipo de propiedad",        "type": "dropdown",
     "choices": prop_categories, "value": prop_categories[0] if prop_categories else None},
    {"name": "state_name",      "label": "Zona/Provincia",           "type": "dropdown",
     "choices": state_categories, "value": state_categories[-1] if state_categories else None},
    {"name": "place_name",      "label": "Barrio / Localidad",       "type": "dropdown",
     "choices": place_name_choices, "value": "Palermo" if "Palermo" in place_name_choices else place_name_choices[0]},
]

# ---------------- Utilidades ----------------
def _format_ars(x: float) -> str:
    return f"$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def _validate_raw(args):
    row = {}
    for f, val in zip(FEATURES, args):
        t = f["type"]
        name = f["name"]
        label = f["label"]

        if val is None or (isinstance(val, str) and val.strip() == ""):
            return None, error_card(f"'{label}' no puede estar vacío.")

        if t in ("number", "slider"):
            try:
                num = float(val)
            except Exception:
                return None, error_card(f"'{label}' debe ser numérico.")
            if np.isnan(num):
                return None, error_card(f"'{label}' no puede estar vacío.")
            min_v = f.get("min", None)
            max_v = f.get("max", None)
            if min_v is not None and num < float(min_v):
                return None, error_card(f"'{label}' debe ser ≥ {min_v}.")
            if max_v is not None and num > float(max_v):
                return None, error_card(f"'{label}' debe ser ≤ {max_v}.")
            row[name] = num

        elif t == "dropdown":
            choices = f.get("choices", [])
            if str(val) not in list(map(str, choices)):
                return None, error_card(f"'{label}' inválido. Debe ser una de: {', '.join(map(str, choices))}.")
            row[name] = str(val)

        else:
            row[name] = val

    # Coherencia adicional
    if row["surface_covered"] > row["surface_total"]:
        return None, error_card("La superficie cubierta no puede ser mayor que la superficie total.")
    if row["bedrooms"] > row["rooms"]:
        return None, error_card("Los dormitorios no pueden superar la cantidad de ambientes.")
    if row["bathrooms"] > row["rooms"]:
        return None, error_card("Los baños no pueden superar la cantidad de ambientes.")

    return row, None

def _build_feature_row(raw: dict) -> pd.DataFrame:
    """
    Construye un DataFrame con UNA fila y columnas EXACTAS a cols_expected.
    Maneja OHE con base implícita y place_name_freq.
    """
    # Inicializa todas las columnas en 0
    data = {c: 0.0 for c in cols_expected}

    # 1) numéricas directas
    for k in ["surface_total", "surface_covered", "rooms", "bedrooms", "bathrooms"]:
        if k in data:
            data[k] = float(raw[k])

    # 2) property_type OHE con base implícita
    # Si la categoría elegida coincide con alguna columna property_type_* => set 1
    # Si no coincide (p.ej. la base 'Casa') => todas 0 (ya está)
    chosen_prop = str(raw["property_type"])
    col_name = f"property_type_{chosen_prop}"
    if col_name in data:
        data[col_name] = 1.0
    # Si no está, asumimos que es la base (ej 'Casa') y queda todo 0

    # 3) state_name OHE con base implícita
    chosen_state = str(raw["state_name"])
    col_state = f"state_name_{chosen_state}"
    if col_state in data:
        data[col_state] = 1.0
    # Si no está, es la base (p.ej. 'Bs.As. G.B.A. Zona Norte') => 0s

    # 4) place_name_freq
    # Nota: el modelo espera la columna 'place_name_freq'
    # Usamos el valor del mapa (si no está, 0.0)
    pname = str(raw["place_name"])
    freq_val = float(place_freq_series.get(pname, 0.0))
    if "place_name_freq" in data:
        data["place_name_freq"] = freq_val

    # Ordenar y devolver DataFrame 1xN
    X = pd.DataFrame([data], columns=cols_expected)
    return X

def predict_price(*args):
    # Validación + coerción
    raw, err = _validate_raw(args)
    if err is not None:
        return err

    # Construcción de features finales
    try:
        X = _build_feature_row(raw)
    except Exception as e:
        return error_card(f"No se pudo construir el vector de entrada: {type(e).__name__}: {e}")

    # Predicción
    try:
        y_pred = model.predict(X)[0]
    except Exception as e:
        # Error típico si columnas no coinciden con lo que espera el RF
        return error_card(
            "El modelo no pudo predecir. Revisa que las columnas coincidan con columnas_features.pkl. "
            f"<br/><small>{type(e).__name__}: {e}</small>"
        )

    # Formato ARS (sin redondeo a miles)
    formatted = _format_ars(float(y_pred))
    return result_card(formatted)

# ---------------- UI (Blocks: 1 Row, 3 Columns) ----------------
with gr.Blocks(title="Predicción de Precio de Propiedad") as demo:
    with gr.Row():
        # Columna 1: Inputs
        with gr.Column(scale=1):
            gr.Markdown(
                """
                # 🏠 Predicción de precio de propiedad
                **Columna 1 – Datos de entrada**

                Completa los campos con las características de la propiedad.
                Se validan valores y se generan las variables internas que el modelo espera.
                """
            )

            inputs = []
            for f in FEATURES:
                t = f["type"]
                if t == "slider":
                    inputs.append(
                        gr.Slider(
                            minimum=f.get("min", 0),
                            maximum=f.get("max", 10),
                            step=f.get("step", 1),
                            value=f.get("value", 0),
                            label=f["label"]
                        )
                    )
                elif t == "number":
                    inputs.append(
                        gr.Number(
                            label=f["label"],
                            value=f.get("value", 0),
                            precision=None,
                            minimum=f.get("min", 0)
                        )
                    )
                elif t == "dropdown":
                    inputs.append(
                        gr.Dropdown(
                            choices=f["choices"],
                            label=f["label"],
                            value=f.get("value")
                        )
                    )
                else:
                    inputs.append(gr.Textbox(label=f["label"], value=f.get("value", "")))

        # Columna 2: Botones + Output
        with gr.Column(scale=1):
            gr.Markdown(
                """
                # 📈 Resultado
                **Columna 2 – Predicción y acciones**

                Presiona **Predecir** para calcular el precio estimado.
                Usa **Limpiar** para resetear todos los campos.
                """
            )
            btn_predict = gr.Button("🔮 Predecir", variant="primary")
            output = gr.HTML()

            gr.ClearButton(components=[*inputs, output], value="🧹 Limpiar")

            btn_predict.click(predict_price, inputs=inputs, outputs=output)
            # Si querés endpoint fijo:
            # btn_predict.click(predict_price, inputs=inputs, outputs=output, api_name="/predict")

        # Columna 3: Ayuda + Ejemplos
        with gr.Column(scale=1):
            gr.Markdown(
                """
                # 🧭 Ayuda & Ejemplos
                **Columna 3 – Guía y ejemplos**

                Reglas:
                - Superficie cubierta ≤ Superficie total
                - Dormitorios ≤ Ambientes
                - Baños ≤ Ambientes

                ### Ejemplos rápidos
                Carga un ejemplo y presiona **Predecir**.
                """
            )
            gr.Examples(
                examples=[
                    [60, 50, 2, 1, 1, "Departamento", "Capital Federal", "Palermo"],
                    [85, 75, 3, 2, 2, "Casa", "Bs.As. G.B.A. Zona Norte", "San Isidro"],
                    [120, 100, 4, 3, 3, "PH", "Bs.As. G.B.A. Zona Sur", "Lanús"],
                ],
                inputs=inputs,
                label="Ejemplos",
            )

if __name__ == "__main__":
    demo.launch(share=True)