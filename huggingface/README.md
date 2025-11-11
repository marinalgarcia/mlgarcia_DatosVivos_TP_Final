---
title: Predicción de Precio de Propiedades
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
python_version: 3.11
pinned: false
---

\# Predicción de Precio de Propiedades (Gradio + Hugging Face Spaces)



Interfaz simple para ingresar variables de una propiedad y obtener la \*\*predicción del precio en USD\*\* utilizando un modelo entrenado.



\## 🔗 Demo en Hugging Face Spaces

\*\*Space:\*\* \[ENLACE\_AL\_SPACE](https://huggingface.co/spaces/mlgarcia/edvai)



\## 🖥️ Captura de pantalla :

!\[Screenshot](./screenshot.png)



\## 🚀 Cómo usar la app (web)

1\. Abre el Space y completa los campos requeridos.

2\. Presiona \*\*“Submit”\*\* para obtener el precio estimado en USD.



\## 🧠 Modelo

\- Archivo: `rf_default.pkl`  

\- Recomendación: un `Pipeline` de scikit-learn que incluya preprocesamiento + modelo.



\## 📦 Ejecución local

```bash

pip install -r requirements.txt

python app.py

