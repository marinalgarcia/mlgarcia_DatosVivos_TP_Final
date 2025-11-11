                                        Datos Vivos TP Final Marina García

# EDA

## Descripción General:
El mercado inmobiliario argentino es dinámico y presenta variaciones significativas dependiendo de la ubicación, el tipo de propiedad y el tamaño de las propiedades, lo que conlleva a un desafio constante para determinar precios acordes a las caracteristicas de los inmuebles por las diferentes zonas geograficas.

## Insights:

Distribución Geográfica de las Propiedades:: 
la mayor concentración de anuncios se encuentra en Capital Federal (52%)
Tipo de Propiedad:
La mayoría de los anuncios (63%) corresponde a departamentos
Superficie Promedio de los Departamentos:
Bs.As. G.B.A. Zona Norte presenta la superficie promedio más grande (77.53 m²), lo que indica un mercado orientado a propiedades más amplias, probablemente debido a su desarrollo residencial y una mayor demanda por espacio

## Conclusión Final: 
Capital Federal sigue siendo el mercado inmobiliario más dinámico y caro, especialmente en barrios como Puerto Madero y Recoleta, que continúan siendo puntos de referencia en términos de valor.

# MODELO

## Elección del Modelo:

La elección de utilizar el modelo de Random Forest (default) porque proporciona un equilibrio adecuado entre precisión y costo computacional

## Insights:

El modelo explica aproximadamente el 68 % de la variabilidad de los precios. 
En promedio, el modelo se equivoca en 39,689 dólares por predicción, PERO SE OBSERVAN algunos errores grandes
Hay factores no incluidos en los datos que influyen en la valuación de una propiedad,, por ejemplo disponibilidad de amenities, antiguedad y estado del inmueble entre otros.

Variables incluidas:
Se eligieron variables que tenen un gran imapctro a la hora de predecir el valor de una propiedad.:
	surface_total y surface_covered
	property_type:
	state_name y place_name
	Rooms, bedrooms y bathrooms
Variables descartadas:
	Latitud y Longitud: la ubicación geográfica ya se encuentra representada en otras variables.
	Fecha de Publicación: no suele tener una relación directa con el precio de la propiedad. 

El storytelling detallado, junto a la justificación de las decisiones se encuentran al final de cada archivo .ipynb.

# Predicción de Precio de Propiedades (Gradio + Hugging Face Spaces)

**Link:** https://huggingface.co/spaces/mlgarcia/edvai

Interfaz simple para ingresar variables de una propiedad y obtener la **predicción del precio en USD** utilizando un modelo entrenado.

## 🔗 Demo en Hugging Face Spaces

**Space:** [ENLACE_AL_SPACE](https://huggingface.co/spaces/mlgarcia/edvai)

## 🖥️ Captura de pantalla :

<img width="1347" height="696" alt="image" src="https://github.com/user-attachments/assets/a60c95b6-e003-4a37-9978-c7860fc0b116" />

## 🚀 Cómo usar la app (web)

1. Abre el Space y completa los campos requeridos.

2. Presiona **“Submit”** para obtener el precio estimado en USD.

## 🧠 Modelo

- Archivo: rf_default.pkl
- Recomendación: un Pipeline de scikit-learn que incluya preprocesamiento + modelo.
- En la carpeta huggingface se encontrarán los archivos mencionados para la ejecucion.

## 📦 Ejecución local

En la carpeta huggingface se encontrarán los archivos mencionados para la ejecucion.

pip install -r requirements.txt

python app.py

## 📦 Ejemplo del predictor por API

<pre>```python
!pip install gradio_client

from gradio_client import Client

client = Client("mlgarcia/edvai")

result = client.predict(

    param_0="Casa",
		param_1="Capital Federal",
		param_2="Palermo",
		param_3=2,
		param_4=1,
		param_5=1,
		param_6=60,
		param_7=50,
		api_name="/predict_price"
)
print(result)```</pre>


El código además se encuentra en Use Via API - ejemplo de predictor.ipynb
