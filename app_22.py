import pandas as pd
import streamlit as st
import numpy as np
from pycaret.regression import load_model, predict_model

st.set_page_config(page_title="Modelo de Predicción de Resultados de Anuncios")

@st.cache(allow_output_mutation=True)
def get_model():
    return load_model("final_et_pauta_3")

def predict(model, df):
    predictions = predict_model(model, data = df)
    return predictions['Label'][0]

model = get_model()

st.title("Modelo de Predicción de Resultados de Anuncios")
st.markdown("Elija los valores para pronosticar el Resultado del Anuncio")

form = st.form("anuncios")
Importe_gastado = form.number_input('Importe_gastado', min_value = 0.0, max_value = 20000.00,value=0.0, format = '%f', step = 0.01)
clasificacion_list = ['Atributos','Especiales','Estratégicas','Estratégico','Spots','Territorio']
Clasificacion_descripcion = form.selectbox('Clasificacion_descripcion', clasificacion_list)
objetivo_list = ['Interacción','ThruPlay']
Objetivo_descripcion = form.selectbox('Objetivo_descripcion', objetivo_list)
redsocial_list = ['Dark Post', 'Facebook','Instagram']
Red_Social_descripcion = form.selectbox('Red_Social_descripcion', redsocial_list)

predict_button = form.form_submit_button('Predict')

input_dict = {'Importe_gastado': Importe_gastado, 'Clasificacion_descripcion': Clasificacion_descripcion, 'Objetivo_descripcion': Objetivo_descripcion, 'Red_Social_descripcion': Red_Social_descripcion}

input_df = pd.DataFrame([input_dict])

if predict_button: 
    out = predict(model, input_df)
    out = '{0:,.2f}'.format(out)
    st.success(f'La prediccion del Resultado Alcanzado es {out}.')
