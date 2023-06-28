import streamlit as st
from preprocess import estandarizar
from segmentar import segmentar_process
from registro import registro_process
from pre_process import pre_process
from previsualize import previsualize
from visualize import visualize

# Definir las páginas y subpáginas
PAGES = {
    "Preprocesar": pre_process,
    "Previsualizar": previsualize,
    "Visualizar": visualize,
    "Estandarizar": estandarizar,
    "Segmentar": segmentar_process,
    "Registro": registro_process    
}

# Mostrar selectbox para seleccionar la página
page = st.sidebar.selectbox("Seleccionar página", options=list(PAGES.keys()))

# Verificar qué página se ha seleccionado
if page in PAGES:
    PAGES[page]()