#!/usr/bin/env python
# coding: utf-8
# environment come ducati_delta_bom


# Importa librerie
import streamlit as st
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image # serve per l'immagine?
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
st.set_option('deprecation.showPyplotGlobalUse', False)
from io import BytesIO
import io
from io import StringIO
import math
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import streamlit.components.v1 as components
st.set_page_config(layout="wide")
 
# Layout iniziale e caricamento dati

#url_immagine = 'https://github.com/MarcelloGalimberti/Crazy/blob/main/logo_crazy.png?raw=true'

url_immagine ='https://github.com/alebelluco/Crazy/blob/main/logo_crazy.png?raw=true'

col_1, col_2 = st.columns([2, 3])

with col_1:
    st.image(url_immagine, width=500)

with col_2:
    st.title('Structura hainei | Struttura del capo')

# Processo di caricamento dati BOM
# Caricare immagine dello schema

uploaded_input = st.file_uploader("Încărcați BOM | Carica BOM") # poi da file Gemma
if not uploaded_input:
    st.stop()
df_uploaded = pd.read_excel(uploaded_input)

def importa_BOM(df):
    df = df['Unnamed: 1'].to_frame()
    df = df.rename(columns={'Unnamed: 1':'Nome componente'})
    df = df[['*' in componente for componente in df['Nome componente'].astype(str)]]
    df = df.reset_index(drop=True)
    df['codice'] = [parola.split()[1][parola.split()[1].find('.')+1:] for parola in df['Nome componente']]
    df['CI'] = [parola.split()[2] for parola in df['Nome componente']]
    df = df[['codice','CI']]
    df = df.rename(columns={'codice':'Nome componente'})
    df['CI'] = df['CI'].astype(int)
    df['Splittato'] = ''
    for i in range(len(df)):
        coefficiente = df.CI.iloc[i]  
        if coefficiente != 1:
            l = len(df)
            originale = df['Nome componente'].iloc[i]
            df['Nome componente'].iloc[i] = df['Nome componente'].iloc[i] + '_1'
            df['CI'].iloc[i] = 1
            df['Splittato'].iloc[i]='*'
            for k in range(coefficiente):
                df.loc[l]=df.iloc[i]
                df['Nome componente'].loc[l] =originale + f'_{k+1}'
                df['CI'].loc[l] = 1  
    df = df.sort_values(by='Nome componente')
    df = df.reset_index(drop=True)
    return df

#df_BOM=pd.read_excel(uploaded_input)
df_BOM = importa_BOM(df_uploaded)

st.header('Fișier încărcat | File caricato', divider='violet')
st.dataframe(df_BOM['Nome componente'], width=1500)
  
# Fine caricamento dati BOM

lista_BOM = list(df_BOM['Nome componente'])

#lista_BOM = ['a','b','c','d','e','f','g','h','i','l','m','n','o','p','q','r','s','t']

#lista_macchine = ['Macchina A', 'Macchina B', 'Macchina C']
#macchine = pd.read_excel('/Users/Alessandro/Desktop/APP/Db_Machines.xlsx')
macchine = pd.read_excel('https://github.com/alebelluco/Crazy/blob/main/Db_Machines.xlsx?raw=true')
lista_macchine=list(macchine['Macchina'])

# crea dizionario e lo inizializza
lista_prelievo=[]
for i in range (len(lista_BOM)):
    lista_prelievo.append(f'prelievo_{i}')
key_value_pairs = zip(lista_BOM, lista_prelievo)

if 'dizionario' not in st.session_state:
    st.session_state.dizionario = dict(key_value_pairs)

#if 'archi' not in st.session_state:  # probabilmente non necessario
#    st.session_state.archi = dict()

if 'G' not in st.session_state:
    st.session_state.G=nx.DiGraph()

#if 'counter' not in st.session_state: # probabilmente non necessario
#    st.session_state.counter = 0

def form_callback():
    contatore_componente = 0
    for componente in st.session_state.my_componenti:
        st.session_state.G.add_edge(st.session_state.dizionario[componente],st.session_state.my_descrizione)
        st.session_state.G.nodes[st.session_state.my_descrizione]['output_nodo'] = st.session_state.my_semilavorato
        # altri attributi nodo
        st.session_state.G.nodes[st.session_state.my_descrizione]['mac'] = st.session_state.my_macchina
        st.session_state.G.nodes[st.session_state.my_descrizione]['tc'] = st.session_state.my_tempo_ciclo


        if st.session_state.dizionario[componente][0:8]=='prelievo':
            st.session_state.G.nodes[st.session_state.dizionario[componente]]['mac']='Operazione manuale'
            st.session_state.G.nodes[st.session_state.dizionario[componente]]['tc']=0
            st.session_state.G.edges[st.session_state.dizionario[componente],st.session_state.my_descrizione]['nome_arco']=[st.session_state.my_componenti[contatore_componente]][0]  
        else:
            st.session_state.G.edges[st.session_state.dizionario[componente],st.session_state.my_descrizione]['nome_arco']= st.session_state.G.nodes[st.session_state.dizionario[componente]]['output_nodo'] 
        del st.session_state.dizionario[componente]
        contatore_componente += 1
    st.session_state.dizionario[st.session_state.my_semilavorato] =  st.session_state.my_descrizione
      
if 'Finito' not in list(st.session_state.dizionario.keys()): # mi propone la form solo se nei componenti non c'è Finito. è il modo per terminare il grafo
    with st.form(key='my_form',clear_on_submit=True): 
        st.write("Inserire dati")
        componenti = st.multiselect('**Seleziona componenti**', list(st.session_state.dizionario.keys()), key = 'my_componenti')
        descrizione_operazione = st.text_input('Descrizione operazione (9999 per fine ciclo)', key = 'my_descrizione')
        macchina = st.selectbox('Scegliere macchina',lista_macchine, index=None, key = 'my_macchina')
        tempo_ciclo = st.number_input('Inserire il tempo ciclo:',min_value=0.00, key = 'my_tempo_ciclo')
        semilavorato = st.text_input('Descrizione semilavorato', key = 'my_semilavorato')
        submit_button = st.form_submit_button(label='Submit', on_click=form_callback)
    


pos = nx.fruchterman_reingold_layout(st.session_state.G, k=2, iterations=500)
fig, ax = plt.subplots(figsize=(16, 12), dpi=600)
nx.draw(st.session_state.G, pos, font_size=12, node_size=250, edge_color='white', with_labels=False, # False
        node_color="#C00000", width=0.5, alpha=1,connectionstyle='arc3,rad=0.2', arrows=True)
nx.draw_networkx_labels(st.session_state.G, pos, font_size=9, font_color='white')
nx.draw_networkx_edge_labels(st.session_state.G, pos,font_size=8, font_color='red', edge_labels = nx.get_edge_attributes(st.session_state.G,'nome_arco'),
                                bbox=dict(ec=(0.0, 0.0, 0.0), fc=(0.0, 0.0, 0.0),alpha=0))
ax.set_facecolor('black')
fig.patch.set_alpha(0.0)
st.pyplot(fig)

#st.write(nx.get_node_attributes(st.session_state.G,'mac'))

tempi_ciclo = nx.get_node_attributes(st.session_state.G,'tc')
macchine = nx.get_node_attributes(st.session_state.G,'mac')

#st.write(tempi_ciclo)
#st.write(st.session_state.my_componenti)

#for v in st.session_state.G.nodes:
 #   st.write(st.session_state.G.nodes[v]['mac'])

df_out=nx.to_pandas_edgelist(st.session_state.G)
df_out['TC']=[tempi_ciclo[fase] for fase in df_out.source.astype(str)]
df_out['Macchina']=[macchine[fase] for fase in df_out.source.astype(str)]
df_out['Operazione']=None
df_out=df_out.rename(columns={'source':'Source','target':'Target'})
df_out.drop('nome_arco', axis=1, inplace=True)
df_out = df_out[df_out.Macchina != 'Operazione manuale']
df_out = df_out[['Source','Operazione','Macchina','TC','Target']]
st.dataframe(df_out, width=1500)
#df_out.to_excel('/Users/Alessandro/Desktop/APP/Output.xlsx')

def convert_df(df):
        return df.to_csv(index=False,decimal=',').encode('utf-8')   # messo index=False
csv = convert_df(df_out)
st.download_button(
    label="descărcare ciclu",
    data=csv,
    file_name='Ciclu.csv',
    mime='text/csv',
)
