import streamlit as st
from Frontend_fct import *


st.set_page_config(
    layout='wide',
    initial_sidebar_state='expanded',
    page_title="2) Clients info")

st.set_option('deprecation.showPyplotGlobalUse', False)

#
# --- Retrieving data from home page ---
#

df_train = st.session_state.df_train
Credit_clf_final = st.session_state.Credit_clf_final
explainer = st.session_state.explainer

#
# --- Preparing the Layout of the page ---
#


st.header("Analysis of loan default for known clients")

st.sidebar.header('Dashboard')

st.sidebar.subheader('Client ID selection')

#
# --- Client Analysis (with client ID, without API) ---
#

index_client = st.sidebar.number_input(
    "Enter Client ID (ex : 100002)",
    format="%d",
    value=100002)

run_btn = st.sidebar.button('See Client Data', on_click=None, type="primary")
if run_btn:
    if index_client in set(df_train.index):
        execute_noAPI(df_train, index_client, Credit_clf_final)

        plot_client(
            df_train.drop(
                columns='TARGET').fillna(0),
            explainer,
            df_reference=df_train,
            index_client=index_client)

        nan_values(df_train.drop(columns='TARGET'), index_client=index_client)
    else:
        st.sidebar.write("Client not in database")
