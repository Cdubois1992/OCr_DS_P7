import streamlit as st

from Frontend_fct import *


st.set_page_config(page_title="3) New Prediction")

st.set_option('deprecation.showPyplotGlobalUse', False)

#
# --- Retrieving data from home page ---
#

df_train = st.session_state.df_train
df_new = st.session_state.df_new
Credit_clf_final = st.session_state.Credit_clf_final
explainer = st.session_state.explainer

#
# --- Creation of the two tabs  ---
#

tab1, tab2 = st.tabs(["ID client", "Manual information"])

# --- Tab 1 for client with an ID (df_test)  ---
with tab1:
    st.header("Prediction on client with ID")
    index_client = st.number_input(
        "Enter Client ID (ex : 100001, 100005)",
        format="%d",
        value=100001)

    # Creating a predict button using the API ! (from FastAPI)
    run_btn = st.button(
        'Predict',
        on_click=None,
        type="primary",
        key='predict_btn1')
    if run_btn:
        if index_client in set(df_new.index):
            data_client = df_new.loc[index_client].fillna(0).to_dict()
            execute_API(data_client)
            plot_client(
                df_new,
                explainer,
                df_reference=df_train,
                index_client=index_client)
            nan_values(df_new, index_client=index_client)
        else:
            st.write("Client not in database")


# --- Tab 2 for client without ID (new)  ---

# this second tab presents three possibilities to enter data for prediction :
# 1 option - manually with streamlite number input
# 2 option - manually with a text pre-formated (as dict)
# 3 option - loading a csv file

with tab2:
    st.header("Prediction on new client")

    option = st.selectbox(
        'How would you like to enter data?',
        ('Manual', 'Text', 'CSV file'))

    # 1 option - manually with streamlite number input

    # The following code create a st.number_input for each features
    # that is different between int and float values
    # /!\ Limits : values cannot be higher (or lower) than the max(min) values of the train df.
    if option == 'Manual':
        with st.expander("Click to enter data manually"):
            data_client = {}
            for features in list(df_new.columns):
                if df_train[features].dtype == np.int64:
                    min_values = df_train[features].min().astype('int')
                    max_values = df_train[features].max().astype('int')
                    data_client["{0}".format(features)] = st.number_input(
                        str(features), min_value=min_values, max_value=max_values, step=1)
                else:
                    min_values = df_train[features].min().astype('float')
                    max_values = df_train[features].max().astype('float')
                    data_client["{0}".format(features)] = st.number_input(
                        str(features), min_value=min_values, max_value=max_values, step=0.1)

    # 2 option - manually with a text pre-formated (as dict)
    # The following code create a text area pre-formatted to enter values
    # manually
    elif option == 'Text':
        with st.expander("Click to enter data as text"):
            data_client = st.text_area('Enter data as dict',
                                       '''{"FLAG_OWN_CAR": 0,
                "AMT_CREDIT": 0,
                "AMT_ANNUITY": 0,
                "AMT_GOODS_PRICE": 0,
                "REGION_POPULATION_RELATIVE": 0,
                "DAYS_BIRTH": 0,
                "DAYS_EMPLOYED": 0,
                "DAYS_REGISTRATION": 0,
                "DAYS_ID_PUBLISH": 0,
                "OWN_CAR_AGE": 0,
                "REGION_RATING_CLIENT_W_CITY": 0,
                "EXT_SOURCE_1": 0,
                "EXT_SOURCE_2": 0,
                "EXT_SOURCE_3": 0,
                "DAYS_LAST_PHONE_CHANGE": 0,
                "NAME_CONTRACT_TYPE_Cashloans": 0,
                "NAME_EDUCATION_TYPE_Highereducation": 0,
                "NAME_FAMILY_STATUS_Married": 0,
                "DAYS_EMPLOYED_PERC": 0,
                "INCOME_CREDIT_PERC": 0,
                "ANNUITY_INCOME_PERC": 0,
                "PAYMENT_RATE": 0,
                "BURO_DAYS_CREDIT_MAX": 0,
                "BURO_DAYS_CREDIT_MEAN": 0,
                "BURO_DAYS_CREDIT_ENDDATE_MAX": 0,
                "BURO_AMT_CREDIT_MAX_OVERDUE_MEAN": 0,
                "BURO_AMT_CREDIT_SUM_MEAN": 0,
                "BURO_AMT_CREDIT_SUM_DEBT_MEAN": 0,
                "BURO_CREDIT_TYPE_Microloan_MEAN": 0,
                "BURO_CREDIT_TYPE_Mortgage_MEAN": 0,
                "ACTIVE_DAYS_CREDIT_MAX": 0,
                "ACTIVE_DAYS_CREDIT_ENDDATE_MIN": 0,
                "ACTIVE_DAYS_CREDIT_ENDDATE_MEAN": 0,
                "ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN": 0,
                "ACTIVE_AMT_CREDIT_SUM_SUM": 0,
                "ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN": 0,
                "CLOSED_DAYS_CREDIT_VAR": 0,
                "CLOSED_AMT_CREDIT_SUM_MAX": 0,
                "CLOSED_AMT_CREDIT_SUM_SUM": 0,
                "PREV_APP_CREDIT_PERC_MIN": 0,
                "PREV_APP_CREDIT_PERC_MEAN": 0,
                "PREV_CNT_PAYMENT_MEAN": 0,
                "PREV_NAME_CONTRACT_STATUS_Refused_MEAN": 0,
                "PREV_NAME_YIELD_GROUP_low_action_MEAN": 0,
                "PREV_PRODUCT_COMBINATION_CashXSelllow_MEAN": 0,
                "APPROVED_AMT_ANNUITY_MEAN": 0,
                "APPROVED_AMT_DOWN_PAYMENT_MAX": 0,
                "APPROVED_CNT_PAYMENT_MEAN": 0,
                "POS_MONTHS_BALANCE_MAX": 0,
                "POS_MONTHS_BALANCE_SIZE": 0,
                "POS_SK_DPD_DEF_MEAN": 0,
                "INSTAL_DPD_MEAN": 0,
                "INSTAL_DBD_SUM": 0,
                "INSTAL_PAYMENT_PERC_SUM": 0,
                "INSTAL_PAYMENT_DIFF_MEAN": 0,
                "INSTAL_AMT_INSTALMENT_MAX": 0,
                "INSTAL_AMT_PAYMENT_MIN": 0,
                "INSTAL_AMT_PAYMENT_SUM": 0,
                "INSTAL_DAYS_ENTRY_PAYMENT_MAX": 0,
                "INSTAL_DAYS_ENTRY_PAYMENT_MEAN": 0,
                "CC_CNT_DRAWINGS_CURRENT_MEAN": 0,
                "CC_CNT_DRAWINGS_CURRENT_VAR": 0
                }''')

            data_client = json.loads(data_client)

    # 3 option - loading a csv file
    # the csv file must have a ";" as sep, and include 2 columns
    # The first one with feature's name, the seconde with feature's value
    else:
        loader = st.file_uploader(" ")
        if loader is not None:
            data_client = pd.read_csv(
                loader,
                sep=";",
                index_col=0,
                header=None).squeeze(1).to_dict()

    # Creating a predict button using the API ! (from FastAPI)
    run_btn2 = st.button(
        'Predict',
        on_click=None,
        type="primary",
        key='predict_btn2')
    if run_btn2:
        execute_API(data_client)
        data_client = pd.DataFrame(data_client, index=[0])
        plot_client(
            data_client,
            explainer,
            df_reference=df_train,
            index_client=0)
        nan_values(data_client, index_client=0)
