
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

st.set_page_config(page_title="1) General information")


#
# --- Retrieving data from home page ---
#

df_train = st.session_state.df_train
Credit_clf_final = st.session_state.Credit_clf_final
explainer = st.session_state.explainer


#
# --- Creating the layout of the page (3 tabs) ---
#

tab1, tab2, tab3 = st.tabs(["Data", "Indicators", "Model"])

# --- Tab1 : Presentation of the dataframe (content, target, missing values) ---
with tab1:

    # Layout of the tab
    st.header("Overview of the Dataframe")
    st.subheader("Content of the dataframe")

    # Number of client registered :
    col1, col2 = st.columns(2)
    col1.metric("Number of (known) client registered", df_train.shape[0])
    # Number of features
    col2.metric(
        "Number of client features",
        df_train.drop(
            columns='TARGET').shape[1])

    # Analysis of the target : Donut chart
    st.subheader("Analysis of the target")
    fig1, ax = plt.subplots()
    ax.pie(df_train.TARGET.value_counts(normalize=True),
           labels=["0",
                   "1"],
           autopct=lambda p: '{:.1f}%\n({:.0f})'.format(p,
                                                        (p / 100) * sum(df_train.TARGET.value_counts())),
           startangle=0,
           pctdistance=0.8,
           explode=(0.05,
                    0.05))  # draw circle
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.gca().add_artist(centre_circle)
    plt.axis('equal')
    plt.title(
        'Number of clients with difficulties (1) or not (0)\nto repay the loan')
    plt.tight_layout()
    plt.close()
    st.pyplot(fig1)

    # Number of NaN (with missingno)
    st.subheader("Analysis of missing values")
    with st.spinner('Loading figure...'):
        figNAN = msno.matrix(
            df_train.drop(
                columns='TARGET'),
            labels=True,
            sort="ascending")
        plt.close()
        st.pyplot(figNAN.figure)

        st.markdown("""
        Information about missing values :
        1) Variables with more than 80% of missing values were previously discarded
        2) All missing values are then replaced by 0
        """)


# --- Tab2 : Presentation of the features ---
# The 62 features has been presented in two columns with 31 plots each.
# For binary features : Plot of the frequency of the feature within each class (TARGET)
# For other features : Box Plot of the feature colored by each class (TARGET)
with tab2:
    cola, colb = st.columns(2)
    with cola:
        for features in list(df_train.drop(columns='TARGET').columns)[:31]:
            if df_train[features].nunique() == 2:
                figInd = sns.barplot(df_train[['TARGET', features]].fillna(0).groupby(
                    'TARGET').value_counts(normalize=True).reset_index(), x=features, y=0, hue="TARGET")
                plt.ylabel('Freq of client')
                legend_handles, _= figInd.get_legend_handles_labels()
                figInd.legend(legend_handles,['No','Yes'], title="LOAN DEFAULT")
                figInd.set_xticklabels(["No", "Yes"])
                plt.close()
                st.pyplot(figInd.figure)
            else:
                figInd = sns.boxplot(
                    data=df_train,
                    y=features,
                    x='TARGET',
                    showfliers=False)
                plt.xlabel('LOAN DEFAULT')
                figInd.set_xticklabels(["No", "Yes"])
                plt.close()
                st.pyplot(figInd.figure)
    with colb:
        for features in list(df_train.drop(columns='TARGET').columns)[31:]:
            if df_train[features].nunique() == 2:
                figInd = sns.barplot(df_train[['TARGET', features]].fillna(0).groupby(
                    'TARGET').value_counts(normalize=True).reset_index(), x=features, y=0, hue="TARGET")
                plt.ylabel('Freq of client')
                figInd.set_xticklabels(["No", "Yes"])
                legend_handles, _= figInd.get_legend_handles_labels()
                figInd.legend(legend_handles,['No','Yes'], title="LOAN DEFAULT")
                plt.close()
                st.pyplot(figInd.figure)
            else:
                figInd = sns.boxplot(
                    data=df_train,
                    y=features,
                    x='TARGET',
                    showfliers=False)
                plt.xlabel('LOAN DEFAULT')
                figInd.set_xticklabels(["No", "Yes"])
                plt.close()
                st.pyplot(figInd.figure)

# --- Tab3 : Presentation of the model ---
with tab3:
    st.header("Model description")

    # Load Feature importance
    st.subheader("Feature importance of the lightgbm model")
    st.image("image/Plot_importance.png")

    # Load parameter
    st.subheader("Parameters (optimised with Optuna)")
    st.table(
        pd.DataFrame.from_dict(
            Credit_clf_final.get_params(),
            orient='index',
            columns=['Parameter']))

    # Load scores
    st.subheader("Score obtained by cross-validation")
    st.write(pd.DataFrame({
        'Metrics': ["AUC", "Accuracy", "F1", "Precision", "Recall", "Profit"],
        'second column': [0.764, 0.869, 0.311, 0.271, 0.366, 1.928],
    }))

    # Load Confusion matrix on the test set
    st.subheader("ROC curve and Confusion matrix on a test set")

    col1, col2 = st.columns(2)
    with col1:
        st.image("image/Test_ROC_AUC.png")
    with col2:
        st.image("image/Test_confusion_matrix.png")
