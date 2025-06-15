import streamlit as st
import pandas as pd
import joblib
import io


st.markdown(
        """
        <style>
            header {visibility: hidden;}
            .main {
                margin-top: -20px;
                padding-top: 10px;
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .navbar {
                padding: 1rem;
                margin-bottom: 2rem;
                background-color: #4267B2;
                color: white;
            }
            .card {
                padding: 1rem;
                margin-bottom: 1rem;
                transition: transform 0.2s;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .card:hover {
                transform: scale(1.02);
            }
            .metric-card {
               
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem;
                
            }
            .search-box {
                margin-bottom: 1rem;
                padding: 0.5rem;
                border-radius: 4px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
st.markdown(
    """
    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.19.1/css/mdb.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"
          integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    """,
    unsafe_allow_html=True
)



def create_navbar():
    st.markdown(
        """
        <nav class="navbar fixed-top navbar-expand-lg navbar-dark bg-blue text-bold shadow-sm">
            <a class="navbar-brand text-white" href="#" target="_blank">
            AZUBI
            </a>
        </nav>
        <br><br><br>
        """,
        unsafe_allow_html=True
    )


create_navbar()


def load_artifacts():
    """
    Load model, scaler, and feature columns.
    """
    model = joblib.load('term_deposit_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('columns.pkl')
    return model, scaler, feature_columns


model, scaler, feature_columns = load_artifacts()

numeric_cols = [
    'age', 'duration', 'campaign', 'pdays', 'previous',
    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
    'euribor3m', 'nr.employed'
]


st.title('Term Deposit Subscription Predictor')


with st.sidebar:
    st.header('Upload Data & Settings')
    delimiter = st.selectbox('CSV Delimiter', [',', ';'], index=1)
    uploaded_file = st.file_uploader('CSV File', type=['csv'])


if uploaded_file:

    raw = uploaded_file.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(raw), sep=delimiter)


    df_enc = pd.get_dummies(df, drop_first=True)
    df_enc = df_enc.reindex(columns=feature_columns, fill_value=0)
    df_enc[numeric_cols] = scaler.transform(df_enc[numeric_cols])


    preds = model.predict(df_enc)
    probs = model.predict_proba(df_enc)[:, 1]
    df['Predicted Subscription'] = ['yes' if p else 'no' for p in preds]
    df['Probability of Yes'] = probs

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Uploaded Data Sample')
        st.write(df.head())
    with col2:
        st.subheader('Prediction Results')
        st.write(df[['Predicted Subscription', 'Probability of Yes']].head())
    
        csv = df.to_csv(index=False)
        st.download_button(
            label='Download as CSV',
            data=csv,
            file_name='predictions.csv',
            mime='text/csv'
        )
else:
    st.info('Awaiting CSV file upload via the sidebar.')
