import streamlit as sl
import pickle 
import pandas as pd
import numpy as np

# sl.title("HALO")
# sl.write("Coba yaa")

sl.title("""
    Selamat datang!
    \nWebsite ini dapat membantu anda untuk memperoleh informasi lebih cepat terkait probabilitas seorang pelanggan melakukan churn
""")

# if 'value' not in sl.session_state:
#     sl.session_state['value'] = 0

# if sl.button("Tambah"):
#     sl.session_state['value'] += 1

# if sl.button("Kurang"):
#     sl.session_state['value'] -= 1
#     if sl.session_state['value'] < 1:
#         sl.session_state['value'] = 0

# sl.write(f"Counter: {sl.session_state['value']}")

## data preprocessing
if 'feature_scaling' not in sl.session_state:
    sl.session_state['feature_scaling'] = pickle.load(open('feature_preprocess.sav', 'rb'))

# load the model
if 'model' not in sl.session_state:
    sl.session_state['model'] = pickle.load(open('rf_model.sav', 'rb'))

call_failure_input = sl.number_input("Enter the number of call failure:")
complain_list = ["No", "Yes"]
complain_select_box = sl.selectbox(
    "Complain:",
    complain_list
    )
complain_input = complain_list.index(complain_select_box)

# sl.write(f"Complain: {complain_list.index(complain_select_box)}")

subscription_length_input = sl.number_input("Enter the number of Subscription Length:")

charge_amount_input = sl.selectbox(
    "Charge Amount:",
    ('0', '1', '2', '3', '4', '5', '6', '7', '8')
    )

second_of_use_input = sl.number_input("Enter total calls (in seconds):")
frequency_of_use_input = sl.number_input("Enter total number calls:")
total_sms_input = sl.number_input("Enter the number of sms used:")
distinct_number_input = sl.number_input("Enter the distinct number:")

age_list = ['15', '25', '30', '45', '55']
age_group_selectbox = sl.selectbox(
    "Age group:",
    age_list
    )
age_group_input = age_list.index(age_group_selectbox) + 1
age_input = age_group_selectbox

tariff_plan_list = ['Pay as You Go', 'Contractual']
tariff_plan_selectbox = sl.selectbox(
    "Tariff Plan:",
    tariff_plan_list
    )
tariff_plan_input = tariff_plan_list.index(tariff_plan_selectbox) + 1

status_list = ['Active', 'Non-Active']
status_selectbox = sl.selectbox(
    "Status:",
    status_list
    )
status_input = status_list.index(status_selectbox)

cust_value_input = sl.number_input("Enter the customer value:")

if sl.button("Dapatkan Prediksi"):
    feature_list = ['Call  Failure', 'Complains', 'Subscription  Length', 'Charge  Amount',
       'Seconds of Use', 'Frequency of use', 'Frequency of SMS',
       'Distinct Called Numbers', 'Tariff Plan', 'Status', 'Age',
       'Customer Value']
    
    data = np.array([call_failure_input, complain_input, subscription_length_input, int(charge_amount_input), second_of_use_input, frequency_of_use_input, total_sms_input, distinct_number_input, tariff_plan_input, status_input, int(age_input), int(cust_value_input)]).reshape(1,-1)
    data_df = pd.DataFrame(data, columns = feature_list)

    # sl.write(f"Data: {sl.session_state['feature_scaling'].transform(data_df)}")

    data_scaled = sl.session_state['feature_scaling'].transform(data_df)
    
    prediction = sl.session_state['model'].predict(data_scaled)
    
    customer_churn_prediction = "tidak akan churn"
    if prediction[0] == 1:
        customer_churn_prediction = "churn"
        """
            Nb:
            \nIni hanya sebuah prediksi. Harap hubungi dokter untuk memperoleh penanganan lebih detail
        """
    sl.write(f'Pelanggan ini berpotensi {customer_churn_prediction} di masa mendatang')

    
else:
    sl.write("Please input the feature above to start modelling")
