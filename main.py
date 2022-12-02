import pandas as pd
import streamlit as st
from server import helper
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")
sns.set()

helper.get_top_page_content(st)

data_df = pd.read_csv('./server/data/car_price_prediction.csv')


def get_unique_list_by_column(column):
    output_list = data_df[column].unique()
    output_list.sort()
    return output_list


manufacturer_list = data_df['Manufacturer'].unique()
manufacturer_list.sort()
model_list = get_unique_list_by_column('Model')
model_list = [token for token in model_list if not token.isdigit()]
category_list = get_unique_list_by_column('Category')
fuel_type_list = get_unique_list_by_column('Fuel type')
engine_volume_list = get_unique_list_by_column('Engine volume')
cylinders_list = get_unique_list_by_column('Cylinders')
gear_box_list = get_unique_list_by_column('Gear box type')
drive_wheels_list = get_unique_list_by_column('Drive wheels')
doors_list = ['2-3', '4-5', '6 o mas']
wheel_list = ['Izquierdo', 'Derecho']
color_list = get_unique_list_by_column('Color')


with st.form("my_form"):
    st.markdown("## Ingresa los datos del vehículo que estás vendiendo")
    manufacturer = st.selectbox('Fabricante', manufacturer_list)
    levy = st.number_input("Impuesto")
    model = st.selectbox('Modelo', model_list)
    prod_year = st.slider("Año del vehículo", min_value=1995, max_value=2022)
    category = st.selectbox('Categoría', category_list)
    leather_interior = st.checkbox("Interior de cuero")
    if leather_interior:
        leather_interior = 1
    else:
        leather_interior = 0
    fuel_type = st.selectbox('Tipo de combustible', fuel_type_list)
    engine_volume = st.selectbox('Volumen del motor', engine_volume_list)
    mileage = st.number_input("Kilometraje de tu vehículo")
    cylinders = st.selectbox('Cilindros', cylinders_list)
    gear_box = st.selectbox('Tipo de caja', gear_box_list)
    drive_wheels_box = st.selectbox('Tracción', drive_wheels_list)
    doors = st.selectbox('Cantidad de puertas', doors_list)
    wheel = st.selectbox('Dirección del volante', wheel_list)
    color = st.selectbox('Color', color_list)
    airbags = st.number_input("Cantidad de airbags")

   # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        input_df = pd.DataFrame({'Levy': [levy], 'Manufacturer': [manufacturer], 'Model': [model], 'Prod. year': [prod_year], 'Category': [category],
                                 'Leather interior': [leather_interior], 'Fuel type': [fuel_type], 'Engine volume': [engine_volume], 'Mileage': [mileage], 'Cylinders': [cylinders],
                                 'Gear box type': [gear_box], 'Drive wheels': [drive_wheels_box], 'Doors': [doors], 'Wheel': [wheel], 'Color': [color], 'Airbags': [airbags]})
        processed_df = helper.clean_cars(input_df).clean_dataframe()
        model = helper.load_model()
        complete_df = helper.clean_cars(
            data_df).clean_dataframe_original().append(processed_df)
        one_hot_encoding_df = helper.clean_cars(complete_df).encode_one_hot_dataframe().drop(
            ['Color', 'Wheel__Left wheel'], axis=1)
        
        binary_encoding_df = helper.clean_cars(complete_df).encode_binary_dataframe().drop(
        ['Manufacturer_6'], axis = 1)
        binary_encoding_df = helper.apply_scaler(binary_encoding_df)
        data_to_predict = binary_encoding_df.iloc[-1:]
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None) 
        #print(data_to_predict)
        #print(model)
        #one_hot_scaled_df = helper.apply_scaler(one_hot_encoding_df)
        st.write("La TRM del día de hoy es: ", helper.get_currency_exchange())
        st.markdown(
            " ### De acuerdo al modelo el precio estimado en el que puedes vender tu vehículo es:")
        st.write('$'+'{:20,.2f}'.format(model.predict(data_to_predict)[0,0]*helper.get_currency_exchange()))
        #st.write('$'+'{:20,.2f}'.format(model.predict(data_to_predict)
                 #[-1]*helper.get_currency_exchange()))
        #print(binary_encoding_df)
        # st.write(one_hot_encoding_df.iloc[-1])
        # st.write(one_hot_encoding_df.columns)
        #st.write("slider", slider_val, "checkbox", checkbox_val)
