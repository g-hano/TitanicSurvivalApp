import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def add_prediction(Class, Sex, Age, Sibling, Parch, Embarked):
    model = joblib.load("model/model.pkl")
    scaler = joblib.load("model/scaler.pkl")

    # change input into a numpy array
    input_array = np.array(input_data)

    # example input
    new_data = [[Class, Sex, Age, Sibling, Parch, Embarked]]

    # scale the datas
    new_data_scaled = scaler.transform(new_data)

    # make predictions
    prediction = model.predict(new_data_scaled)

    return prediction
def add_sidebar():
    st.sidebar.header("Your informations")

    # we need to change values to numeric for using it in our model
    age_numeric = st.sidebar.number_input('Age ', min_value=0, max_value=120, value=25)

    Gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    gender_numeric = 1 if Gender == 'Male' else 0

    Class = st.sidebar.selectbox('Seat Class', ('1', '2', '3'))
    class_numeric = int(Class)

    sibling_numeric = st.sidebar.number_input('Number of siblings ', min_value=0, max_value=5, value=0)

    parch_numeric = st.sidebar.number_input('Number of parents ', min_value=0, max_value=2, value=1)

    Embark = st.sidebar.selectbox('Did you embark the Titanic', ('Yes', 'No'))
    embark_numeric = 3 if Embark == 'Yes' else 4

    button_style = '''
        <style>
        .stButton button {
            width: 300px;
            height: 50px;
            font-size: 20px;
            background-color: #FF6D60; /* Set the background color to a desired value */
            color: #ffffff; /* Set the text color to a desired value */
            border-radius: 5px;
        }
        .stButton button:active {
            color: #000000; /* Set the text color to black when the button is clicked */
        }
        .stButton button:hover {
            color: #000000; /* Set the text color to black when the button is clicked */
        }
        </style>
        '''

    st.markdown(button_style, unsafe_allow_html=True)
    col1, col2 = st.sidebar.columns(2)

    with col1:
        button = st.button("Jack & Rose")
    if button:
        return class_numeric, gender_numeric, age_numeric, sibling_numeric, parch_numeric, embark_numeric

st.set_page_config(
    page_title="Titanic Predictor",
    page_icon=":ship:",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.container():
    st.title("Titanic🚢")
    st.write("Let's check out who would survive the disaster!")

life = -1
input_data = add_sidebar()
col3, col4 = st.columns([4, 1])

if input_data is not None:
    class_numeric, gender_numeric, age_numeric, sibling_numeric, parch_numeric, embark_numeric = input_data
    result = add_prediction(class_numeric, gender_numeric, age_numeric, sibling_numeric, parch_numeric, embark_numeric)
    with col4:
        if result == 0:
            st.error("You would die")
            life = 0
        else:
            st.success("You would survive")
            life = 1

        if gender_numeric == 0:
            gender = "male"
        else:
            gender = "female"

        if embark_numeric == 3:
            embark = "Yes"
        else:
            embark = "No"

            # Display the user information with frames
        st.markdown('Class: ')
        st.markdown(
            f"<div style='border: 1px solid gray; border-radius: 20px; padding: 10px'>{str(class_numeric)}</div>",
            unsafe_allow_html=True)

        st.markdown('Gender: ')
        st.markdown(f"<div style='border: 1px solid gray; border-radius: 20px; padding: 10px'>{gender}</div>",
                    unsafe_allow_html=True)

        st.markdown('Age: ')
        st.markdown(f"<div style='border: 1px solid gray; border-radius: 20px; padding: 10px'>{str(age_numeric)}</div>",
                    unsafe_allow_html=True)

        st.markdown('Sibling: ')
        st.markdown(
            f"<div style='border: 1px solid gray; border-radius: 20px; padding: 10px'>{str(sibling_numeric)}</div>",
            unsafe_allow_html=True)

        st.markdown('Parch: ')
        st.markdown(
            f"<div style='border: 1px solid gray; border-radius: 20px; padding: 10px'>{str(parch_numeric)}</div>",
            unsafe_allow_html=True)

        st.markdown('Embark: ')
        st.markdown(f"<div style='border: 1px solid gray; border-radius: 5px; padding: 10px'>{embark}</div>",
                    unsafe_allow_html=True)


if life == -1:
    with col3:
        enjoy_image = Image.open('images/enjoy.jpeg')
        # Resize the image to a smaller size
        new_width = 400
        new_height = int((enjoy_image.size[1] * new_width) / enjoy_image.size[0])
        resized_enjoy_image = enjoy_image.resize((new_width, new_height))

        # Display the resized image
        st.image(resized_enjoy_image, use_column_width=300)
elif life == 0:
    with col3:
        die_image = Image.open('images/die.jpg')
        # Resize the image to a smaller size
        new_width = 400
        new_height = int((die_image.size[1] * new_width) / die_image.size[0])
        resized_die_image = die_image.resize((new_width, new_height))

        # Display the resized image
        st.image(resized_die_image, use_column_width=300)
else:
    with col3:
        live_image = Image.open('images/live.jpg')
        # Resize the image to a smaller size
        new_width = 400
        new_height = int((live_image.size[1] * new_width) / live_image.size[0])
        resized_live_image = live_image.resize((new_width, new_height))

        # Display the resized image
        st.image(resized_live_image, use_column_width=300)

st.markdown("---")
# Create a container for the name and date
name_date_container = st.container()

# Display the name and date with smaller font size
name_date_container.markdown(
    f"<div style='text-align: center; font-size: 12px;'>Cihan Yalçın - 05/18/2023 - github.com/Pharaoh-C</div>",
    unsafe_allow_html=True
)



# Cihan Yalçın - 05/18/2023 - https://github.com/Pharaoh-C