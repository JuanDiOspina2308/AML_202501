import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Cargar el logo
logo = Image.open("logo.png")
st.image(logo, width=100)

st.title("RUSH OR PASS PREDICTOR")

st.subheader("Parameters of the play")

# Inputs
quarter = st.number_input("Quarter", min_value=1, max_value=4, value=1)
down = st.number_input("Down", min_value=1, max_value=4, value=1)
yards_to_go = st.number_input("Yards to go", min_value=1, max_value=100, value=10)
game_clock = st.number_input("Game clock (in minutes, e.g. 1:30 → 1.5)", min_value=0.0, max_value=15.0, value=3.16, step=0.01)
home_score = st.number_input("Home Score", value=0)
visitor_score = st.number_input("Visitor Score", value=0)
is_dropback = st.selectbox("Is dropback (True - 1 / False - 0)", [0, 1])
pff_rpo = st.selectbox("PFF run pass option (True - 1 / False - 0)", [0, 1])

# Offensive formation (solo uno puede estar en 1)
formations = {
    "EMPTY": 0,
    "IFORM": 0,
    "JUMBO": 0,
    "PISTOL": 0,
    "SHOTGUN": 0,
    "SINGLEBACK": 0,
    "WILDCAT": 0
}
formation_choice = st.selectbox("Offensive Formation", list(formations.keys()))
formations[formation_choice] = 1

# Cargar modelos
modelo_RF = joblib.load('clasificadorRandomForest.joblib')
modelo_DT = joblib.load('clasificadorDecisionTree.joblib')

# Botón para predecir
if st.button("Predict"):
    input_data = np.array([[
        quarter, down, yards_to_go, game_clock,
        home_score, visitor_score, is_dropback, pff_rpo,
        formations["EMPTY"], formations["IFORM"], formations["JUMBO"],
        formations["PISTOL"], formations["SHOTGUN"],
        formations["SINGLEBACK"], formations["WILDCAT"]
    ]])

    # Lógica ejemplo: usar modelo_2 si quedan menos de 2 minutos del último cuarto
    # if quarter == 4 and game_clock < 2:
    #     model = modelo_DT
    # else:
    #     model = modelo_RF

    model = modelo_DT

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    # Etiquetas legibles
    label = "RUN" if prediction == 1 else "PASS"
    probability = round(probabilities[prediction] * 100, 2)

    # Mostrar resultado
    st.markdown("### Rush or pass prediction")
    st.success(f"**PREDICTION: {label}**  \nConfidence: {probability}%")