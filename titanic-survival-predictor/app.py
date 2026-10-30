
import gradio as gr
import pandas as pd
import joblib

model = joblib.load("titanic_model.pkl")


features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

def predict_survival(pclass, sex, age, sibsp, parch, fare): 
    
    sex_value = 1 if sex == "Female" else 0


    passenger = pd.DataFrame([[
        int(pclass),
        sex_value,
        age,
        sibsp,
        parch,
        fare
    ]], columns=features)


    probability = model.predict_proba(passenger)[0][1]

    return {
        "Survived": float(probability),
        "Did not survive": float(1 - probability)
    }



demo = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Dropdown(
            choices=[1, 2, 3],
            value=3,
            label="Passenger Class"
        ),
        gr.Radio(
            choices=["Male", "Female"],
            value="Male",
            label="Sex"
        ),
        gr.Slider(
            minimum=0,
            maximum=100,
            value=25,
            label="Age"
        ),
        gr.Slider(
            minimum=0,
            maximum=8,
            step=1,
            value=0,
            label="Siblings / Spouses aboard"
        ),
        gr.Slider(
            minimum=0,
            maximum=6,
            step=1,
            value=0,
            label="Parents / Children aboard"
        ),
        gr.Number(
            value=20,
            minimum=0,
            label="Ticket Fare"
        )
    ],
    outputs=gr.Label(
        num_top_classes=2,
        label="Survival Prediction"
    ),
    title=" Titanic Survival Predictor",
    description="Enter passenger details to predict their probability of surviving the Titanic disaster."
)

demo.launch()