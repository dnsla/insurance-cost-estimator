# importing required libraries
import pandas as pd
import joblib
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# loading trained Ridge regression model from file
best_ridge = joblib.load("ridge_model.pkl")

# initializing dash app
app = dash.Dash(__name__)
server = app.server  # exposing server variable for deployment

# defining layout of the web app
app.layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f9f9f9', 'padding': '50px'},
    children=[
        # creating app title
        html.H1("Insurance Cost Estimator", style={'textAlign': 'center', 'color': '#333'}),

        # creating form container
        html.Div([
            # creating input for age
            html.Label("Age"),
            dcc.Input(id='age', type='number', placeholder='Enter age', min=18, max=64,
                      style={'width': '100%', 'padding': '8px'}),

            # creating dropdown for sex
            html.Label("Sex"),
            dcc.Dropdown(
                id='sex',
                options=[{'label': 'Male', 'value': 1}, {'label': 'Female', 'value': 0}],
                placeholder="Select sex",
                style={'width': '100%'}
            ),

            # creating input for BMI
            html.Label("BMI"),
            dcc.Input(id='bmi', type='number', placeholder='Enter BMI', min=10, max=60,
                      style={'width': '100%', 'padding': '8px'}),

            # creating input for number of children
            html.Label("Number of Children"),
            dcc.Input(id='children', type='number', placeholder='Enter number of children', min=0,
                      style={'width': '100%', 'padding': '8px'}),

            # creating dropdown for smoker status
            html.Label("Smoker"),
            dcc.Dropdown(
                id='smoker',
                options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                placeholder="Select smoker status",
                style={'width': '100%'}
            ),

            # creating dropdown for region
            html.Label("Region"),
            dcc.Dropdown(
                id='region',
                options=[
                    {'label': 'Northeast', 'value': 0},
                    {'label': 'Northwest', 'value': 1},
                    {'label': 'Southeast', 'value': 2},
                    {'label': 'Southwest', 'value': 3}
                ],
                placeholder="Select region",
                style={'width': '100%'}
            ),

            # adding spacing and predict button
            html.Br(),
            html.Button("Predict", id='predict-button', n_clicks=0,
                        style={
                            'width': '100%',
                            'backgroundColor': '#4CAF50',
                            'color': 'white',
                            'padding': '12px',
                            'border': 'none',
                            'borderRadius': '5px',
                            'fontSize': '16px',
                            'cursor': 'pointer'
                        }),

            # creating area to display prediction result
            html.Br(),
            html.Div(id='prediction-output', style={'marginTop': '20px', 'fontSize': '20px', 'fontWeight': 'bold'})
        ], style={
            'backgroundColor': 'white',
            'padding': '30px',
            'borderRadius': '8px',
            'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.1)',
            'maxWidth': '500px',
            'margin': 'auto'
        })
    ]
)

# defining callback to handle prediction logic
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('age', 'value'),
    State('sex', 'value'),
    State('bmi', 'value'),
    State('children', 'value'),
    State('smoker', 'value'),
    State('region', 'value')
)
def predict_charges(n_clicks, age, sex, bmi, children, smoker, region):
    try:
        # checking if all fields are filled
        if None in [age, sex, bmi, children, smoker, region]:
            return "Please fill in all fields before predicting."

        # defining min and max values for scaling
        age_min, age_max = 18, 64
        bmi_min, bmi_max = 15.96, 53.13
        charge_min, charge_max = 1121.87, 63770.43

        # scaling age and bmi
        age_scaled = (age - age_min) / (age_max - age_min)
        bmi_scaled = (bmi - bmi_min) / (bmi_max - bmi_min)

        # determining age group
        if age < 30:
            age_group = 'adult'
        elif age < 60:
            age_group = 'middle_aged'
        else:
            age_group = 'senior'

        # determining bmi category
        if bmi < 25:
            bmi_category = 'normal'
        elif bmi < 30:
            bmi_category = 'overweight'
        else:
            bmi_category = 'obese'

        # encoding categorical variables
        sex_male = 1 if sex == 1 else 0
        smoker_yes = 1 if smoker == 1 else 0
        region_northwest = 1 if region == 1 else 0
        region_southeast = 1 if region == 2 else 0
        region_southwest = 1 if region == 3 else 0

        age_group_adult = 1 if age_group == 'adult' else 0
        age_group_middle_aged = 1 if age_group == 'middle_aged' else 0
        age_group_senior = 1 if age_group == 'senior' else 0

        bmi_category_normal = 1 if bmi_category == 'normal' else 0
        bmi_category_overweight = 1 if bmi_category == 'overweight' else 0
        bmi_category_obese = 1 if bmi_category == 'obese' else 0

        # creating input dataframe for prediction
        input_data = pd.DataFrame([{
            'age': age_scaled,
            'bmi': bmi_scaled,
            'children': children,
            'sex_male': sex_male,
            'smoker_yes': smoker_yes,
            'region_northwest': region_northwest,
            'region_southeast': region_southeast,
            'region_southwest': region_southwest,
            'age_group_adult': age_group_adult,
            'age_group_middle_aged': age_group_middle_aged,
            'age_group_senior': age_group_senior,
            'bmi_category_normal': bmi_category_normal,
            'bmi_category_overweight': bmi_category_overweight,
            'bmi_category_obese': bmi_category_obese
        }])

        # making prediction and clamping output between 0 and 1
        scaled_prediction = best_ridge.predict(input_data)[0]
        scaled_prediction = max(0, min(scaled_prediction, 1))

        # converting scaled output to real dollar amount
        real_prediction = scaled_prediction * (charge_max - charge_min) + charge_min

        # returning formatted prediction string
        return f"Estimated Insurance Charges: ${real_prediction:,.2f}"

    except Exception as e:
        # returning error message if something goes wrong
        return f"⚠️ Prediction error: {str(e)}"

# running the app
if __name__ == '__main__':
    app.run(debug=True)
