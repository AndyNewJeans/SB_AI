import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load your model and scaler
model = tf.keras.models.load_model("saved_model.h5")
scaler = MinMaxScaler()  # Assume your scaler is already fitted

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("Water Activity Level Prediction"),
    dcc.Slider(id='drying_time', min=0, max=100, step=1, value=50, marks={i: str(i) for i in range(0, 101, 20)}),
    dcc.Slider(id='moisture_content', min=0, max=100, step=1, value=50, marks={i: str(i) for i in range(0, 101, 20)}),
    dcc.Slider(id='primary_drying_temp', min=-10, max=30, step=1, value=10, marks={i: str(i) for i in range(-10, 31, 10)}),
    dcc.Slider(id='secondary_drying_temp', min=20, max=80, step=1, value=50, marks={i: str(i) for i in range(20, 81, 20)}),
    dcc.Slider(id='pressure', min=0, max=1, step=0.1, value=0.5, marks={i/10: str(i/10) for i in range(0, 11, 2)}),
    dcc.Slider(id='sample_thickness', min=0, max=1, step=0.1, value=0.5, marks={i/10: str(i/10) for i in range(0, 11, 2)}),
    html.Div(id='prediction-output', style={'margin-top': '20px'})
])

# Define callback to update prediction
@app.callback(
    Output('prediction-output', 'children'),
    [Input('drying_time', 'value'),
     Input('moisture_content', 'value'),
     Input('primary_drying_temp', 'value'),
     Input('secondary_drying_temp', 'value'),
     Input('pressure', 'value'),
     Input('sample_thickness', 'value')]
)
def update_prediction(drying_time, moisture_content, primary_drying_temp, secondary_drying_temp, pressure, sample_thickness):
    input_data = np.array([[drying_time, moisture_content, primary_drying_temp, secondary_drying_temp, pressure, sample_thickness]])
    scaled_input = scaler.transform(input_data)  # Apply scaling to input data
    prediction = model.predict(input_data)
    return f'Predicted Water Activity Level: {prediction[0][0]}'

if __name__ == '__main__':
    app.run_server(debug=True)
