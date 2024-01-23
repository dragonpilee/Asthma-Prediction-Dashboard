import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Dummy DataFrame for testing (replace this with your actual DataFrame)
data = {'Age': [25, 30, 35, 40, 45],
        'Smoking': ['No', 'Yes', 'No', 'Yes', 'No'],
        'Asthma': [0, 1, 0, 1, 0]}
df = pd.DataFrame(data)

# Initialize the Dash app
app = dash.Dash(__name__)

# Set the app layout
app.layout = html.Div(
    children=[
        html.H1("Asthma Prediction Dashboard", style={'textAlign': 'center'}),
        
        # File browser (replace this with your actual file browser component)
        dcc.Upload(
            id='upload-data',
            children=html.Button('Upload Data'),
            multiple=False
        ),
        
        # Analyze button
        html.Button('Analyze', id='analyze-button', style={'marginBottom': '20px'}),
        
        # Uploaded message
        html.Div(id='uploaded-message', style={'marginBottom': '20px'}),
        
        # Prediction result
        html.Div(id='prediction-result', style={'marginBottom': '20px'}),
        
        # Scientific visualizations
        dcc.Graph(id='output-histogram'),
        dcc.Graph(id='output-scatter'),
        dcc.Graph(id='output-pie'),
        
        # Store to keep track of upload status
        dcc.Store(id='upload-status', data={'uploaded': False}),
    ],
    style={'width': '80%', 'margin': 'auto'},
)

# Callback to update the upload status and message
@app.callback(
    [Output('uploaded-message', 'children'),
     Output('upload-status', 'data')],
    [Input('upload-data', 'contents')]
)
def update_upload_status(contents):
    if contents is not None:
        uploaded_message = html.Div("Data uploaded successfully.", style={'color': 'green'})
        upload_status = {'uploaded': True}
    else:
        uploaded_message = html.Div("No data uploaded.", style={'color': 'red'})
        upload_status = {'uploaded': False}

    return uploaded_message, upload_status

# Callback to update the graphs based on user input
@app.callback(
    [Output('output-histogram', 'figure'),
     Output('output-scatter', 'figure'),
     Output('output-pie', 'figure')],
    [Input('analyze-button', 'n_clicks')],
    [State('upload-status', 'data')]
)
def update_graphs(analyze_clicks, upload_status):
    # Check if data is uploaded
    if not upload_status['uploaded']:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Check if 'Asthma' column is present in the DataFrame
    if 'Asthma' not in df.columns:
        return html.Div("Error: 'Asthma' column not found in the dataset."), dash.no_update, dash.no_update
    
    # Features and target variable
    X = df.drop('Asthma', axis=1)
    y = df['Asthma']

    # Convert 'Smoking' to one-hot encoding
    X = pd.get_dummies(X, columns=['Smoking'], drop_first=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Display histogram
    fig_hist = px.histogram(df, x='Asthma', color='Asthma', labels={'Asthma': 'Asthma'},
                            title='Asthma Histogram', width=800, height=400)
    
    # Customize histogram layout
    fig_hist.update_layout(
        bargap=0.1,
        barmode='overlay',
        xaxis_title='Asthma',
        yaxis_title='Count',
        template='plotly'  # Change template to 'plotly' for light theme
    )

    # Display scatter plot
    fig_scatter = px.scatter(df, x='Age', y='Asthma', color='Asthma',
                             title='Scatter Plot', width=800, height=400)
    
    # Customize scatter plot layout
    fig_scatter.update_layout(
        xaxis_title='Age',
        yaxis_title='Asthma',
        template='plotly'  # Change template to 'plotly' for light theme
    )

    # Display pie chart
    fig_pie = px.pie(df, names='Smoking', title='Smoking Distribution', width=800, height=400)
    
    # Customize pie chart layout
    fig_pie.update_layout(
        template='plotly'  # Change template to 'plotly' for light theme
    )

    return fig_hist, fig_scatter, fig_pie

# Callback to update the prediction result
@app.callback(
    Output('prediction-result', 'children'),
    [Input('analyze-button', 'n_clicks')],
    [State('upload-status', 'data')]
)
def update_prediction_result(analyze_clicks, upload_status):
    # Check if data is uploaded
    if not upload_status['uploaded']:
        return dash.no_update
    
    # Check if 'Asthma' column is present in the DataFrame
    if 'Asthma' not in df.columns:
        return html.Div("Error: 'Asthma' column not found in the dataset.", style={'color': 'red'})

    # Features and target variable
    X = df.drop('Asthma', axis=1)
    y = df['Asthma']

    # Convert 'Smoking' to one-hot encoding
    X = pd.get_dummies(X, columns=['Smoking'], drop_first=True)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Make prediction for a new data point (you can replace this with your actual data)
    new_data_point = {'Age': [28], 'Smoking_Yes': [1]}
    new_data_df = pd.DataFrame(new_data_point)
    prediction = model.predict(new_data_df)
    probabilities = model.predict_proba(new_data_df)

    # Display the result
    result_text = f"Prediction: {'Asthma' if prediction[0] == 1 else 'No Asthma'}\n" 
    result_text += f"\nProbability of Asthma: {probabilities[0][1]*100:.2f}%"
    
    return dcc.Markdown(result_text)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
