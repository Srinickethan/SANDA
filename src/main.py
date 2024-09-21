'''
    Main entry point for the FastAPI app.
'''

# Import the required libraries
from fastapi import FastAPI
from src.predict.service import Prediction
from src.constants import MODEL_PATH

# Define the FastAPI app
app = FastAPI()

# Load the trained model
model = Prediction(MODEL_PATH)

# Define a route to make predictions
@app.get("/predict/")
def predict(input_text: str) -> dict:
    '''
    Make predictions using the trained model.
    '''
    # Make predictions
    predictions = model.prediction(input_text)
    return predictions

#To run the FastAPI app, execute the following command in the terminal:
# uvicorn src.main:app --reload
