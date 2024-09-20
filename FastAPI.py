from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Define the FastAPI app
app = FastAPI()

# Load the trained model
with open('kmeans_model_older.pkl', 'rb') as model_file:
    kmeans_model = pickle.load(model_file)

# Define the input data structure
class PredictionInput(BaseModel):
    input_text: str

# Define a route to make predictions
@app.post("/predict/")
def predict(input_data: PredictionInput):
    input_text = input_data.input_text
    # Assuming you have a function to preprocess the input text and generate embeddings
    embeddings = preprocess_text_to_embeddings(input_text)
    
    # Make predictions using the model
    if embeddings.size != 0:
        cluster_label = kmeans_model.predict(embeddings.reshape(1, -1))
        return {"Predicted Cluster": int(cluster_label[0])}
    else:
        return {"Error": "No valid embeddings found for input text"}

# You can define the preprocessing function used above (adjust it as per your model)
def preprocess_text_to_embeddings(text):
    # Example of converting text to embeddings, modify this based on your exact model's preprocessing
    # Assume you're using BERT-like embeddings or Word2Vec embeddings
    embedding_vector = np.random.rand(768)  # Dummy embedding, replace with actual model processing
    return embedding_vector

