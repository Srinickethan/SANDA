'''
This file contains the Prediction class which is used to make predictions using the trained model.
'''
import pickle
import numpy as np

class Prediction():
    '''
    Class to define the input data structure for making predictions.
    '''
    def __init__(self, model_path: str):
        '''
        Load the trained model when the class is initialized.
        '''

        self.load_model(model_path)


    def load_model(self, model_path):
        '''
        Load the trained model.
        '''
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)


    def prediction(self, input_text: str) -> dict:
        '''
        Make predictions using the model.
        Currently we only have kmeans model.
        '''
        embeddings = self.preprocess_text_to_embeddings(input_text)
        # Make predictions using the model
        if embeddings.size != 0:
            cluster_label = self.model.predict(embeddings.reshape(1, -1))
            return {"Predicted Cluster": int(cluster_label[0])}
        return {"Error": "No valid embeddings found for input text"}


    def preprocess_text_to_embeddings(self, input_text: str) -> np.array:
        '''
        Example of converting text to embeddings, 
        modify this based on your exact model's preprocessing.
        Assume you're using BERT-like embeddings or Word2Vec embeddings.
        '''
        # Dummy embedding, replace with actual model processing
        print(f"Generating embeddings for input text: {input_text}")
        embedding_vector = np.random.rand(768)
        embedding_vector = embedding_vector.astype(np.double)
        return embedding_vector
        