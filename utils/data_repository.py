import os
import pickle

class DataRepository:
    def __init__(self, default_path):
        self.default_path = default_path
    
    def load_data(self, filename):
        file_path = os.path.join(self.default_path, filename)
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def save_data(self, data, filename):
        file_path = os.path.join(self.default_path, filename)
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)