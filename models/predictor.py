# models/predictor.py
import numpy as np

class DummyPredictor:
    def predict(self, observation, action):
        # Apply simple transformation to simulate prediction
        keypoints = observation + action
        tactile = np.random.rand(4, 4)  # Simulated 4x4 tactile map
        return keypoints, tactile

