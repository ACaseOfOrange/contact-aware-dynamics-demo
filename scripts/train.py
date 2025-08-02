# scripts/train.py
import numpy as np
from models.predictor import DummyPredictor
from classifiers.grasp_classifier import SimpleGraspClassifier
from utils.ik_decompose import decompose_motion

# Mock input data
initial_observation = np.random.rand(10, 3)  # 10 keypoints in 3D
motion_plan = np.array([[0.1, 0, 0], [0.1, 0, 0], [0.1, 0, 0]])  # simple motion in x

# Decompose motion into sub-actions
actions = decompose_motion(motion_plan)

# Initialize model and classifier
predictor = DummyPredictor()
classifier = SimpleGraspClassifier()

print("Initial Observation:", initial_observation)

# Run prediction loop
for step, action in enumerate(actions):
    pred_keypoints, pred_tactile = predictor.predict(initial_observation, action)
    grasped = classifier.predict(pred_tactile)
    print(f"Step {step+1}:")
    print("Predicted Keypoints:", pred_keypoints)
    print("Predicted Tactile:", pred_tactile)
    print("Grasped:", grasped)

    initial_observation = pred_keypoints  # update for next step

