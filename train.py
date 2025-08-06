# train.py
import torch
from torch.optim import Adam
from src.forward_model import ForwardModel
from src.grasp_classifier import GraspClassifier
from src.object_classifier import ObjectClassifier
from src.losses import compute_losses
from src.utils import get_training_batch


def train(num_epochs=5, lr=1e-3):
    # Initialize models
    fwd_model = ForwardModel()
    grasp_cls = GraspClassifier()
    obj_cls = ObjectClassifier()

    optimizer = Adam(
        list(fwd_model.parameters()) +
        list(grasp_cls.parameters()) +
        list(obj_cls.parameters()), lr=lr)

    for epoch in range(num_epochs):
        obs, q, action, gt_kp, gt_tactile, gt_grasp, gt_obj = get_training_batch()

        pred_kp = fwd_model(obs, q, action)
        pred_tactile = pred_kp.clone()
        pred_grasp = grasp_cls(pred_tactile)
        pred_obj = obj_cls(pred_tactile, pred_kp)

        loss = compute_losses(
            pred_kp, pred_tactile, pred_grasp, pred_obj,
            gt_kp, gt_tactile, gt_grasp, gt_obj)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

if __name__ == '__main__':
    train()

