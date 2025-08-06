import torch
from src.ik import IKDecompose
from src.segmentation import SegmentPointCloud, ClassifyObjectCategories
from src.forward_model import ForwardModel
from src.grasp_classifier import GraspClassifier
from src.object_classifier import ObjectClassifier
from src.utils import get_dummy_scene


def main():
    # Simulate high-level motion, initial observation and joint config
    M, o_0, q_0 = get_dummy_scene()
    actions = IKDecompose(M)

    # Scene preprocessing
    segments = SegmentPointCloud(o_0)
    category_map = ClassifyObjectCategories(segments)
    o_prev = torch.cat([o_0, category_map], dim=-1)
    q_prev = q_0

    # Instantiate models
    output_dim = 32
    fwd_model = ForwardModel(obs_dim=o_prev.shape[-1], action_dim=q_prev.shape[-1], q_dim=q_prev.shape[-1], output_dim=output_dim)
    grasp_cls = GraspClassifier(tactile_dim=output_dim)
    obj_cls = ObjectClassifier(input_dim=output_dim)

    # Run pipeline
    for step, a_t in enumerate(actions):
        K_t = fwd_model(o_prev, q_prev, a_t)
        T_t = K_t.clone()  # placeholder tactile = predicted state
        c_t = grasp_cls(T_t)
        o_logits = obj_cls(T_t, K_t)

        print(f"Step {step+1}:")
        print(" Predicted Keypoints:", K_t[0].tolist())
        print(" Grasp logit:", c_t.item())
        print(" Object category logits:", o_logits[0].tolist())

                # update for next step
        o_prev = K_t  # use predicted keypoints as next observation
        q_prev = q_prev  # keep joint config unchanged (or update with IK integration as needed)
if __name__ == '__main__':
    main()