import torch

def get_dummy_scene():
    M = None
    o_0 = torch.randn(1, 36)  # obs + category dims
    q_0 = torch.randn(1, 8)
    return M, o_0, q_0


def get_training_batch(batch_size=4):
    obs = torch.randn(batch_size, 36)
    q = torch.randn(batch_size, 8)
    action = torch.randn(batch_size, 8)
    gt_kp = torch.randn(batch_size, 32)
    gt_tactile = torch.randn(batch_size, 32)
    gt_grasp = torch.rand(batch_size, 1)
    gt_obj = torch.randint(0, 4, (batch_size,))
    return obs, q, action, gt_kp, gt_tactile, gt_grasp, gt_obj