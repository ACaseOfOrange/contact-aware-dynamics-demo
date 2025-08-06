import torch.nn.functional as F

def compute_losses(pred_kp, pred_tactile, pred_grasp, pred_obj,
                   gt_kp, gt_tactile, gt_grasp, gt_obj,
                   lambda_kp=1.0, lambda_tact=1.0, lambda_hold=1.0, lambda_obj=1.0):
    L_kp = F.mse_loss(pred_kp, gt_kp)
    L_tact = F.l1_loss(pred_tactile, gt_tactile)
    L_hold = F.binary_cross_entropy_with_logits(pred_grasp, gt_grasp)
    L_obj = F.cross_entropy(pred_obj, gt_obj)
    return lambda_kp * L_kp + lambda_tact * L_tact + lambda_hold * L_hold + lambda_obj * L_obj
