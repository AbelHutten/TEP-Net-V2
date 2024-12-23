import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, prediction, target):
        target = target.flatten()
        prediction = prediction.view(target.shape[0], -1)
        return self.loss(prediction, target)


class TrainEgoPathRegressionLoss(nn.Module):
    def __init__(self, ylimit_loss_weight, perspective_weight_limit=None):
        super(TrainEgoPathRegressionLoss, self).__init__()
        self.ylimit_loss_weight = ylimit_loss_weight
        self.perspective_weight_limit = perspective_weight_limit
        self.unreduced_smae = nn.SmoothL1Loss(reduction="none", beta=0.005)
        self.batchaveraged_smae = nn.SmoothL1Loss(reduction="mean", beta=0.015)

    def trajectory_loss(self, traj_prediction, traj_target, ylim_target):
        traj_se = self.unreduced_smae(traj_prediction, traj_target)  # (B, 2, H)
        ylim_target_idx = ylim_target * (traj_target.size(2) - 1)  # (B,)
        range_matrix = torch.arange(
            traj_target.size(2), device=ylim_target_idx.device
        ).expand(traj_target.size(0), -1)  # (B, H)
        loss_mask = (range_matrix <= ylim_target_idx.unsqueeze(1)).float()  # (B, H)
        rail_width = traj_target[:, 1, :] - traj_target[:, 0, :]  # (B, H)
        weights = (loss_mask / rail_width).unsqueeze(dim=1)  # (B, 1, H)
        if self.perspective_weight_limit is not None:
            weights = torch.clamp(weights, max=self.perspective_weight_limit)
        traj_loss = (traj_se * weights).sum(dim=(1, 2)) / loss_mask.sum(dim=1)
        mask = ylim_target == 0
        traj_loss[mask] = 0
        return traj_loss.mean()  # ()

    def ylim_loss(self, ylim_prediction, ylim_target):
        return self.batchaveraged_smae(torch.sigmoid(ylim_prediction), ylim_target)

    def forward(self, prediction, target):
        traj_target, ylim_target = target
        traj_prediction = prediction[:, :-1].view_as(traj_target)
        ylim_prediction = prediction[:, -1]
        traj_loss = self.trajectory_loss(traj_prediction, traj_target, ylim_target)
        ylim_loss = self.ylim_loss(ylim_prediction, ylim_target)
        combined_loss = traj_loss + self.ylimit_loss_weight * ylim_loss
        return combined_loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, prediction, target):
        prediction = torch.sigmoid(prediction)
        prediction = prediction.flatten(start_dim=1)  # (B, H * W)
        target = target.flatten(start_dim=1)
        zero_target_mask = target.sum(dim=1) == 0  # (B,)
        if zero_target_mask.any():
            prediction[zero_target_mask] = 1 - prediction[zero_target_mask]
            target[zero_target_mask] = 1 - target[zero_target_mask]
        intersection = (prediction * target).sum(dim=1)  # (B,)
        cardinality = (prediction + target).sum(dim=1)
        scores = 2 * intersection / cardinality
        return 1 - scores.mean()  # ()


class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        """
        Computes the IoU loss between predicted and target rail positions.
        """
        traj_target, ylim_target = target
        traj_prediction = prediction[:, :-1].view_as(traj_target)
        ylim_prediction = prediction[:, -1]
        ylim_prediction = torch.sigmoid(ylim_prediction)
        ylim_target_idx = ylim_target * (traj_target.size(2) - 1)  # (B,)
        ylim_prediction_idx = ylim_prediction * (traj_target.size(2) - 1)  # (B,)
        left_rail_max = torch.maximum(traj_prediction[:, 0, :], traj_target[:, 0, :])
        right_rail_min = torch.minimum(traj_prediction[:, 1, :], traj_target[:, 1, :])
        batch_size, _, anchor_count = traj_prediction.shape
        range_matrix = (
            torch.arange(anchor_count, device=ylim_prediction.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        prediction_mask = range_matrix < ylim_prediction_idx.unsqueeze(1)
        target_mask = range_matrix < ylim_target_idx.unsqueeze(1)
        min_mask = prediction_mask * target_mask
        intersection = (
            torch.clamp((right_rail_min - left_rail_max), min=0) * min_mask
        ).sum(dim=1)
        pred_area = (
            torch.clamp((traj_prediction[:, 1, :] - traj_prediction[:, 0, :]), min=0)
            * prediction_mask
        ).sum(dim=1)
        target_area = (
            torch.clamp((traj_target[:, 1, :] - traj_target[:, 0, :]), min=0)
            * target_mask
        ).sum(dim=1)
        union = pred_area + target_area - intersection
        return 1 - (intersection / union).mean()


class Mean1DGIoULoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.batchaveraged_smae = nn.SmoothL1Loss(reduction="mean", beta=0.015)
        self.ylim_loss_weight = 0.5

    def forward(self, prediction, target):
        traj_target, ylim_target = target
        traj_prediction = prediction[:, :-1].view_as(traj_target)
        ylim_prediction = prediction[:, -1]
        ylim_target_idx = ylim_target * (traj_target.size(2) - 1)  # (B,)
        range_matrix = torch.arange(
            traj_target.size(2), device=ylim_target_idx.device
        ).expand(traj_target.size(0), -1)  # (B, H)
        loss_mask = range_matrix <= ylim_target_idx.unsqueeze(1).float()  # (B, H)
        a2 = traj_prediction[:, 1, :]
        b2 = traj_target[:, 1, :]
        a1 = traj_prediction[:, 0, :]
        b1 = traj_target[:, 0, :]
        intersection = torch.clamp(torch.minimum(a2, b2) - torch.maximum(a1, b1), min=0)
        union = (a2 - a1) + (b2 - b1) - intersection
        iou = intersection / union
        c = torch.maximum(a2, b2) - torch.minimum(a1, b1)
        giou = iou - (c - union) / (c + 1e-6)
        giou = giou * loss_mask
        return (
            1
            - giou.mean()
            + self.ylim_loss_weight
            * self.batchaveraged_smae(torch.sigmoid(ylim_prediction), ylim_target)
        )
