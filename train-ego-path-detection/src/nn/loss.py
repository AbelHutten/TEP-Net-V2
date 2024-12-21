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


class GIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        """
        Computes the Generalized IoU loss between predicted and target rail positions,
        incorporating the y-limit as the vertical bound of the bounding box.

        Args:
            prediction (torch.Tensor): Predicted values containing trajectory and ylim.
            target (tuple): Tuple containing (traj_target, ylim_target).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        traj_target, ylim_target = target
        traj_prediction = prediction[:, :-1].view_as(traj_target)
        ylim_prediction = prediction[:, -1]

        # Mask for anchors within the y-limit
        ylim_target_idx = (ylim_target * (traj_target.size(2) - 1)).long()  # (B,)
        range_matrix = torch.arange(
            traj_target.size(2), device=ylim_target_idx.device
        ).expand(traj_target.size(0), -1)  # (B, H)
        loss_mask = (range_matrix <= ylim_target_idx.unsqueeze(1)).float()  # (B, H)

        # Calculate the intersection
        left_intersection = torch.maximum(
            traj_prediction[:, 0, :], traj_target[:, 0, :]
        )
        right_intersection = torch.minimum(
            traj_prediction[:, 1, :], traj_target[:, 1, :]
        )
        intersection_width = torch.clamp(right_intersection - left_intersection, min=0)

        # Only consider intersections within the y-limit
        intersection_height = torch.minimum(
            ylim_prediction.unsqueeze(1), ylim_target.unsqueeze(1)
        )  # (B, 1)
        intersection = (intersection_width * intersection_height * loss_mask).sum(
            dim=1
        )  # (B,)

        # Calculate the union
        left_union = torch.minimum(traj_prediction[:, 0, :], traj_target[:, 0, :])
        right_union = torch.maximum(traj_prediction[:, 1, :], traj_target[:, 1, :])
        union_width = torch.clamp(right_union - left_union, min=0)
        union_height = torch.maximum(
            ylim_prediction.unsqueeze(1), ylim_target.unsqueeze(1)
        )  # (B, 1)
        union = (union_width * union_height * loss_mask).sum(dim=1)  # (B,)

        # IoU calculation
        iou = intersection / (union + 1e-6)

        # Calculate the smallest enclosing box
        enclosing_left = torch.minimum(traj_prediction[:, 0, :], traj_target[:, 0, :])
        enclosing_right = torch.maximum(traj_prediction[:, 1, :], traj_target[:, 1, :])
        enclosing_width = torch.clamp(enclosing_right - enclosing_left, min=0)
        enclosing_height = torch.maximum(
            ylim_prediction.unsqueeze(1), ylim_target.unsqueeze(1)
        )
        enclosing_area = (enclosing_width * enclosing_height * loss_mask).sum(
            dim=1
        )  # (B,)

        # Generalized IoU calculation
        giou = iou - ((enclosing_area - union) / (enclosing_area + 1e-6))

        # GIoU loss
        giou_loss = 1 - giou.mean()

        return giou_loss


# Example usage
if __name__ == "__main__":
    # Example tensors with batch size B=2, 2 rails, and H=3 anchors
    traj_target = torch.tensor([
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]],
    ])
    ylim_target = torch.tensor([0.2, 0.9])
    target = (traj_target, ylim_target)

    traj_prediction = torch.tensor([
        [[0.15, 0.25, 0.35], [0.45, 0.55, 0.65]],
        [[0.25, 0.35, 0.45], [0.55, 0.65, 0.75]],
    ])
    ylim_prediction = torch.tensor([0.85, 0.95])
    prediction = torch.cat(
        [traj_prediction.view(2, -1), ylim_prediction.unsqueeze(1)], dim=1
    )

    criterion = GIoULoss()
    loss = criterion(prediction, target)
    print("GIoU Loss:", loss.item())
