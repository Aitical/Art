import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class Rebalance_L1(nn.Module):

    def __init__(self):
        super(Rebalance_L1, self).__init__()

    def forward(self, restored, cleanpatch, de_id, T, loss_list_last, loss_list_min):

        total_loss = 0
        loss_grad_dict = [
            torch.tensor(-100000, device=restored.device) for i in range(3)
        ]
        weight_list_dict = [torch.tensor([0], device=restored.device) for i in range(3)]

        derain_list = []
        dehaze_list = []
        snow_list = []

        # T = 10
        snow_lq_list = []
        derain_lq_list = []
        dehaze_lq_list = []

        for p, t, img_type in zip(restored, cleanpatch, de_id):
            if img_type == 0:
                dehaze_lq_list.append(t)
                dehaze_list.append(p)
            elif img_type == 1:
                derain_lq_list.append(t)
                derain_list.append(p)
            elif img_type == 2:
                snow_lq_list.append(t)
                snow_list.append(p)

        snow_loss = 0
        if snow_lq_list != []:
            snow_loss = F.l1_loss(
                torch.stack(snow_lq_list),
                torch.stack(snow_list),
                reduction="mean",
            )
            loss_grad_dict[2] = (
                (snow_loss * torch.log10(loss_list_min[2]) / loss_list_last[2])
                / torch.log10(snow_loss)
            ) / T

        derain_loss = 0
        if derain_lq_list != []:
            derain_loss = F.l1_loss(
                torch.stack(derain_lq_list), torch.stack(derain_list), reduction="mean"
            )
            loss_grad_dict[1] = (
                (derain_loss * torch.log10(loss_list_min[1]) / loss_list_last[1])
                / torch.log10(derain_loss)
            ) / T

        dehaze_loss = 0
        if dehaze_lq_list != []:
            dehaze_loss = F.l1_loss(
                torch.stack(dehaze_lq_list), torch.stack(dehaze_list), reduction="mean"
            )
            loss_grad_dict[0] = (
                (dehaze_loss * torch.log10(loss_list_min[0]) / loss_list_last[0])
                / torch.log10(dehaze_loss)
            ) / T

        weight_list = torch.softmax(torch.tensor(loss_grad_dict), dim=0).to(
            restored.device
        )

        weight_list_dict[0] = weight_list[0].detach()
        weight_list_dict[1] = weight_list[1].detach()
        weight_list_dict[2] = weight_list[2].detach()

        total_loss += (
            dehaze_loss * weight_list[0]
            + derain_loss * weight_list[1]
            + snow_loss * weight_list[2]
        )

        my_zero = torch.tensor(0, device=restored.device)
        if snow_lq_list == []:
            loss_grad_dict[2] = my_zero
        if derain_lq_list == []:
            loss_grad_dict[1] = my_zero
        if dehaze_lq_list == []:
            loss_grad_dict[0] = my_zero

        return (
            total_loss,
            [
                dehaze_loss,
                derain_loss,
                snow_loss,
            ],
            loss_grad_dict,
            weight_list_dict,
        )
