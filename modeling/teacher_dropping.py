import random
import torch


def aggregate_losses(loss_dict, drop_prob=0.5):
    """
    Given a dictionary of losses, which are expected to be 2D tensors of shape (B, N),
    where B is the batch size and N is the number of tokens, this function aggregates
    the losses into a single loss tensor.
    """
    sorted_keys = sorted(loss_dict.keys())

    B = next(iter(loss_dict.values())).shape[0]
    loss_list = torch.stack([loss_dict[key] for key in sorted_keys])
    assert loss_list.shape == (len(loss_dict), B)

    coeffs = torch.stack(
        [
            _get_teacher_coefficients_by_loss(loss_list[:, lix], drop_prob=drop_prob)
            for lix in range(loss_list.shape[1])
        ]
    ).t()
    assert coeffs.shape == loss_list.shape

    # make sure each image is assigned at least one teacher
    assert torch.all(coeffs.sum(dim=0) >= 1)

    #####
    # normalize coefficients such that
    # each image contributes to the loss with equal weight
    coeffs.div_(coeffs.sum())
    loss = (coeffs.clone().detach() * loss_list).sum()

    coeffs = coeffs.sum(dim=1)  # for logging teacher coefficients
    coeffs = {key: coeff for key, coeff in zip(sorted_keys, coeffs)}

    return loss, coeffs


@torch.no_grad()
def _get_teacher_coefficients_by_loss(losses, drop_prob=0.5):
    """
    Given a list of losses from all teachers, return a list for their loss coefficients.
    Initially, all coefficients are 1.
    Then we flip coefficients for teachers with lowest loss to zeros with a probability drop_prob.
    """
    if isinstance(losses, (list, tuple)):
        losses = torch.stack(losses)

    # make sure that losses are 1D
    assert len(losses.shape) == 1

    coeffs = torch.ones_like(losses, requires_grad=False)

    # find the teacher with the highest loss
    max_loss_idx = torch.argmax(losses)

    # go through other teachers and
    # flip their coefficients to zeros with a probability drop_prob
    for i in range(len(losses)):
        if i != max_loss_idx:
            p = random.random()
            if p < drop_prob:
                coeffs[i] = 0

    return coeffs
