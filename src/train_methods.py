import inspect
import sys
from src.model_tools import train_step
import torch.nn.functional as F


def get_training_method(name):
    """Looks for a training method function and returns it if it exists in this module.

    Args:
        name (str): name of the training method function.

    Raises:
        ModuleNotFoundError: if the function does not exist, this exception is raised.

    Returns:
        function: the required training method function.
    """
    # Find the requested model by name
    cls_members = dict(inspect.getmembers(sys.modules[__name__]))
    if name not in cls_members:
        raise ModuleNotFoundError(f"Function {name} not found in module {__name__}")
    method = cls_members[name]

    return method


def regular(net, criterion, optimizer, inputs, target):
    net.reset_masks()
    loss = train_step(
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        inputs=inputs,
        target=target,
        mask=True,
    )
    return loss


def step(net, criterion, optimizer, inputs, target):
    # Train step with dropout
    loss = regular(
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        inputs=inputs,
        target=target,
    )
    # Train step with inverted dropout
    net.invert_masks()
    loss = train_step(
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        inputs=inputs,
        target=target,
        mask=True,
    )
    return loss


def distillation(net, criterion, optimizer, inputs, target):
    # Train step with dropout
    net.reset_masks()
    loss = train_step(
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        inputs=inputs,
        target=target,
        mask=True,
    )
    # Train step with inverted dropout
    soft_target = net(inputs, mask=True).detach()
    soft_target = F.softmax(soft_target, dim=1)
    net.invert_masks()
    loss = train_step(
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        inputs=inputs,
        target=soft_target,
        mask=True,
    )
    return loss


def fulldistillation(net, criterion, optimizer, inputs, target):
    # Train step with dropout
    net.reset_masks()
    loss = train_step(
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        inputs=inputs,
        target=target,
        mask=True,
    )
    # Train step with inverted dropout
    soft_target = net(inputs, mask=True).detach()
    soft_target = F.softmax(soft_target, dim=1)
    loss = train_step(
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        inputs=inputs,
        target=soft_target,
        mask=False,
    )
    return loss
