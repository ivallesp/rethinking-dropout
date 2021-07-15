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
    cls_members_norm = {k.lower(): v for k, v in cls_members.items()}
    # If there has been a name collision, raise an exception
    if len(cls_members_norm) < len(cls_members):
        raise ValueError(
            "Some of the training methods have the same name when "
            "normalization is applied. This is not allowed."
        )
    # Find the requested method
    if name not in cls_members_norm:
        raise ModuleNotFoundError(f"Function {name} not found in module {__name__}")
    object = cls_members_norm[name]

    # If the requested method is a class, instantiate it
    if inspect.isclass(object):
        method = object()
    # If the requested method is a function, just return it
    else:
        method = object
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
    # Train step with inverted dropout on distilled target
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
    # Train step without dropout on distilled target
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


def fulldistillationreset(net, criterion, optimizer, inputs, target):
    # Train step with dropout
    net.reset_masks()
    state_dict = net.state_dict()
    opt_state_dict = optimizer.state_dict()
    loss = train_step(
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        inputs=inputs,
        target=target,
        mask=True,
    )
    # Undo train step
    soft_target = net(inputs, mask=True).detach()
    soft_target = F.softmax(soft_target, dim=1)
    net.load_state_dict(state_dict)
    optimizer.load_state_dict(opt_state_dict)
    # Train step without dropout on distilled target
    loss = train_step(
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        inputs=inputs,
        target=soft_target,
        mask=False,
    )
    return loss


class AlternateSteps:
    def __init__(self):
        self.state = True

    def toggle_state(self):
        self.state = not self.state

    def __call__(self, net, criterion, optimizer, inputs, target):
        # Train step with dropout
        if self.state:
            net.reset_masks()
        else:
            net.invert_masks()
        loss = train_step(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            inputs=inputs,
            target=target,
            mask=True,
        )
        self.toggle_state()
        return loss