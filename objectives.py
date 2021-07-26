import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

import copy
import numpy as np
from collections import defaultdict

OBJECTIVES = [
    'ERM',
    'IRM',
    'VREx',
    'SD',
    'ANDMask',
    'SANDMask',
    'IGA'
]

def get_objective_class(objective_name):
    """Return the objective class with the given name."""
    if objective_name not in globals():
        raise NotImplementedError("objective not found: {}".format(objective_name))
    return globals()[objective_name]


class Objective(torch.nn.Module):
    """
    A subclass of Objective implements a domain generalization Gradients.
    Subclasses should implement the following:
    - gradients()
    """
    def __init__(self, hparams):
        super(Objective, self).__init__()

        self.hparams = hparams

    def backward(self, losses):
        """
        Computes the Gradients for model update

        Admits a list of unlabeled losses from the test domains: losses
        """
        raise NotImplementedError

class ERM(Objective):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, model, hparams):
        super(ERM, self).__init__(hparams)

        self.model = model

    def gather_logits_and_labels(self, logits, labels):
        pass

    def backward(self, losses):

        objective = losses.mean()
        objective.backward()

class IRM(ERM):
    """
    Invariant Risk Minimization (IRM)
    """

    def __init__(self, model, hparams):
        super(IRM, self).__init__(model, hparams)

        # Hyper parameters
        self.penalty_weight = self.hparams['penalty_weight']
        self.anneal_iters = self.hparams['anneal_iters']

        # Memory
        self.penalty = 0
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def gather_logits_and_labels(self, logits, labels):
        self.penalty += self._irm_penalty(logits, labels)

    def backward(self, losses):
        # Define stuff
        penalty_weight = (self.penalty_weight   if self.update_count >= self.anneal_iters 
                                                else 1.0)

        # Compute objective
        n_env = losses.shape[0]
        loss = losses.mean()
        penalty = self.penalty / n_env

        objective = loss + (penalty_weight * penalty)

        # Backpropagate
        objective.backward()

        # Update memory
        self.update_count += 1
        self.penalty = 0


class VREx(ERM):
    """
    V-REx Objective from http://arxiv.org/abs/2003.00688
    """
    def __init__(self, model, hparams):
        super(VREx, self).__init__(model, hparams)

        # Hyper parameters
        self.penalty_weight = self.hparams['penalty_weight']
        self.anneal_iters = self.hparams['anneal_iters']

        # Memory
        self.register_buffer('update_count', torch.tensor([0]))

    def backward(self, losses):
        
        # Define stuff
        penalty_weight = (self.penalty_weight   if self.update_count >= self.anneal_iters 
                                                else 1.0)

        # Compute objective
        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        objective = mean + penalty_weight * penalty

        objective.backward()

        # Update memory
        self.update_count += 1

class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """
    def __init__(self, model, hparams):
        super(SD, self).__init__(model, hparams)

        # Hyper parameters
        self.penalty_weight = self.hparams['penalty_weight']

        # Memory
        self.penalty = 0

    def gather_logits_and_labels(self, logits, labels):

        self.penalty += (logits ** 2).mean()

    def backward(self, losses):

        # Compute Objective
        n_env = losses.shape[0]
        loss = losses.mean()
        penalty = self.penalty / n_env
        objective = loss + self.penalty_weight * penalty

        objective.backward()

        # Update memory
        self.penalty = 0

class ANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """

    def __init__(self, model, hparams):
        super(ANDMask, self).__init__(model, hparams)

        # Hyper parameters
        self.tau = self.hparams['tau']

    def backward(self, losses):
        
        param_gradients = [[] for _ in self.model.parameters()]
        for env_loss in losses:

            env_grads = autograd.grad(env_loss, self.model.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)
            
        mean_loss = losses.mean()

        self.mask_grads(self.tau, param_gradients, self.model.parameters())

    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

class IGA(ERM):
    """
    Inter-environmental Gradient Alignment
    From https://arxiv.org/abs/2008.01883v2
    """

    def __init__(self, model, hparams):
        super(IGA, self).__init__(model, hparams)

        # Hyper parameters
        self.penalty_weight = self.hparams['penalty_weight']

    def backward(self, losses):

        # Get the gradients
        grads = []
        for env_loss in losses:

            env_grad = autograd.grad(env_loss, self.model.parameters(), 
                                        create_graph=True)

            grads.append(env_grad)
            
        # Compute the mean loss and mean loss gradient
        mean_loss = losses.mean()
        mean_grad = autograd.grad(mean_loss, self.model.parameters(), 
                                        create_graph=True)

        # compute trace penalty
        penalty_value = 0
        for grad in grads:
            for g, mean_g in zip(grad, mean_grad):
                penalty_value += (g - mean_g).pow(2).sum()

        objective = mean_loss + self.penalty_weight * penalty_value

        # Backpropagate
        objective.backward()
    
class SANDMask(ERM):
    """
    SAND-mask: An Enhanced Gradient Masking Strategy for the Discovery of Invariances in Domain Generalization
    <https://arxiv.org/abs/2106.02266>
    """

    def __init__(self, model, hparams):
        super(SANDMask, self).__init__(model, hparams)

        self.tau = self.hparams['tau']
        self.k = self.hparams['k']

    def backward(self, losses):

        # Get environment gradients
        param_gradients = [[] for _ in self.model.parameters()]
        for env_loss in losses:

            env_grads = autograd.grad(env_loss, self.model.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        # Get mean loss
        mean_loss = losses.mean()

        # Backpropagate
        self.mask_grads(param_gradients, self.model.parameters())

    def mask_grads(self, gradients, params):
        '''
        Here a mask with continuous values in the range [0,1] is formed to control the amount of update for each
        parameter based on the agreement of gradients coming from different environments.
        '''
        device = gradients[0][0].device
        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            avg_grad = torch.mean(grads, dim=0)
            grad_signs = torch.sign(grads)
            gamma = torch.tensor(1.0).to(device)
            grads_var = grads.var(dim=0)
            grads_var[torch.isnan(grads_var)] = 1e-17
            lam = (gamma * grads_var).pow(-1)
            mask = torch.tanh(self.k * lam * (torch.abs(grad_signs.mean(dim=0)) - self.tau))
            mask = torch.max(mask, torch.zeros_like(mask))
            mask[torch.isnan(mask)] = 1e-17
            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

