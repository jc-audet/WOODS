
"""Defining domain generalization algorithms"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy

import matplotlib.pyplot as plt

from woods.models import LSTM, MNIST_LSTM

OBJECTIVES = [
    'ERM',
    'GroupDRO',
    'IRM',
    'VREx',
    'SD',
    # 'ANDMask', # Requires update
    # 'IGA', # Requires update
    # 'Fish', # Requires update
    'IB_ERM',
    # 'IB_IRM' # Requires update
    'CAD',
    'CondCAD',
    'Transfer'
]

def get_objective_class(objective_name):
    """Return the objective class with the given name."""
    if objective_name not in globals():
        raise NotImplementedError("objective not found: {}".format(objective_name))
    return globals()[objective_name]

class Objective(nn.Module):
    """
    A subclass of Objective implements a domain generalization Gradients.
    Subclasses should implement the following:
    - update
    - predict
    """
    def __init__(self, hparams):
        super(Objective, self).__init__()

        self.hparams = hparams

    def predict(self, all_x):
        raise NotImplementedError

    def update(self, losses):
        """
        Computes the Gradients for model update

        Admits a list of unlabeled losses from the test domains: losses
        """
        raise NotImplementedError

class ERM(Objective):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, model, dataset, optimizer, hparams):
        super(ERM, self).__init__(hparams)

        # Save hparams
        self.device = self.hparams['device']

        # Save training components
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer

        # Get some other useful info
        self.nb_training_domains = dataset.get_nb_training_domains()

    def predict(self, all_x):
        return self.model(all_x)

    def update(self):

        # Put model into training mode
        self.model.train()

        # Get next batch
        X, Y = self.dataset.get_next_batch()

        # Split into input / target
        # X, Y = self.dataset.split_input(batch)
        # print("input shape:", X.shape)

        # Get predict and get (logit, features)
        out, _ = self.predict(X)
        # # print("output shape", out.shape)

        # Compute mean loss
        domain_losses = self.dataset.loss_by_domain(out, Y, self.nb_training_domains)
        # # print("domain_losses shape: ", domain_losses.shape)

        # Compute objective
        objective = domain_losses.mean()
        
        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

class GroupDRO(ERM):
    """
    GroupDRO
    """

    def __init__(self, model, dataset, optimizer, hparams):
        super(GroupDRO, self).__init__(model, dataset, optimizer, hparams)

        # Save hparams
        self.device = self.hparams['device']
        self.eta = hparams['eta']
        self.register_buffer("q", torch.Tensor())

        # Save training components
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer

        # Get some other useful info
        self.nb_training_domains = dataset.get_nb_training_domains()

    def predict(self, all_x):
        return self.model(all_x)

    def update(self):

        # Put model into training mode
        self.model.train()

        # Get next batch
        X, Y = self.dataset.get_next_batch()

        if not len(self.q):
            print("hello, creating Q")
            self.q = torch.ones(self.nb_training_domains).to(self.device)

        # Split input / target
        # X, Y = self.dataset.split_input(batch)

        # Get predict and get (logit, features)
        out, _ = self.predict(X)

        # Compute losses
        domain_losses = self.dataset.loss_by_domain(out, Y, self.nb_training_domains)

        # Update weights
        for dom_i, dom_loss in enumerate(domain_losses):
            self.q[dom_i] *= (self.eta * dom_loss.data).exp()
        self.q /= self.q.sum()

        # Compute objective
        objective = torch.dot(domain_losses, self.q)

        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

class IRM(ERM):
    """
    Invariant Risk Minimization (IRM)
    """

    def __init__(self, model, dataset, optimizer, hparams):
        super(IRM, self).__init__(model, dataset, optimizer, hparams)

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

    def update(self):

        # Define penalty value (Annealing)
        penalty_weight = (self.penalty_weight   if self.update_count >= self.anneal_iters 
                                                else 1.0)

        # Put model into training mode
        self.model.train()

        # Get next batch
        X, Y = self.dataset.get_next_batch()

        # Split input / target
        # X, Y = self.dataset.split_input(batch)

        # Get predict and get (logit, features)
        out, _ = self.predict(X)

        # Compute losses
        n_domains = self.dataset.get_nb_training_domains() 
        domain_losses = self.dataset.loss_by_domain(out, Y, n_domains)

        # Create domain dimension in tensors. 
        #   e.g. for source domains: (ENVS * batch_size, ...) -> (ENVS, batch_size, ...)
        #        for time domains: (batch_size, ENVS, ...) -> (ENVS, batch_size, ...)
        out, labels = self.dataset.split_tensor_by_domains(out, Y, n_domains)
        # env_labels = self.dataset.split_tensor_by_domains(n_domains, Y)

        # Compute loss and penalty for each domains
        irm_penalty = torch.zeros(n_domains).to(self.device)
        for i, (env_out, env_labels) in enumerate(zip(out, labels)):
            irm_penalty[i] += self._irm_penalty(env_out, env_labels)

        # Compute objective
        irm_penalty = irm_penalty.mean()
        # print(domain_losses.mean(), irm_penalty)
        objective = domain_losses.mean() + (penalty_weight * irm_penalty)

        # Reset Adam, because it doesn't like the sharp jump in gradient
        # magnitudes that happens at this step.
        if self.update_count == self.anneal_iters:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.optimizer.param_groups[0]['lr'],
                weight_decay=self.optimizer.param_groups[0]['weight_decay'])

        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        # Update memory
        self.update_count += 1

class VREx(ERM):
    """
    V-REx Objective from http://arxiv.org/abs/2003.00688
    """
    def __init__(self, model, dataset, optimizer, hparams):
        super(VREx, self).__init__(model, dataset, optimizer, hparams)

        # Hyper parameters
        self.penalty_weight = self.hparams['penalty_weight']
        self.anneal_iters = self.hparams['anneal_iters']

        # Memory
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self):

        # Define stuff
        penalty_weight = (self.penalty_weight   if self.update_count >= self.anneal_iters 
                                                else 1.0)

        # Put model into training mode
        self.model.train()

        # Get next batch
        X, Y = self.dataset.get_next_batch()

        # Split input / target
        # X, Y = self.dataset.split_input(batch)

        # Get predict and get (logit, features)
        out, _ = self.predict(X)

        # Compute losses
        n_domains = self.dataset.get_nb_training_domains()
        domain_losses = self.dataset.loss_by_domain(out, Y, n_domains)

        # Compute objective
        mean = domain_losses.mean()
        penalty = ((domain_losses - mean).pow(2)).mean()
        objective = mean + penalty_weight * penalty

        # Reset Adam, because it doesn't like the sharp jump in gradient
        # magnitudes that happens at this step.
        if self.update_count == self.anneal_iters:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.optimizer.param_groups[0]['lr'],
                weight_decay=self.optimizer.param_groups[0]['weight_decay'])

        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        # Update memory
        self.update_count += 1

class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """
    def __init__(self, model, dataset, optimizer, hparams):
        super(SD, self).__init__(model, dataset, optimizer, hparams)

        # Hyper parameters
        self.penalty_weight = self.hparams['penalty_weight']

    def update(self):

        # Put model into training mode
        self.model.train()

        # Get next batch
        X, Y = self.dataset.get_next_batch()

        # Split input / target
        # X, Y = self.dataset.split_input(env_batches)

        # Get predict and get (logit, features)
        out, _ = self.predict(X)

        # Compute losses
        n_domains = self.dataset.get_nb_training_domains()
        domain_losses = self.dataset.loss_by_domain(out, Y, n_domains)

        # Create domain dimension in tensors:
        #   e.g. for source domains: (ENVS * batch_size, ...) -> (ENVS, batch_size, ...)
        #        for time domains: (batch_size, ENVS, ...) -> (ENVS, batch_size, ...) 
        domain_out, _ = self.dataset.split_tensor_by_domains(out, Y, n_domains)

        # Compute loss for each environment
        sd_penalty = torch.pow(out, 2).sum(dim=-1)

        # sd_penalty = torch.zeros(env_out.shape[0]).to(env_out.device)
        # for i in range(env_out.shape[0]):
        #     for t_idx in range(env_out.shape[2]):     # Number of time steps
        #         sd_penalty[i] += (env_out[i, :, t_idx, :] ** 2).mean()

        sd_penalty = sd_penalty.mean()
        objective = domain_losses.mean() + self.penalty_weight * sd_penalty

        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

# class ANDMask(ERM):
#     """
#     Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
#     AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
#     """

#     def __init__(self, model, dataset, loss_fn, optimizer, hparams):
#         super(ANDMask, self).__init__(model, dataset, loss_fn, optimizer, hparams)

#         # Hyper parameters
#         self.tau = self.hparams['tau']

#     def mask_grads(self, tau, gradients, params):

#         for param, grads in zip(params, gradients):
#             grads = torch.stack(grads, dim=0)
#             grad_signs = torch.sign(grads)
#             mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
#             mask = mask.to(torch.float32)
#             avg_grad = torch.mean(grads, dim=0)

#             mask_t = (mask.sum() / mask.numel())
#             param.grad = mask * avg_grad
#             param.grad *= (1. / (1e-10 + mask_t))

#     def update(self, minibatches_device, dataset, device):

#         ## Group all inputs and send to device
#         all_x = torch.cat([x for x,y in minibatches_device]).to(device)
#         all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
#         # Get logit and make prediction on PRED_TIME
#         ts = torch.tensor(dataset.PRED_TIME).to(device)
#         out, _ = self.predict(all_x, ts, device)

#         # Split data in shape (n_train_envs, batch_size, len(PRED_TIME), num_classes)
#         out_split = dataset.split_output(out)
#         labels_split = dataset.split_labels(all_y)

#         # Compute loss for each environment 
#         env_losses = torch.zeros(out_split.shape[0]).to(device)
#         for i in range(out_split.shape[0]):
#             for t_idx in range(out_split.shape[2]):     # Number of time steps
#                 env_losses[i] += self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 

#         # Compute gradients for each env
#         param_gradients = [[] for _ in self.model.parameters()]
#         for env_loss in env_losses:

#             env_grads = autograd.grad(env_loss, self.model.parameters(), retain_graph=True)
#             for grads, env_grad in zip(param_gradients, env_grads):
#                 grads.append(env_grad)
            
#         # Back propagate
#         self.optimizer.zero_grad()
#         self.mask_grads(self.tau, param_gradients, self.model.parameters())
#         self.optimizer.step()

# class IGA(ERM):
#     """
#     Inter-environmental Gradient Alignment
#     From https://arxiv.org/abs/2008.01883v2
#     """

#     def __init__(self, model, dataset, optimizer, hparams):
#         super(IGA, self).__init__(model, dataset, optimizer, hparams)

#         # Hyper parameters
#         self.penalty_weight = self.hparams['penalty_weight']

#     def update(self):

#         # Put model into training mode
#         self.model.train()

#         # Get next batch
#         X, Y = self.dataset.get_next_batch()

#         # Split input / target
#         # X, Y = self.dataset.split_input(env_batches)

#         # There is an unimplemented feature of cudnn that makes it impossible to perform double backwards pass on the network
#         # This is a workaround to make it work proposed by pytorch, but I'm not sure if it's the right way to do it
#         with torch.backends.cudnn.flags(enabled=False):
#             out, _ = self.predict(X)

#         # Compute losses
#         batch_losses = self.dataset.loss(out, Y)

#         # Create domain dimension in tensors. 
#         #   e.g. for source domains: (ENVS * batch_size, ...) -> (ENVS, batch_size, ...)
#         env_losses = self.dataset.split_tensor_by_domains(len(env_batches), batch_losses)

#         # Get the gradients
#         grads = []
#         for env_loss in env_losses:

#             env_grad = autograd.grad(env_loss.mean(), [p for p in self.model.parameters() if p.requires_grad], 
#                                         create_graph=True)
#             grads.append(env_grad)

#         # Compute the mean loss and mean loss gradient
#         mean_loss = env_losses.mean()
#         mean_grad = autograd.grad(mean_loss, [p for p in self.model.parameters() if p.requires_grad], 
#                                         create_graph=True)

#         # compute trace penalty
#         penalty_value = 0
#         for grad in grads:
#             for g, mean_g in zip(grad, mean_grad):
#                 penalty_value += (g - mean_g).pow(2).sum()

#         objective = mean_loss + self.penalty_weight * penalty_value

#         # Back propagate
#         self.optimizer.zero_grad()
#         objective.backward()
#         self.optimizer.step()
        
# class Fish(ERM):
#     """
#     Implementation of Fish, as seen in Gradient Matching for Domain 
#     Generalization, Shi et al. 2021.
#     """

#     def __init__(self, model, dataset, loss_fn, optimizer, hparams):
#         super(Fish, self).__init__(model, dataset, loss_fn, optimizer, hparams)
        
#          # Hyper parameters
#         self.meta_lr = self.hparams['meta_lr']
        
#         self.model = model
#         self.loss_fn = loss_fn
#         self.optimizer = optimizer

#     def create_copy(self, device):
#         self.model_inner = self.model
#         self.optimizer_inner = self.optimizer
#         self.loss_fn_inner = self.loss_fn 

#     def update(self, minibatches_device, dataset, device):
        
#         self.create_copy(minibatches_device[0][0].device)

#         ## Group all inputs and send to device
#         all_x = torch.cat([x for x,y in minibatches_device]).to(device)
#         all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
#         # Get logit and make prediction on PRED_TIME
#         ts = torch.tensor(dataset.PRED_TIME).to(device)


#         # There is an unimplemented feature of cudnn that makes it impossible to perform double backwards pass on the network
#         # This is a workaround to make it work proposed by pytorch, but I'm not sure if it's the right way to do it
#         with torch.backends.cudnn.flags(enabled=False):
#             out, _ = self.predict(all_x, ts, device)

#         # Split data in shape (n_train_envs, batch_size, len(PRED_TIME), num_classes)
#         out_split = dataset.split_output(out)
#         labels_split = dataset.split_labels(all_y)

#         # Compute loss for each environment 
#         env_losses = torch.zeros(out_split.shape[0]).to(device)
#         for i in range(out_split.shape[0]):
#             for t_idx in range(out_split.shape[2]):     # Number of time steps
#                 env_losses[i] += self.loss_fn_inner(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 

#         # Get the gradients
#         param_gradients = [[] for _ in self.model_inner.parameters()]
#         for env_loss in env_losses:
#             env_grads = autograd.grad(env_loss, self.model_inner.parameters(), retain_graph=True)
#             for grads, env_grad in zip(param_gradients, env_grads):
#                 grads.append(env_grad)
                
#       # Compute the meta penalty and update objective 
#         mean_loss = env_losses.mean()
#         meta_grad = autograd.grad(mean_loss, self.model.parameters(), 
#                                         create_graph=True)

#         # compute trace penalty
#         meta_penalty = 0
#         print(param_gradients[0])
#         for grad in param_gradients:
#             for g, meta_grad in zip(grad, meta_grad):
#                 meta_penalty += self.meta_lr * (g-meta_grad).sum() 

#         objective = mean_loss + meta_penalty

#         # Back propagate
#         self.optimizer.zero_grad()
#         objective.backward()
#         self.optimizer.step()
        
# class SANDMask(ERM):
#     """
#     Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
#     AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
#     """

#     def __init__(self, model, dataset, loss_fn, optimizer, hparams):
#         super(SANDMask, self).__init__(model, dataset, loss_fn, optimizer, hparams)

#         # Hyper parameters
#         self.tau = self.hparams['tau']
#         self.k = self.hparams['k']
#         self.betas = self.hparams['betas']

#         # Memory
#         self.register_buffer('update_count', torch.tensor([0]))

#     def mask_grads(self, tau, k, gradients, params, device):
#         '''
#         Mask are ranged in [0,1] to form a set of updates for each parameter based on the agreement 
#         of gradients coming from different environments.
#         '''
        
#         for param, grads in zip(params, gradients):
#             grads = torch.stack(grads, dim=0)
#             avg_grad = torch.mean(grads, dim=0)
#             grad_signs = torch.sign(grads)
#             gamma = torch.tensor(1.0).to(device)
#             grads_var = grads.var(dim=0)
#             grads_var[torch.isnan(grads_var)] = 1e-17
#             lam = (gamma * grads_var).pow(-1)
#             mask = torch.tanh(self.k * lam * (torch.abs(grad_signs.mean(dim=0)) - self.tau))
#             mask = torch.max(mask, torch.zeros_like(mask))
#             mask[torch.isnan(mask)] = 1e-17
#             mask_t = (mask.sum() / mask.numel())
#             param.grad = mask * avg_grad
#             param.grad *= (1. / (1e-10 + mask_t))    

#     def update(self, minibatches_device, dataset, device):

#         ## Group all inputs and send to device
#         all_x = torch.cat([x for x,y in minibatches_device]).to(device)
#         all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
#         # Get logit and make prediction on PRED_TIME
#         ts = torch.tensor(dataset.PRED_TIME).to(device)
#         out, _ = self.predict(all_x, ts, device)

#         # Split data in shape (n_train_envs, batch_size, len(PRED_TIME), num_classes)
#         out_split = dataset.split_output(out)
#         labels_split = dataset.split_labels(all_y)

#         # Compute loss for each environment 
#         env_losses = torch.zeros(out_split.shape[0]).to(device)
#         for i in range(out_split.shape[0]):
#             for t_idx in range(out_split.shape[2]):     # Number of time steps
#                 env_losses[i] += self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 

#         # Compute the grads for each environment
#         param_gradients = [[] for _ in self.model.parameters()]
#         for env_loss in env_losses:

#             env_grads = autograd.grad(env_loss, self.model.parameters(), retain_graph=True)
#             for grads, env_grad in zip(param_gradients, env_grads):
#                 grads.append(env_grad)
            
#         # Back propagate with the masked gradients
#         self.optimizer.zero_grad()
#         self.mask_grads(self.tau, self.k, param_gradients, self.model.parameters(), device)
#         self.optimizer.step()

#         # Update memory
#         self.update_count += 1
                
class IB_ERM(ERM):
    """Information Bottleneck based ERM on feature with conditionning"""

    def __init__(self, model, dataset, optimizer, hparams):
        super(IB_ERM, self).__init__(model, dataset, optimizer, hparams)

        # Hyper parameters
        self.ib_weight = self.hparams['ib_weight']

        # Memory
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self):

        # Put model into training mode
        self.model.train()

        # Get next batch
        X, Y = self.dataset.get_next_batch()

        # Split input / target
        # X, Y = self.dataset.split_input(env_batches)

        # Get predict and get (logit, features)
        out, out_features = self.predict(X)

        # Compute losses
        n_domains = self.dataset.get_nb_training_domains()
        domain_losses = self.dataset.loss_by_domain(out, Y, n_domains)

        # Create domain dimension in tensors. 
        #   e.g. for source domains: (ENVS * batch_size, ...) -> (ENVS, batch_size, ...)
        domain_features, _ = self.dataset.split_tensor_by_domains(out_features, Y, n_domains)

        # For each environment, compute penalty
        ib_penalty = torch.zeros(n_domains).to(domain_losses.device)
        for i, d_feat in enumerate(domain_features):
            ib_penalty[i] = d_feat.var(dim=0).mean()
        
        objective = domain_losses.mean() + (self.ib_weight * ib_penalty.mean())

        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        # Update memory
        self.update_count += 1
         
# class IB_IRM(ERM):
#     """Information Bottleneck based IRM on feature with conditionning"""

#     def __init__(self, model, dataset, loss_fn, optimizer, hparams):
#         super(IB_IRM, self).__init__(model, dataset, loss_fn, optimizer, hparams)

#         # Hyper parameters
#         self.ib_weight = self.hparams['ib_weight']
#         self.ib_anneal = self.hparams['ib_anneal']
#         self.irm_weight = self.hparams['irm_weight']
#         self.irm_anneal = self.hparams['irm_anneal']

#         # Memory
#         self.register_buffer('update_count', torch.tensor([0]))

#     @staticmethod
#     def _irm_penalty(logits, y):
#         device = "cuda" if logits[0][0].is_cuda else "cpu"
#         scale = torch.tensor(1.).to(device).requires_grad_()
#         loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
#         loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
#         grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
#         grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
#         result = torch.sum(grad_1 * grad_2)
#         return result

#     def update(self, minibatches_device, dataset, device):

#         # Get penalty weight
#         ib_penalty_weight = (self.ib_weight if self.update_count
#                           >= self.ib_anneal else
#                           0.0)
#         irm_penalty_weight = (self.irm_weight if self.update_count
#                           >= self.irm_anneal else
#                           1.0)

#         ## Group all inputs and send to device
#         all_x = torch.cat([x for x,y in minibatches_device]).to(device)
#         all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
#         # Get time predictions and get logits
#         ts = torch.tensor(dataset.PRED_TIME).to(device)
#         out, features = self.predict(all_x, ts, device)

#         # Split data in shape (n_train_envs, batch_size, len(PRED_TIME), num_classes)
#         out_split = dataset.split_output(out)
#         features_split = dataset.split_output(features)
#         labels_split = dataset.split_labels(all_y)

#         # For each environment, accumulate loss for all time steps
#         ib_penalty = torch.zeros(out_split.shape[0]).to(device)
#         irm_penalty = torch.zeros(out_split.shape[0]).to(device)
#         env_losses = torch.zeros(out_split.shape[0]).to(device)
#         for i in range(out_split.shape[0]):
#             for t_idx in range(out_split.shape[2]):     # Number of time steps
#                 # Compute the penalty
#                 env_losses[i] += self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 
#                 # Compute the information bottleneck penalty
#                 ib_penalty[i] += features_split[i, :, t_idx, :].var(dim=0).mean()
#                 # Compute the invariant risk minimization penalty
#                 irm_penalty[i] += self._irm_penalty(out_split[i,:,t_idx,:], labels_split[i,:,t_idx])

#         objective = env_losses.mean() + ib_penalty_weight * ib_penalty.mean() + irm_penalty_weight * irm_penalty.mean()

#         if self.update_count == self.ib_anneal or self.update_count == self.irm_anneal:
#             # Reset Adam, because it doesn't like the sharp jump in gradient
#             # magnitudes that happens at this step.
#             self.optimizer = torch.optim.Adam(
#                 self.model.parameters(),
#                 lr=self.optimizer.param_groups[0]['lr'],
#                 weight_decay=self.optimizer.param_groups[0]['weight_decay'])

#         # Back propagate
#         self.optimizer.zero_grad()
#         objective.backward()
#         self.optimizer.step()

#         # Update memory
#         self.update_count += 1



class AbstractCAD(ERM):
    """Contrastive adversarial domain bottleneck (abstract class)
    from Optimal Representations for Covariate Shift <https://arxiv.org/abs/2201.00057>
    """

    def __init__(self, model, dataset, optimizer, hparams, is_conditional):
        super(AbstractCAD, self).__init__(model, dataset, optimizer, hparams)
    # def __init__(self, input_shape, num_classes, num_domains,
    #              hparams, is_conditional):
    #     super(AbstractCAD, self).__init__(input_shape, num_classes, num_domains, hparams)

        # input_shape = dataset.INPUT_SHAPE
        # num_classes = dataset.OUTPUT_SIZE
        self.num_domains = len(dataset.ENVS)
        self.num_train_domains = self.num_domains - 1 if dataset.test_env is not None else self.num_domains

        # self.featurizer = networks.Featurizer(input_shape, self.hparams)
        # self.classifier = networks.Classifier(
        #     self.featurizer.n_outputs,
        #     num_classes,
        #     self.hparams['nonlinear_classifier'])
        # params = list(self.featurizer.parameters()) + list(self.classifier.parameters())
        self.model = model
        self.optimizer = optimizer

        # parameters for domain bottleneck loss
        self.is_conditional = is_conditional  # whether to use bottleneck conditioned on the label
        self.base_temperature = 0.07
        self.temperature = hparams['temperature']
        self.is_project = hparams['is_project']  # whether apply projection head
        self.is_normalized = hparams['is_normalized'] # whether apply normalization to representation when computing loss

        # whether flip maximize log(p) (False) to minimize -log(1-p) (True) for the bottleneck loss
        # the two versions have the same optima, but we find the latter is more stable
        self.is_flipped = hparams["is_flipped"]

        # if self.is_project:
        #     self.project = nn.Sequential(
        #         nn.Linear(feature_dim, feature_dim),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(feature_dim, 128),
        #     )
        #     params += list(self.project.parameters())

        # # Optimizers
        # self.optimizer = torch.optim.Adam(
        #     params,
        #     lr=self.hparams["lr"],
        #     weight_decay=self.hparams['weight_decay']
        # )

    def bn_loss(self, z, y, dom_labels):
        """Contrastive based domain bottleneck loss
         The implementation is based on the supervised contrastive loss (SupCon) introduced by
         P. Khosla, et al., in “Supervised Contrastive Learning“.
        Modified from  https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/losses.py#L11
        """
        device = z.device

        # Flatten tensor (batch, time, ...) -> (batch*time, ...)
        z, y, dom_labels = z.view(-1, *z.shape[2:]), y.view(-1), dom_labels.view(-1)
        batch_size = z.shape[0]

        y = y.contiguous().view(-1, 1)
        dom_labels = dom_labels.contiguous().view(-1, 1)
        mask_y = torch.eq(y, y.T).to(device)
        mask_d = (torch.eq(dom_labels, dom_labels.T)).to(device)
        mask_drop = ~torch.eye(batch_size).bool().to(device)  # drop the "current"/"self" example
        mask_y &= mask_drop
        mask_y_n_d = mask_y & (~mask_d)  # contain the same label but from different domains
        mask_y_d = mask_y & mask_d  # contain the same label and the same domain
        mask_y, mask_drop, mask_y_n_d, mask_y_d = mask_y.float(), mask_drop.float(), mask_y_n_d.float(), mask_y_d.float()

        # compute logits
        if self.is_project:
            z = self.project(z)
        if self.is_normalized:
            z = F.normalize(z, dim=1)

        # For all prediction in the time series compute the CAD objective
        outer = z @ z.T
        logits = outer / self.temperature
        logits = logits * mask_drop
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        if not self.is_conditional:
            # unconditional CAD loss
            denominator = torch.logsumexp(logits + mask_drop.log(), dim=1, keepdim=True)
            log_prob = logits - denominator

            mask_valid = (mask_y.sum(1) > 0)
            log_prob = log_prob[mask_valid]
            mask_d = mask_d[mask_valid]

            if self.is_flipped:  # maximize log prob of samples from different domains
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (~mask_d).float().log(), dim=1)
            else:  # minimize log prob of samples from same domain
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (mask_d).float().log(), dim=1)
        else:
            # conditional CAD loss
            if self.is_flipped:
                mask_valid = (mask_y_n_d.sum(1) > 0)
            else:
                mask_valid = (mask_y_d.sum(1) > 0)

            mask_y = mask_y[mask_valid]
            mask_y_d = mask_y_d[mask_valid]
            mask_y_n_d = mask_y_n_d[mask_valid]
            logits = logits[mask_valid]

            # compute log_prob_y with the same label
            denominator = torch.logsumexp(logits + mask_y.log(), dim=1, keepdim=True)
            log_prob_y = logits - denominator

            if self.is_flipped:  # maximize log prob of samples from different domains and with same label
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_n_d.log(), dim=1)
            else:  # minimize log prob of samples from same domains and with same label
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_d.log(), dim=1)

        def finite_mean(x):
            # only 1D for now
            num_finite = (torch.isfinite(x).float()).sum()
            mean = torch.where(torch.isfinite(x), x, torch.tensor(0.0).to(x)).sum()
            if num_finite != 0:
                mean = mean / num_finite
            else:
                return torch.tensor(0.0).to(x)
            return mean

        return finite_mean(bn_loss)

    def update(self):

        # Put model into training mode
        self.model.train()

        # Get next batch
        X, Y = self.dataset.get_next_batch()
        device = X.device
        X_split, Y_split = self.dataset.split_tensor_by_domains(X, Y, self.num_train_domains)

        all_pred, all_z = self.model(X)

        all_d = torch.cat([
            torch.full(y.shape, i, dtype=torch.int64, device=device)
            for i, y in enumerate(Y_split)
        ])

        bn_loss = self.bn_loss(all_z, Y, all_d)
        clf_loss = self.dataset.loss(all_pred, Y)
        total_loss = clf_loss + self.hparams['lmbda'] * bn_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

class CAD(AbstractCAD):
    """Contrastive Adversarial Domain (CAD) bottleneck
       Properties:
       - Minimize I(D;Z)
       - Require access to domain labels but not task labels
       """

    def __init__(self, model, dataset, optimizer, hparams):
        super(CAD, self).__init__(model, dataset, optimizer, hparams, is_conditional=False)
    # def __init__(self, input_shape, num_classes, num_domains, hparams):
    #     super(CAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=False)


class CondCAD(AbstractCAD):
    """Conditional Contrastive Adversarial Domain (CAD) bottleneck
    Properties:
    - Minimize I(D;Z|Y)
    - Require access to both domain labels and task labels
    """
    def __init__(self, model, dataset, optimizer, hparams):
        super(CondCAD, self).__init__(model, dataset, optimizer, hparams, is_conditional=True)
    # def __init__(self, input_shape, num_classes, num_domains, hparams):
    #     super(CondCAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=True)


class Transfer(ERM):
    '''Algorithm 1 in Quantifying and Improving Transferability in Domain Generalization (https://arxiv.org/abs/2106.03632)'''
    ''' tries to ensure transferability among source domains, and thus transferabiilty between source and target'''
    def __init__(self, model, dataset, optimizer, hparams):
        super(Transfer, self).__init__(model, dataset, optimizer, hparams)
    # def __init__(self, input_shape, num_classes, num_domains, hparams):
    #     super(Transfer, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.d_steps_per_g = hparams['d_steps_per_g']

        # Number of domain definition
        self.num_domains = len(dataset.ENVS)
        self.num_train_domains = self.num_domains - 1 if dataset.test_env is not None else self.num_domains

        # Architecture 
        self.model = model

        # Quick Fix of the deep copy problem for stored datasets
        if isinstance(model, LSTM):
            self.model.dataset = None
            self.adv_classifier = copy.deepcopy(model).to(self.model.device)
            self.model.dataset = dataset
        elif isinstance(model, MNIST_LSTM):
            self.model.home_lstm.dataset = None
            self.adv_classifier = copy.deepcopy(model).to(self.model.device)
            self.model.home_lstm.dataset = dataset
        else:
            self.adv_classifier = copy.deepcopy(model).to(self.model.device)
        self.model.dataset = dataset
        self.adv_classifier.dataset = dataset
        # No need to load state dict because it is deepcopied
        self.adv_classifier.load_state_dict(self.model.state_dict())

        # Optimizers
        def get_optimizer_params(optimizer):
            for p_grp in optimizer.param_groups:
                return p_grp
        opt_params = get_optimizer_params(optimizer)
        if self.hparams['gda']:
            self.optimizer = torch.optim.SGD(self.adv_classifier.parameters(), lr=opt_params['lr']) 
        # else:
        #     self.optimizer = torch.optim.Adam(
        #     (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
        #         lr=opt_params['lr'],
        #         weight_decay=opt_params['weight_decay'])

        self.adv_opt = torch.optim.SGD(self.adv_classifier.parameters(), lr=self.hparams['lr_d']) 

    def update(self):

        # Put model into training mode
        self.model.train()

        # Get next batch
        X, Y = self.dataset.get_next_batch()

        preds, _ = self.model(X)
        loss = self.dataset.loss(preds, Y)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        gap = self.hparams['t_lambda'] * self.loss_gap(X,Y)

        objective = loss + gap
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        self.adv_classifier.load_state_dict(self.model.state_dict())
        for _ in range(self.d_steps_per_g):
            self.adv_opt.zero_grad()
            gap = -self.hparams['t_lambda'] * self.loss_gap(X,Y)
            gap.backward()
            self.adv_opt.step()
            updated_adv_classifier = self.proj(self.hparams['delta'], self.adv_classifier.get_classifier_network(), self.model.get_classifier_network())
            self.adv_classifier.get_classifier_network().load_state_dict(updated_adv_classifier.state_dict())

    def loss_gap(self, X, Y):
        ''' compute gap = max_i loss_i(h) - min_j loss_j(h), return i, j, and the gap for a single batch'''
        device = X.device
        max_env_loss, min_env_loss =  torch.tensor([-float('inf')], device=device), torch.tensor([float('inf')], device=device)

        # Get adv prediction
        _, feats = self.model(X)
        pred = self.adv_classifier.classify(feats)
        losses = self.dataset.loss_by_domain(pred, Y, self.num_train_domains)

        min_env_loss = min(losses)
        max_env_loss = max(losses)

        return max_env_loss - min_env_loss
        
    def distance(self, h1, h2):
        ''' distance of two networks (h1, h2 are classifiers)'''
        dist = 0.
        for param in h1.state_dict():
            h1_param, h2_param = h1.state_dict()[param], h2.state_dict()[param]
            dist += torch.norm(h1_param - h2_param) ** 2  # use Frobenius norms for matrices
        return torch.sqrt(dist)


    def proj(self, delta, adv_h, h):
        ''' return proj_{B(h, \delta)}(adv_h), Euclidean projection to Euclidean ball'''
        ''' adv_h and h are two classifiers'''
        dist = self.distance(adv_h, h)
        if dist <= delta:
            return adv_h
        else:
            ratio = delta / dist
            for param_h, param_adv_h in zip(h.parameters(), adv_h.parameters()):
                param_adv_h.data = param_h + ratio * (param_adv_h - param_h)
            # print("distance: ", distance(adv_h, h))
            return adv_h

    def update_second(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count = (self.update_count + 1) % (1 + self.d_steps_per_g)
        if self.update_count.item() == 1:
            all_x = torch.cat([x for x,y in minibatches])
            all_y = torch.cat([y for x,y in minibatches])
            loss = F.cross_entropy(self.predict(all_x), all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            del all_x, all_y
            gap = self.hparams['t_lambda'] * loss_gap(minibatches, self, device)
            self.optimizer.zero_grad()
            gap.backward()
            self.optimizer.step()
            self.adv_classifier.load_state_dict(self.classifier.state_dict())
            return {'loss': loss.item(), 'gap': gap.item()}
        else:
            self.adv_opt.zero_grad()
            gap = -self.hparams['t_lambda'] * loss_gap(minibatches, self, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(self.hparams['delta'], self.adv_classifier, self.classifier)
            return {'gap': -gap.item()}
