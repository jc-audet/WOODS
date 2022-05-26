
"""Defining domain generalization algorithms"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import matplotlib.pyplot as plt

OBJECTIVES = [
    'ERM',
    'GroupDRO',
    'IRM',
    'VREx',
    'SD',
    # 'ANDMask', # Requires update
    'IGA',
    # 'Fish', # Requires update
    'IB_ERM',
    # 'IB_IRM' # Requires update
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
        batch = self.dataset.get_next_batch()

        # Split into input / target
        X, Y = self.dataset.split_input(batch)
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
        batch = self.dataset.get_next_batch()

        if not len(self.q):
            print("hello, creating Q")
            self.q = torch.ones(self.nb_training_domains).to(self.device)

        # Split input / target
        X, Y = self.dataset.split_input(batch)

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
        env_batches = self.dataset.get_next_batch()

        # Split input / target
        X, Y = self.dataset.split_input(env_batches)

        # Get predict and get (logit, features)
        out, _ = self.predict(X)

        # Compute losses
        batch_losses = self.dataset.loss(out, Y)

        # Create domain dimension in tensors. 
        #   e.g. for source domains: (ENVS * batch_size, ...) -> (ENVS, batch_size, ...)
        #        for time domains: (batch_size, ENVS, ...) -> (ENVS, batch_size, ...) 
        env_out = self.dataset.split_tensor_by_domains(len(env_batches), out)
        env_labels = self.dataset.split_tensor_by_domains(len(env_batches), Y)
        env_losses = self.dataset.split_tensor_by_domains(len(env_batches), batch_losses)
        
        # Compute loss and penalty for each domains
        irm_penalty = torch.zeros(env_out.shape[0]).to(self.device)
        for i in range(env_out.shape[0]):
            for t_idx in range(env_out.shape[2]):     # Number of time steps
                irm_penalty[i] += self._irm_penalty(env_out[i,:,t_idx,:], env_labels[i,:,t_idx])

        # Compute objective
        irm_penalty = irm_penalty.mean()
        objective = env_losses.mean() + (penalty_weight * irm_penalty)

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
        env_batches = self.dataset.get_next_batch()

        # Split input / target
        X, Y = self.dataset.split_input(env_batches)

        # Get predict and get (logit, features)
        out, _ = self.predict(X)

        # Compute losses
        batch_losses = self.dataset.loss(out, Y)

        # Create domain dimension in tensors. 
        #   e.g. for source domains: (ENVS * batch_size, ...) -> (ENVS, batch_size, ...)
        #        for time domains: (batch_size, ENVS, ...) -> (ENVS, batch_size, ...)
        env_out = self.dataset.split_tensor_by_domains(len(env_batches), out)
        env_labels = self.dataset.split_tensor_by_domains(len(env_batches), Y)
        env_losses = self.dataset.split_tensor_by_domains(len(env_batches), batch_losses)

        # Compute objective
        mean = env_losses.mean()
        print(mean.shape, env_losses.shape)
        penalty = ((env_losses - mean) ** 2).mean()
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
        env_batches = self.dataset.get_next_batch()

        # Split input / target
        X, Y = self.dataset.split_input(env_batches)

        # Get predict and get (logit, features)
        out, _ = self.predict(X)

        # Compute losses
        batch_losses = self.dataset.loss(out, Y)

        # Create domain dimension in tensors:
        #   e.g. for source domains: (ENVS * batch_size, ...) -> (ENVS, batch_size, ...)
        #        for time domains: (batch_size, ENVS, ...) -> (ENVS, batch_size, ...) 
        env_out = self.dataset.split_tensor_by_domains(len(env_batches), out)
        env_losses = self.dataset.split_tensor_by_domains(len(env_batches), batch_losses)

        # Compute loss for each environment 
        sd_penalty = torch.pow(env_out, 2).sum(dim=-1)

        # sd_penalty = torch.zeros(env_out.shape[0]).to(env_out.device)
        # for i in range(env_out.shape[0]):
        #     for t_idx in range(env_out.shape[2]):     # Number of time steps
        #         sd_penalty[i] += (env_out[i, :, t_idx, :] ** 2).mean()

        sd_penalty = sd_penalty.mean()
        objective = env_losses.mean() + self.penalty_weight * sd_penalty

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

class IGA(ERM):
    """
    Inter-environmental Gradient Alignment
    From https://arxiv.org/abs/2008.01883v2
    """

    def __init__(self, model, dataset, optimizer, hparams):
        super(IGA, self).__init__(model, dataset, optimizer, hparams)

        # Hyper parameters
        self.penalty_weight = self.hparams['penalty_weight']

    def update(self):

        # Put model into training mode
        self.model.train()

        # Get next batch
        env_batches = self.dataset.get_next_batch()

        # Split input / target
        X, Y = self.dataset.split_input(env_batches)

        # There is an unimplemented feature of cudnn that makes it impossible to perform double backwards pass on the network
        # This is a workaround to make it work proposed by pytorch, but I'm not sure if it's the right way to do it
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = self.predict(X)

        # Compute losses
        batch_losses = self.dataset.loss(out, Y)

        # Create domain dimension in tensors. 
        #   e.g. for source domains: (ENVS * batch_size, ...) -> (ENVS, batch_size, ...)
        env_losses = self.dataset.split_tensor_by_domains(len(env_batches), batch_losses)

        # Get the gradients
        grads = []
        for env_loss in env_losses:

            env_grad = autograd.grad(env_loss.mean(), [p for p in self.model.parameters() if p.requires_grad], 
                                        create_graph=True)
            grads.append(env_grad)

        # Compute the mean loss and mean loss gradient
        mean_loss = env_losses.mean()
        mean_grad = autograd.grad(mean_loss, [p for p in self.model.parameters() if p.requires_grad], 
                                        create_graph=True)

        # compute trace penalty
        penalty_value = 0
        for grad in grads:
            for g, mean_g in zip(grad, mean_grad):
                penalty_value += (g - mean_g).pow(2).sum()

        objective = mean_loss + self.penalty_weight * penalty_value

        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        
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
        env_batches = self.dataset.get_next_batch()

        # Split input / target
        X, Y = self.dataset.split_input(env_batches)

        # Get predict and get (logit, features)
        out, out_features = self.predict(X)

        # Compute losses
        batch_losses = self.dataset.loss(out, Y)

        # Create domain dimension in tensors. 
        #   e.g. for source domains: (ENVS * batch_size, ...) -> (ENVS, batch_size, ...)
        env_features = self.dataset.split_tensor_by_domains(len(env_batches), out_features)
        env_losses = self.dataset.split_tensor_by_domains(len(env_batches), batch_losses)

        # For each environment, accumulate loss for all time steps
        ib_penalty = torch.zeros(env_features.shape[0]).to(env_features.device)
        for i in range(env_features.shape[0]):
            for t_idx in range(env_features.shape[2]):     # Number of time steps
                # Compute the information bottleneck
                ib_penalty[i] += env_features[i, :, t_idx, :].var(dim=0).mean()

        objective = env_losses.mean() + (self.ib_weight * ib_penalty.mean())

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