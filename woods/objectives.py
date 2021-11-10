
"""Defining domain generalization algorithms"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


OBJECTIVES = [
    'ERM',
    'IRM',
    'VREx',
    'SD',
    'ANDMask',
    'IGA',
    'Fish',
    'SANDMask'
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

    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(ERM, self).__init__(hparams)

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def predict(self, all_x, ts, device):

        # Get logit and make prediction
        out = self.model(all_x, ts)

        return out

    def update(self, minibatches_device, dataset, device):

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
        ts = torch.tensor(dataset.PRED_TIME).to(device)
        out = self.predict(all_x, ts, device)

        out_split, labels_split = dataset.split_data(out, all_y)

        env_losses = torch.zeros(out_split.shape[0]).to(device)
        for i in range(out_split.shape[0]):
            for t_idx in range(out_split.shape[2]):     # Number of time steps
                env_losses[i] += self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx])

        objective = env_losses.mean()

        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

class IRM(ERM):
    """
    Invariant Risk Minimization (IRM)
    """

    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(IRM, self).__init__(model, dataset, loss_fn, optimizer, hparams)

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

    def predict(self, all_x, ts, device):

        # Get logit and make prediction
        out = self.model(all_x, ts)

        return out

    def update(self, minibatches_device, dataset, device):

        # Define stuff
        penalty_weight = (self.penalty_weight   if self.update_count >= self.anneal_iters 
                                                else 1.0)

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
        ts = torch.tensor(dataset.PRED_TIME).to(device)
        out = self.predict(all_x, ts, device)

        out_split, labels_split = dataset.split_data(out, all_y)

        penalty = 0
        env_losses = torch.zeros(len(minibatches_device)).to(device)
        for i, (x, y) in enumerate(minibatches_device):

            for t_idx in range(y.shape[1]):     # Number of time steps
                env_losses[i] += self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 
                penalty += self._irm_penalty(out_split[i, :, t_idx, :], labels_split[i,:,t_idx])

        penalty = penalty / out_split.shape[0]
        objective = env_losses.mean() + (penalty_weight * penalty)

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
    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(VREx, self).__init__(model, dataset, loss_fn, optimizer, hparams)

        # Hyper parameters
        self.penalty_weight = self.hparams['penalty_weight']
        self.anneal_iters = self.hparams['anneal_iters']

        # Memory
        self.register_buffer('update_count', torch.tensor([0]))

    def predict(self, all_x, ts, device):

        # Get logit and make prediction
        out = self.model(all_x, ts)

        return out

    def update(self, minibatches_device, dataset, device):

        # Define stuff
        penalty_weight = (self.penalty_weight   if self.update_count >= self.anneal_iters 
                                                else 1.0)

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
        ts = torch.tensor(dataset.PRED_TIME).to(device)
        out = self.predict(all_x, ts, device)

        out_split, labels_split = dataset.split_data(out, all_y)

        env_losses = torch.zeros(len(minibatches_device)).to(device)
        for i, (x, y) in enumerate(minibatches_device):

            for t_idx in range(y.shape[1]):     # Number of time steps
                env_losses[i] += self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 

        mean = env_losses.mean()
        penalty = ((env_losses - mean) ** 2).mean()
        objective = mean + penalty_weight * penalty

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
    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(SD, self).__init__(model, dataset, loss_fn, optimizer, hparams)

        # Hyper parameters
        self.penalty_weight = self.hparams['penalty_weight']

    def predict(self, all_x, ts, device):

        # Get logit and make prediction
        out = self.model(all_x, ts)

        return out

    def update(self, minibatches_device, dataset, device):

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
        ts = torch.tensor(dataset.PRED_TIME).to(device)
        out = self.predict(all_x, ts, device)

        out_split, labels_split = dataset.split_data(out, all_y)

        penalty = 0
        env_losses = torch.zeros(len(minibatches_device)).to(device)
        for i, (x, y) in enumerate(minibatches_device):

            for t_idx in range(y.shape[1]):     # Number of time steps
                env_losses[i] += self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 
                penalty += (out_split[i, :, t_idx, :] ** 2).mean()

        penalty = penalty / out_split.shape[0]
        objective = env_losses.mean() + self.penalty_weight * penalty

        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()


class ANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """

    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(ANDMask, self).__init__(model, dataset, loss_fn, optimizer, hparams)

        # Hyper parameters
        self.tau = self.hparams['tau']

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

    def predict(self, all_x, ts, device):

        # Get logit and make prediction
        out = self.model(all_x, ts)

        return out

    def update(self, minibatches_device, dataset, device):

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
        ts = torch.tensor(dataset.PRED_TIME).to(device)
        out = self.predict(all_x, ts, device)

        out_split, labels_split = dataset.split_data(out, all_y)

        env_losses = torch.zeros(len(minibatches_device)).to(device)
        for i, (x, y) in enumerate(minibatches_device):

            for t_idx in range(y.shape[1]):     # Number of time steps
                env_losses[i] += self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 

        param_gradients = [[] for _ in self.model.parameters()]
        for env_loss in env_losses:

            env_grads = autograd.grad(env_loss, self.model.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)
            
        # Back propagate
        self.optimizer.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.model.parameters())
        self.optimizer.step()

class IGA(ERM):
    """
    Inter-environmental Gradient Alignment
    From https://arxiv.org/abs/2008.01883v2
    """

    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(IGA, self).__init__(model, dataset, loss_fn, optimizer, hparams)

        # Hyper parameters
        self.penalty_weight = self.hparams['penalty_weight']

    def predict(self, all_x, ts, device):

        # Get logit and make prediction
        out = self.model(all_x, ts)

        return out

    def update(self, minibatches_device, dataset, device):

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
        ts = torch.tensor(dataset.PRED_TIME).to(device)
        out = self.predict(all_x, ts, device)

        out_split, labels_split = dataset.split_data(out, all_y)

        env_losses = torch.zeros(len(minibatches_device)).to(device)
        for i, (x, y) in enumerate(minibatches_device):

            for t_idx in range(y.shape[1]):     # Number of time steps
                env_losses[i] += self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 

        # Get the gradients
        grads = []
        for env_loss in env_losses:

            env_grad = autograd.grad(env_loss, self.model.parameters(), 
                                        create_graph=True)

            grads.append(env_grad)
            
        # Compute the mean loss and mean loss gradient
        mean_loss = env_losses.mean()
        mean_grad = autograd.grad(mean_loss, self.model.parameters(), 
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
        
        
        
class Fish(Objective):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain 
    Generalization, Shi et al. 2021.
    """

    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(Fish, self).__init__(hparams)
        
         # Hyper parameters
        self.meta_lr = self.hparams['meta_lr']
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def create_copy(self, device):
        self.model_inner = self.model
        self.optimizer_inner = self.optimizer
        self.loss_fn_inner = self.loss_fn 


    def predict(self, all_x, ts, device):

        # Get logit and make prediction
        out = self.model(all_x, ts)

        return out   

    def update(self, minibatches_device, dataset, device):
        
        self.create_copy(minibatches_device[0][0].device)

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
        ts = torch.tensor(dataset.PRED_TIME).to(device)
        out = self.predict(all_x, ts, device)

        out_split, labels_split = dataset.split_data(out, all_y)

        env_losses = torch.zeros(len(minibatches_device)).to(device)
        for i, (x, y) in enumerate(minibatches_device):

            for t_idx in range(y.shape[1]):     # Number of time steps
                env_losses[i] += self.loss_fn_inner(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 

        param_gradients = [[] for _ in self.model_inner.parameters()]
        for env_loss in env_losses:

            env_grads = autograd.grad(env_loss, self.model_inner.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)
      # Compute the meta penalty and update objective 
        mean_loss = env_losses.mean()
        meta_grad = autograd.grad(mean_loss, self.model.parameters(), 
                                        create_graph=True)
        # compute trace penalty
        meta_penalty = 0
        for grad in grads:
            for g, meta_grad in zip(grad, meta_grad):
                meta_penalty += self.meta_lr * (g-meta_grad).sum() 
        objective = mean_loss + meta_penalty

        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        
        

        
class SANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """

    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(SANDMask, self).__init__(model, dataset, loss_fn, optimizer, hparams)

        # Hyper parameters
        self.tau = self.hparams['tau']
        self.k = self.hparams['k']
        self.betas = self.hparams['betas']

        # Memory
        self.register_buffer('update_count', torch.tensor([0]))

    def mask_grads(self, tau, k, gradients, params, device):
        '''
        Mask are ranged in [0,1] to form a set of updates for each parameter based on the agreement 
        of gradients coming from different environments.
        '''
        
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

    def predict(self, all_x, ts, device):

        # Get logit and make prediction
        out = self.model(all_x, ts)

        return out

    def update(self, minibatches_device, dataset, device):

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
        ts = torch.tensor(dataset.PRED_TIME).to(device)
        out = self.predict(all_x, ts, device)

        out_split, labels_split = dataset.split_data(out, all_y)

        env_losses = torch.zeros(len(minibatches_device)).to(device)
        for i, (x, y) in enumerate(minibatches_device):

            for t_idx in range(y.shape[1]):     # Number of time steps
                env_losses[i] += self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx]) 

        param_gradients = [[] for _ in self.model.parameters()]
        for env_loss in env_losses:

            env_grads = autograd.grad(env_loss, self.model.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)
            
        # Back propagate
        self.optimizer.zero_grad()
        self.mask_grads(self.tau, self.k, param_gradients, self.model.parameters(), device)
        self.optimizer.step()
                