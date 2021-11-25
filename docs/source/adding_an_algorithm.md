
# Adding an Algorithm

In this section, we will walk through the process of adding an algorithm to the framework.

## Defining the Algorithm

We first define the algorithm by creating a new class in the objectives module. In this example we will add scaled_ERM which is simply ERM with a random scale factor between 0 and max_scale for each environment in a dataset, where max_scale is an hyperparameter of the objective.

Let's first define the class and its __int__ method to initialize the algorithm.
```python
class scaled_ERM(ERM):
    """
    Scaled Empirical Risk Minimization (scaled ERM)
    """

    def __init__(self, model, dataset, loss_fn, optimizer, hparams):
        super(scaled_ERM, self).__init__(model, dataset, loss_fn, optimizer, hparams)

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.max_scale = hparams['max_scale']
        self.scaling_factor = self.max_scale * torch.rand(len(dataset.train_names)) 
```
We then need to define the update function, which take a minibatch of data and compute the loss and update the model according to the algorithm definition. Note here that we do not need to define the predict function, as it is already defined in the base class.
```python
    def update(self, minibatches_device, dataset, device):

        ## Group all inputs and send to device
        all_x = torch.cat([x for x,y in minibatches_device]).to(device)
        all_y = torch.cat([y for x,y in minibatches_device]).to(device)
        
        ts = torch.tensor(dataset.PRED_TIME).to(device)
        out = self.predict(all_x, ts, device)

        ## Reshape the data so the first dimension are environments)
        out_split, labels_split = dataset.split_data(out, all_y)

        env_losses = torch.zeros(out_split.shape[0]).to(device)
        for i in range(out_split.shape[0]):
            for t_idx in range(out_split.shape[2]):     # Number of time steps
                env_losses[i] += self.scaling_factor[i] * self.loss_fn(out_split[i, :, t_idx, :], labels_split[i,:,t_idx])

        objective = env_losses.mean()

        # Back propagate
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
```

## Adding necessary pieces
Now that our algorithm is defined, we can add it to the list of algorithms at the top of the objectives module.
```python
OBJECTIVES = [
    'ERM',
    'IRM',
    'VREx',
    'SD',
    'ANDMask',
    'IGA',
    'scaled_ERM',
]
```
Before being able to use the algorithm, we need to add the hyper parameters related to this algorithm in the hyperparams module. Note: the name of the funtion needs to be the same as the name of the algorithm followed by _hyper.
```python
def scaled_ERM_hyper(sample):
    """ scaled ERM objective hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
    if sample:
        return {
            'max_scale': lambda r: r.uniform(1.,10.)
        }
    else:
        return {
            'max_scale': lambda r: 2.
        }
```
## Run some tests
We can now run a simple test to check that everything is working as expected
```sh
pytest
```
## Try the algorithm
Then we can run a training run to see how the algorithm performs on any dataset
```sh
python3 -m woods.scripts.main train \
        --dataset Spurious_Fourier \
        --objective scaled_ERM \
        --test_env 0 \
        --data_path ./data
```
## Run a sweep
Finally, we can run a sweep to see how the algorithm performs on all the datasets
```sh
python3 -m woods.scripts.hparams_sweep \
        --objective scaled_ERM \
        --dataset Spurious_Fourier \
        --data_path ./data \
        --launcher dummy
```