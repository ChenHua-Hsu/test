<!--### Time derivative score-based model notes-->
<!-- ![img](Banner_grey.jpg) -->
# Setup

## lxplus setup
```
source /cvmfs/sft.cern.ch/lcg/views/LCG_102b_cuda/x86_64-centos7-gcc8-opt/setup.sh
```
To check which versions are intalled with you python installation start a python session and run:
```
print(torch_geometric.__version__)
```
## Conda
Used conda pacakage installation and management tool to create new environment 'tdsbmodel' holding all required packages
conda activate tdsbmodel

## Python Virtual Environment
You can set up a virtualenv. Using the following commands I updated to python 3.8 
on lxplus:
```
scl enable rh-python38 bash
python -V
```
which enabled me to setup a venv for python 3.8:
```
python -m venv virtualenvpy3p8
```
update pip
```
pip install --upgrade pip
```
and install the latest stable pytorch version:
```
pip install torch
```
Now the command 
```
torch.cuda.get_arch_list()
```
when run on a machine with GPU capability, should return a list that includes sm_80.

Living list of required libraries:
```
torch
functools
numpy
matplotlib
```

# Condor
## Running Jobs on Condor
Note the condor scripts available in the condor_scripts directory allow one to submit jobs to run on CERNs lxplus farm remotely. You can ask for more CPUs and/or memory but the system will scale the number of CPUs you receive to respect the 2GB per core limit.

To ask for more CPUs, alter N in the following command in the submission script:
```
request_CPUs = N
```
By default, the job will get one slot on a single CPU core with 2GB of (RAM) memory and 20GB of disk space. You can see the portion of memory that is occupied by a process that is held in the main RAM memory by checking the resident set size (RSS). Currently, jobs seem to be close to the limit (~2GB) so sometimes we get memory warning/failures. (numbers reutned by memory_info are expressed in bytes. We divide them to get number in GB)

```
condor_q -global -all -const '(requestGPUs > 0)'
```
Number of GPU with 4 or more slots:
```
condor_status -const 'TotalSlotGpus >= 4' -compact
```
## Priority
Check real priority with
```
condor_userprio -all
```
0.5 is best possible value. The higher, the worse. User with priority 10 will get twice as many machines as a user with priority 20.

## Monitoring jobs
A very useful command to monitor jobs that perhaps don't run, start to run and stop, or get held etc. (i.e. don't fail but don't succeed) is:
```
condor_q -better -analyze <jobid>
```
If the problem was that the environment/machine specs you've asked for just don't exist, this is where you'll find the proof.

## Running on GPUs
GPU resources are available through HTCondor. One can simply add the folowing line to the submission script:
```
request_GPUs = N
```
You can request < 5 slots on a GPU core (not sure it's possible to request multiple different machines?). Pytorch has built in methods for parrellising processes. DataParrallel is a single process multi-thread parrallelism that requires minimal extra code. DistributedDataParrallel is a multi-process parrallelism that can work on different machines by grouping DDP instances together under a group ID. It is faster and scalable and incredibly useful if you have several GPU machines

Careful when requesting several GPU. After considering machines only with more than one slot, that are accessible in the longer queues etc you may not have many options and might be waiting a while to get on a machine that can run your job.


# Datasets
Taking datasets from calochallenge atm. See calochallenge page for where to download datasets from.

# Useful/interesting Pytorch Info
## Modules
Pytorch uses modules to represent neural networks. Pytorch provides a robust library of modules and makes it simple to define new ones.
All NN modules should:
- Inherit from the base module class nn.Module.
- Define some 'state used in computation i.e. randomly-initialised weights and bias tensors. These are defined as nn.Parameter i.e. tensors to be considered a learnable module parameter and are registered and tracked.
- Define some forward function e.g. in a 'Linear' module, an affine transformation like w(x)+b is performed. This will matrix-multiply the input with the weights and add the bias tensor to create some output tensor.

# Understanding score-matching
If we take for starters, the simplified time-independent case with no noise perturbation, what we do is take all the input images, pass them through a neural network which gives us an estimate of the scores for each pixel in the image. This will be a tensor of scores for for each pixel for all images (same input and output dimensions). Predicting the scores rather than the the probability density avoids having to calculate the intractable normalisation constant.

In an ideal world we would then use the Fisher divergence between the model and data distributions as an objective function. This tells us how the distribution of scores we predicted compares to the true distribution of scores in our data. In reality we don't have access to the true data distribution, so we use the integration by parts trick which eliminates the dependence on the ground truth. 

We then treat this objective function as we would any loss function. We perform backpropagation to update the parameters of the network in such a way that minimises the objective function.

# Denoising Score matching (DSM)
Denoising score matching allows us to avoid calculating the trace of the Jacobian matrix representing the derivatives of the scores in the objective function, which becomes necessary when using the score matching trick of integration by parts to avoid having to know the gradients of the log of the ground truth data density.

It is inspired by score matching and the denoising autoencoder approach of using pairs of clean and corrupted examples (x_, x). First, we perturb the data a little with a known noise and then estimate the scores on that. The idea is that following the gradients of the log of the density from some corrupted points should lead us towards the clean sample. This means we end up with an objective function that is faster to calculate.

# Stochastic Differential Equations
The idea of perturbing the data in order to improve the objective functions caculability ties in with the desire to improve the accuracy of the score estimation in regions of low data density. By perturbing the data with noise, we can populate the lower density regions, thus making the score estimates in those regions, which are initially not great, much better.

The perturbations are done in a continuous manner using stochastic differential equations that perturb the data into some prior distribution. The objective function of the NN is then parametrised as a function of the time steps in this procedure and can be calculated if we know the perturbation kernel.

# DSM loss
A very useful link to understand what's going on here is here: https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf. The DSM objective tries to minimise the square error between the scores for the pixels in the image wrt the data distribution. It is a form of fishers divergence tries to evaluate how far away the target image’s pixels are from the predicted/generated image’s pixels

Inside the l2-norm brackets i.e. || . . . . || can be explained by doing the math. We can substitute for the second term in the l2-norm i.e. in the derivative of the log of the transition probability. The transition probability, what we perturb our data with, is the additivie gaussian noise i.e. our transition kernel. If we do the derivative wrt x(t) using the chain rule and knowing we perturbed the data (x_ = x + z*std) we are left with z/std. The score was divided by std in the model, we can multiply this by std which gives us
```
score * std + z
```
This is then squared so we now have the squared difference of the scores for each pixel.
This is then summed together for all pixels in the example to give the sum of the squared difference of the scores across the pixels, summarising this into a per image difference.
```
torch.sum( (....)**2, dim=(1,2,3) ) 
```
The average of these values for the examples in the batch is then calculated via the mean. In the code this amounts to the simplified:
```
torch.mean(torch.sum( (score * std[:, None, None, None] + z) **2, dim=(1,2,3)))
```

We know std from our SDE we have setup. Given the SDE, the transition kernel is:
```
N( x(t); x(0), 1/(2log{sigma}) * (sigma^{2t-1}) I )
```
So the std is given by:
```
sqrt{ 1/(2log{sigma}) * (sigma^{2t-1}) }
```

# ScoreNet model
Based on the U-net architecture. In this particulare application, the mmodel takes pixel values for images as input, outputs an estimate for the score for each pixel.

# Sampling using the model
Once we have a trained model, we can use MCMC techniques to generate samples from this distribution. We can basically take some image/example sampled from an arbitrary prior and generate the next image through a chained procedure in which the image is iteratively altered using the reverse process of the noising procedure.

# Solving the reverse SDE
Any SDE of the form:
'''
dx = f(x,t)dt + g(t)dw
'''
has a reverse-time SDE given by:
```
dx = [f(x,t) - g(t)**2 d(logpt(x))dx]dt + g(t)dw
```
We have a model trained to approximate logpt(x). We can draw a sample from our prior distribution, taking some image w randomly assigned pixel values and multiplying this with the std of the transition kernel gaussian:
```
init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:,None,None,None]
```
We can plop this into our model to estimate this part of the reverse-time SDE. We can use certain numerical methods to solve the reverse-time SDE. These provide us with an iterative procedure allowing us to generate samples from earlier in the diffusion process.

# Euler-Maruyama Method
This method discretises the SDE:
```
dt -> delta(t)
dw -> z ~ N(0, g(t)**2 delta(t)I)
```

# Predictor-corrector method
This method adds a corrector method (e.g. MCMC Langevin dynamics) to a predictor method (e.g. Euler-Maruyama method). The corrector method leverages score-based MCMC approaches to correct the solution obtained by the numerical SDE solver. After each predictor step, there is a corrector step that follows an MCMC iteration rule such as:
```
x_i+1 = x_i + e delta_x log p(x_i) + sqrt(2e) z_i
```
Where:
z_i ~ N(0,I)
e > 0

After the predictor, the corrector takes several steps of Langevin MCMC to refine x_t such that it becomes a more accurate sample from p_{t-deltat}(x). This helps refine the error of the numerical SDE solver.

# Ordinary Differential Equation method
For any SDE there exists an ODE. Note: it is very similar to the reverse-time SDE just without the stochastic term g(t)dw. The trajectories of x have the same marginal probability density p_{t}(x). Therefore, solving the reverse-time ODE allows us to sample from the same distribution as the reverse-time SDE allows. We call this ODE the probability-flow ODE.

We can start from a sample in p_T, integrate the ODE in the reverse-time direction, from t = T, to 0 and get sample p_0. There are many black-box ODE solvers that can perform this integration. Here we use scipy.


# Likelihood Computation
A by-product of using the probability flow ODE method is that we can calculate the likelihood. If we have a a differentiable 1-2-1 mapping 'h' that transforms a data sample x ~ p_0 to a prior h(x) ~ p_T, we can comput the likelihood of p_0(x) via the change of variable formula:
```
p_0(x) = p_T(h(x)) | det(J_h(x)) |
```
where J_h(x) represents the Jacobian of the mapping h


# Transformer models
How you order the dimensions of the input is important when passing to the transformer / attention mechanism. In the pytorch attention mechanism, the code is expecting the input dimensionality to match:
```
[# batches, # points, # features]
```


# Data Transformation
Transforming the input data of your machine learning algorithm is often key to its success. Preprocessing data will often be one of the most time-consuming parts of the project. Custom transformations often necessary and finding the best one can prove incredibly helpful to the algorithm (see here for a nice visual explanation https://donaldpinckney.com/books/pytorch/book/ch2-linreg/2018-11-15-feature-scaling.html)

Pytorch provides a suite of transformations ready to go as well as the functionality to compose your own custom transformations. We can write a custom transformation function as a callable class, rather than a simple function, so that the parameters of the transformation do not need to be passed every time it is called.
