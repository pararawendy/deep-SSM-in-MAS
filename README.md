# Deep State-Space Models in Multi-Agent Systems
Code implementation of deep state-space models (SSM) in multi-agent-systems (MAS) as proposed by [Indarjo's master thesis](https://www.universiteitleiden.nl/binaries/content/assets/science/mi/scripties/master/2018-2019/master-thesis-fin---pararawendy.pdf) (2019)

Outline:
1. Explain about the method, incl. how to use the method
2. Method with covariance structure
3. method implementation (explain the toy problem), note: modif toy problem in phase 2 (ground truth of 2nd agent)

![equation](https://latex.codecogs.com/gif.latex?)

## Problem assumptions
We consider multi-agent systems (MAS) that consist of a controllable agent and some other agents (can be either collaborators or opponents, or both) which interact with each other. The general assumptions on the systems are as follows:
- Observation in each time step, denoted by ![equation](https://latex.codecogs.com/gif.latex?o_t) is a noisy non-linear mapping from a true environment state ![equation](https://latex.codecogs.com/gif.latex?s_t), which is unobserved (latent).
- The true environment state ![equation](https://latex.codecogs.com/gif.latex?s_t) evolves over time and gets affected also by both controllable agent's action ![equation](https://latex.codecogs.com/gif.latex?u_t), and other agents' actions ![equation](https://latex.codecogs.com/gif.latex?a_t).
- Other agents' actions ![equation](https://latex.codecogs.com/gif.latex?a_t) also come from a noisy non-linear mapping of another true latent state representation ![equation](https://latex.codecogs.com/gif.latex?z_t).
- The true latent state representation ![equation](https://latex.codecogs.com/gif.latex?z_t) also evolves over time and gets affected by the environment state ![equation](https://latex.codecogs.com/gif.latex?s_t), which can be seen as the *summary* of all current observation and previous actions (both controllable agent's and other agents').

We propose a formulation of deep state-space models (DSSMs) over the considered systems. There are two main goals of this formulation. First is to model the environment dynamics. That is, the change on the environment as a result of all actions taken by the involved agents in the systems. Secondly, to predict the other agents' actions to be taken utilizing some results from the first modelling phase.

## The models
The key idea of the models is to perform two phases of DSSMs, each for modelling one of the two goals mentioned above. There are three sequences available (observed) in this problem: the observations ![equation](https://latex.codecogs.com/gif.latex?o_{1:T}=[o_1,...,o_T]), our controllable agent's actions ![equation](https://latex.codecogs.com/gif.latex?u_{1:T}=[u_1,...,u_{T}]) and other agents' actions ![equation](https://latex.codecogs.com/gif.latex?a_{1:T}=[a_1,...,a_{T}]). For convenience, we will denote the concatenation of the two agents' actions as ![equation](https://latex.codecogs.com/gif.latex?c_t=[u_t,a_t]).

Phase 1 of the models is devoted to model the sequence observations, conditioned on all actions in place (our controllable agent's and the other agents'). The main goal is to perform inference on the sequence of environment states ![equation](https://latex.codecogs.com/gif.latex?s_{1:T}=[s_1,...,s_T]) from the observations ![equation](https://latex.codecogs.com/gif.latex?o_{1:T}), which is also affected by both agents' actions ![equation](https://latex.codecogs.com/gif.latex?c_{1:T}). As environment states, we can think of  ![equation](https://latex.codecogs.com/gif.latex?s_{1:T}) as a sequence of summaries that contains all the information of the system in a more compact way.  After we have ![equation](https://latex.codecogs.com/gif.latex?s_{1:T}) at hand, we go to Phase 2 of the models, namely to model the sequence of other agents' actions ![equation](https://latex.codecogs.com/gif.latex?a_{1:T}), conditioned by the system summaries ![equation](https://latex.codecogs.com/gif.latex?s_{1:T}).

### Phase 1
In this phase, the goal is originally to fit a generative model for the sequence of observations conditioned on the sequence of all actions in place, i.e. ![equation](https://latex.codecogs.com/gif.latex?p(o_{1:T}|c_{1:T})). We achieve this goal by incorporating self-introduced latent variables (here they are known as the environment states) ![equation](https://latex.codecogs.com/gif.latex?s_{1:T}). The joint distribution (or equivalently the generative model) in this phase is

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20p_%7B%5Ctheta%2C%5Cgamma%7D%28o_%7B1%3AT%7D%2Cs_%7B1%3AT%7D%7Cc_%7B1%3AT%7D%29%20%26%3D%20p_%5Ctheta%28o_%7B1%3AT%7D%7Cs_%7B1%3AT%7D%29%20p_%5Cgamma%28s_%7B1%3AT%7D%7Cc_%7B1%3AT%7D%29%5Cnonumber%5C%5C%20%26%3D%20%5Cprod_%7Bt%3D1%7D%5ET%20p_%5Ctheta%28o_t%7Cs_t%29%20p_%5Cgamma%28s_t%7Cs_%7Bt-1%7D%2Cc_t%29%20%5Clabel%7Bgenerative_phase1%7D%5Cnonumber%20%5Cend%7Balign%7D)

Along with this, we subsequently introduce a variational distribution ![equation](https://latex.codecogs.com/gif.latex?q_\phi(s_{1:T}|o_{1:T},c_{1:T})) to approximate the true (but intractable) posterior ![equation](https://latex.codecogs.com/gif.latex?p_\gamma(s_{1:T}|o_{1:T},c_{1:T})). We define the factorization of this variational distribution as follows

![equation](https://latex.codecogs.com/gif.latex?q_%5Cphi%28s_%7B1%3AT%7D%7Co_%7B1%3AT%7D%2Cc_%7B1%3AT%7D%29%3D%5Cprod_%7Bt%3D1%7D%5ET%20q_%5Cphi%28s_t%7Cs_%7Bt-1%7D%2Cc_t%2Co_t%29)

Therefore, the objective function we want to optimize (maximize) in Phase 1 is the following ELBO (Evidence Lower BOund) of ![equation](https://latex.codecogs.com/gif.latex?p(o_{1:T}|c_{1:T})), which is the simplification of the following expression

![equation](https://latex.codecogs.com/gif.latex?%5Ctext%7BELBO%7D%20%3D%20%5Cmathbb%7BE%7D_%7Bq_%5Cphi%28s_%7B1%3AT%7D%7Co_%7B1%3AT%7D%2Cc_%7B1%3AT%7D%29%7D%20%5Cleft%5B%20%5Csum_%7Bt%3D1%7D%5ET%20%5Clog%20%5Cfrac%7Bp_%5Ctheta%28o_t%7Cs_t%29%20p_%5Cgamma%28s_t%7Cs_%7Bt-1%7D%2Cc_t%29%7D%7Bq_%5Cphi%28s_t%7Cs_%7Bt-1%7D%2Cc_t%2Co_t%29%7D%5Cright%5D%20%5Cnonumber)

We assume all unobserved latent variables are multivariate Gaussian distributions, while the distribution of the observation depends on the data specification. It can be modelled as Gaussian (real-valued), binary, or categorical distribution. Concluding this construction, an example of graphical form of the generative model in this phase is given in the following.

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20s_0%20%26%5Csim%20p_0%28s_0%29%20%3D%20%5Cmathcal%7BN%7D%28s_0%3B0%2CI%29%5Cnonumber%5C%5C%20s_t%20%26%5Csim%20p_%5Cgamma%28s_t%7Cs_%7Bt-1%7D%2Cc_%7Bt-1%7D%29%20%3D%20%5Cmathcal%7BN%7D%28s_t%3B%20%7B%7D_%7B%5Cgamma%7D%5Cmu_t%2C%5Ctext%7Bdiag%7D%28%7B%7D_%7B%5Cgamma%7D%5Csigma_t%5E2%29%29%5Cnonumber%5C%5C%20o_t%20%26%5Csim%20p_%5Ctheta%28o_t%7Cs_t%29%20%3D%20%5Cmathcal%7BN%7D%28o_t%3B%7B%7D_%7B%5Ctheta%7D%5Cmu_t%2C%5Ctext%7Bdiag%7D%28%7B%7D_%7B%5Ctheta%7D%5Csigma_t%5E2%29%29%5Cnonumber%20%5Cend%7Balign%7D)

We parameterize each distribution ![equation](https://latex.codecogs.com/gif.latex?p_\theta), ![equation](https://latex.codecogs.com/gif.latex?p_\gamma) and ![equation](https://latex.codecogs.com/gif.latex?q_\phi) by neural networks, such that ![equation](https://latex.codecogs.com/gif.latex?\theta), ![equation](https://latex.codecogs.com/gif.latex?\gamma) and ![equation](https://latex.codecogs.com/gif.latex?\phi) are the parameters of neural networks ![equation](https://latex.codecogs.com/gif.latex?p_\theta), ![equation](https://latex.codecogs.com/gif.latex?p_\gamma) and ![equation](https://latex.codecogs.com/gif.latex?q_\phi) respectively. These neural networks are called *emission net, transition net*, and *inference net*, respectively. Note that, with parameterization by neural networks we mean that the neural networks emit the parameters of the distribution, e.g. the mean and variance values (hence the output layer consists of two blocks in this case) for Gaussian distribution.

Since we parameterize all the distributions using neural networks, ELBO maximization is now carried out through jointly tune all the neural networks parameters ![equation](https://latex.codecogs.com/gif.latex?\theta), ![equation](https://latex.codecogs.com/gif.latex?\gamma) and ![equation](https://latex.codecogs.com/gif.latex?\phi). We train all the neural networks using backpropagation algorithm.

The Bayesian net of Phase 1 is given below.
![Screenshot](phase1.png)



