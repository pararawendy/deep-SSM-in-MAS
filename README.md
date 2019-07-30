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





