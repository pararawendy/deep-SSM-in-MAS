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




