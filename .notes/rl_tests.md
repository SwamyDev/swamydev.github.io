# Conclusions from testing REINFORCE

## Why is it important that returns are stochastic and not fixed?
When an edge case is reached such as all returns are the same or most of them (i.e. when an agent always fails or learns the optimal policy), the mean is nearly equal to most of the returns. Additionally the standard deviation will approach 0. This means we have a ratio of the form, epsilon/epsilon which is highly unstable.

## Why is the initialization of weights important?
Agent's can get stuck at local minima easily, when the initial weights are set unfavourable. One can observe this in the numpy agent, when setting m to a negative value, as the derivative is fixed it will get stuck in a negative loss. However, there might be a problem in implementation of that agent, I'd need to think it through again.

## What is one of the most important hyper parameters?
The learning rate, because it it is too small learning happens to slow. This can lead to the agent getting stuck in a local minima, because the reward signal is to small. However, if it is too large, the agent never settles on a good policy. It seems that in general RL agents need a higher LR than in traditional ML? Also my experiments show that the optimal LR value is linked to the magnitude of the average return signal -> here normalization might help to stabilize it.
