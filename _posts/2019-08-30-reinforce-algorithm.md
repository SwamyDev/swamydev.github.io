---
title: "Modelling AI Agents with Bayesian Inference"
date: 2019-08-10T06:34:30+02:00
comments: true
categories:
  - blog
tags:
  - AI
  - Math
  - Reinforcement Learning
mathjax: true
---
Today I'd like to explain the principles of the REINFORCE algorithm. This algorithm forms the basis of many modern AI algorithms, which have been seen beating humans at Go, StarCraft or DotA. I think understanding it, is a corner stone in understanding the field of of reinforcement learning. But first we need to know some basics, like how we model the world and the decisions process of our AI agent.

## The Environment

### Markov Decision Process
In the field of AI it is common to model the decision process of an agent within an environment as so called 'Markov Decision Process' or MDP. Basically it is a simple graph that maps wwich actions lead to which states. For instance you could model using an elevator as MDP. Here selecting the floor level, the action, leads to a new state, the elevator moving to that level. This is a fairly simple MDP and unfortunately most environments are much more complex than this model captures. In most cases actions do not necessarly lead always to the same outcomes. Even in our simple elevator example, this is not guaranteed. For example, the elevator could malfunction and get stuck at a different level than the one we requested, through ouractions. To model these situations we usually use 'Stochastic Markov Decision Processes'. In this model an action only has a certain chance to lead to a certain state. It might also lead to a different one, or just return to the same state. 

[ADD SIMPLE MDP GRAPHIC HERE]

### Reward Signals
Another important feature we need to be able to work with MDPs in Reinforcement Learning are rewards. Somewhere in this decision process we need some form of reward signal. It could be that each action we take or state transition we do, is accompanied by a certain reward. But it is also possible that rewards are only received once we reach a special terminal state of the MDP (In this case you could think of all other state transition rewards would be 0). Looking back at our elevator example again, we could get a positive reward when we reach our destination in time, and a negative reward when we don't because we got stuck in the elevator. 

These are the main ingredietns necessary to make reinforcement learning work. In the next section I'll exaplain how we can actually model the actual action selection of an agent within such an environment. 

### Modelling Decitions
Now that we know how a MDP is set up, how do we actually model an agent selecting actions within such an environment. Specifically we wan't to know the chance that an agent selects a certain action when it is in a certain state. Or more formally we want to know the action selection probability density *given* a certain state. If this sounds like Bayesian statistics all over again to you then you are absolutely right (if now I recommend reading my previous posts[LINK HERE]). In the reinforcement learning litrature the policy is usually expressed as the function $$pi$$:

$$FORMULA HERE Policy follows distributuon pi(a|s)$$

But often we do not or can not model this distribution precicely, so we use a *parameterized approximation*. What this means is that the action selection probability produced by our policy function not only depends on the state, but also some approximation parameters. These are usually represented with $$theta$$. This $$theta$$ can stand in for anything, from the parameters of a linear equation to the weights of an Artifical Neural Network. The point is those paraters are "somehow" involved in producing the action selection probability density distribution. What we wan't in the end with all these algorithms is to find good values for these paramters so our policy functions produces good probabilites.

## Selecting better actions - Policy Gradient
Sow how do we get better estimates for these probabilites. Recall from Bayesian statistics, that we can improve our posterior by collecting more data. In our case here the posterior is the probability density distribution of our policy function. The data we can collect in our environment is the reward signal, which tells as how "well" our agent has faired within the environment. The agent can traverse the environment and use the reward signal to inform itself about the value of certain states or actions (there is an important destinction between the two but we don't worry about that in this post). In the next section I'll explain how we can integrate this reward signal into our current model of the agent. 

### Integrating the reward signal
What we want in the end is that our action selects actions in a way so it gathers the highest possible reward within the constraints of the environment. Because we are usually dealing with a stochastic decision process (the policy) and a stochastic MDP, we acutally want the highest possible *expected* reward. And we are back to statistics again. Recall that for the descrete case, the expected value is the sum of all values weighted by their probability occuring. In our case the probability of a certain action given a certain state is defined by our policy function, and the value is gathared as a reward signal from our environment. This means we can write our expected rewards in terms of the policy of the agent:

$$Expected reward formula here$$

Unfortunately we usually do not know, or the environment doesn't give us the specific reward signal for each action in each state, so we usually have to work with estimates. There are many different strategies of selecting a good approximation for the reward signal, like for instance an Advantage function, Q-Values, Average Returns [LINKS TO SOURCES] and many more. However, I save explaining these for a different post (or just follow the links). For now we will just say that the action selection probabilites are weighted by some return value which is high if the choice was good and low if it was bad. 

### Updating our Belief with Reward Signal
How do we update so we select better actions. Explain expected return and gradient. Use as loss function like in other ML. Explain why negative

### Stuck at Local Minima
Explain why we devide by probability density functions -> To award "curiosity" and weight unlikely actions more.

## Conclusion
Conclusion, and tease implementation for next post.
