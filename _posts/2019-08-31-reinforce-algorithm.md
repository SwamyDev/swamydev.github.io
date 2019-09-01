---
title: "AI Basics - Policy Gradient"
date: 2019-08-31T06:34:30+02:00
comments: true
categories:
  - blog
tags:
  - AI
  - Math
  - Reinforcement Learning
mathjax: true
---
Today I'd like to explain the principles of the REINFORCE algorithm one of the most fundamental policy gradient algorithms. It forms the basis of many modern AI systems, which have been seen beating humans at Go, StarCraft or DotA. I think understanding it, is a cornerstone in understanding the field of reinforcement learning. In the following sections, I'll explain how the REINFORCE algorithm mathematically expresses an agent's decisions. However, first, we need to know some basics, like how we model the environment and the decisions process of our AI agent.

## The Environment

### Markov Decision Process
In the field of AI, it is common to model the decision process of an agent within an environment as so-called _Markov Decision Process_ or MDP. It is a simple graph that maps which actions lead to which states. For instance, you could model the usage of an elevator as MDP. In this example, **selecting** the floor level, the **action**, leads to a **new state**, the elevator **moving to** that **level**. Granted, this is a fairly simple MDP, and unfortunately, most environments are much more complex than this.

![Elevator MDP Graph]({{ site.url }}{{ site.baseurl}}/assets/images/reinforce-mdp.png)

The labels $$a_1$$ and $$a_2$$ represent the actions of pressing the button for floor 1 and 2 respectively. 
{: .notice--info}

In most cases, actions do not necessarily lead always to the same outcomes. Even in our simple elevator example, in reality, this is not guaranteed. For example, the elevator could malfunction and get stuck at a different level than the one we requested. To model these situations, we usually use _Stochastic Markov Decision Processes_. In this model, an action only has a certain chance to lead to a particular state. It might also lead to a different one or return to the same state. 

![Elevator Stochastic MDP Graph]({{ site.url }}{{ site.baseurl}}/assets/images/reinforce-stochastic-mdp.png)

The numbers next to the actions indicate the probability that this particular action results in the represented state transition.
{: .notice--info}

### Reward Signals
Another important feature we need to be able to work with MDPs in reinforcement learning are the rewards. Somewhere in this decision process, we need some form of reward-signal. It could be that entering particular states is accompanied by certain rewards. However, it is also possible that rewards are only received once we enter a specific terminal state of the MDP.[^terminal] Looking back at our elevator example, we could, for instance, get a positive reward when we reach our destination in time. However, when we are late, because we got stuck in the elevator, we receive a negative reward. 

[^terminal]: In this case, you could think of all other state rewards to be 0.

These are the main ingredients necessary to make reinforcement learning work. In the next section, I'll explain how we can model the actual action selection of an agent within such an environment. 

### Decision making as a mathematical formula
Now that we know how an MDP is set up, how do we model an agent selecting actions within such an environment? Specifically, we want to know the chance that an agent selects a certain action when it is in a certain state. Speaking more mathematical, we want to know the action selection **probability density given a** certain **state**. If this sounds like Bayesian statistics[^bayes] all over again to you, then you are right. In the reinforcement learning literature, the policy is usually expressed as the probability density function $$\pi$$:

$$Policy \sim \pi(a|s)$$

However, often we do not or can not model this distribution precisely, so we use a **parameterized approximation**. What this means is that the action selection probability produced by our policy function depends on not only the state but also some approximation parameters. These are usually represented with $$\theta$$, and we write it as a subscript to our policy function to indicate that it also depends on these parameters:

$$Policy \sim \pi_\theta(a|s)$$


This $$\theta$$ could represent the parameters of a linear equation, the weights of an Artifical Neural Network or any other parameterization. The point is that those parameters are "somehow" involved in producing the action selection probabilities[^density]. The goal of the REINFORCE algorithm is now to find good values for these parameters, so our policy function produces good probabilities.

[^bayes]: If not, I recommend reading my [previous posts]({% post_url 2019-08-10-bayesian-inference %})
[^density]: More precisely the action selection probability density distribution

## Selecting better actions - Policy Gradient
Sow how do we get better estimates for these probabilities. Recall from Bayesian statistics, that we can improve our posterior by collecting more data. In our case here, the posterior is the probability density distribution of our policy function. The data we can collect in our environment is the reward signal, which tells us how "well" our agent has faired within the environment. The agent can traverse the environment and use the reward signal to inform itself about the value of individual states or actions[^states]. In the next section, I'll explain how we can integrate this reward signal into our current model of the agent. 

[^states]: There is an essential distinction between the two, but we don't worry about that in this post

### Incorporating the reward signal
After all, our goal is that our agent selects actions in a way, so it gathers the highest possible reward within the constraints of its environment. Because we are usually dealing with a stochastic decision process (the policy) and a stochastic MDP, we want the highest possible **expected reward**. So we are back to statistics again. Recall that for the discrete case, the expected value is the sum of all values weighted by their probability. Our policy function defines the probability of the agent selecting an action within a particular state. The environment then rewards or punishes our agent with reward signals when entering states as a result of its actions. Taking those two pieces of information, we can write our expected rewards in terms of the policy of the agent:

$$E(X) = \sum_{i}^{n}\pi_\theta(a_i|s_i)*r_i$$

Here $$r_i$$ is the reward signal received for the action $$a_i$$ at state $$s_i$$.
{: .notice--info}

Unfortunately, we usually do not know, or the environment doesn't give us the specific reward signal for each action-state transition. In practice, we usually have to work with estimates. There are many different strategies of selecting a good approximation for the reward signal, like for instance, the Advantage function, Q-Values, Average Returns[^returns] and many more. However, I save explaining these for a different post. For now, it suffices to say that there is some return value, which is high if we entered a beneficial state and low if it is a harmful one.

$$E(X) = \sum_{i}^{n}\pi_\theta(a_i|s_i)*R_i$$
  
Where $$R_i$$ is some return value indicating good or bad action selection
{: .notice--info}

[^returns]: [This excellent article](https://towardsdatascience.com/an-intuitive-explanation-of-policy-gradient-part-1-reinforce-aa4392cbfd3c) by Adrien Lucas Ecoffet does a great job in explaining REINFORCE and also the different return values.

### Updating our Belief with Reward Signal
So now the big question is, how do we update the parameters $$\theta$$ of our policy, so actions that result in higher returns become more likely. From calculus, we know that taking the [gradient](https://betterexplained.com/articles/vector-calculus-understanding-the-gradient/) of a function gives us the **direction of highest increase** - a.k.a the slope of a function. So taking the gradient with respect to $$\theta$$ from our expected reward formula should give us the direction in which we need to update the parameters $$\theta$$ to maximize our expected return. 

$$E(X) = \sum_{i}^{n}\nabla_\theta\pi_\theta(a_i|s_i)*R_i$$

The $$\nabla_\theta$$ means taking the gradient of the following function with respect to $$\theta$$. How this is done in details is not that important for understanding the intuition of REINFORCE. It suffices to know that the gradient gives you the direction of highest increase.
{: .notice--info}

We can also think of the problem in standard Machine Learning terms, which is also what is usually done in practice when implementing such algorithms. In that case, we interpret our expected reward function as a **negative loss** function, and we try to **minimize the negative return**. The reason for this is rather pragmatic, as it allows you to use a machine learning framework like [tensorflow](https://www.tensorflow.org) or [PyTorch](https://pytorch.org) to do the gradient updates for you. 

$$Loss \to \mathcal{L} = -\sum_{i}^{n}\nabla_\theta\pi_\theta(a_i|s_i)*R_i$$ 

### Making our AI curious
We're almost done with describing our REINFORCE agent, but there is still one problem remaining we need to tackle. Our current algorithm is prone to get stuck at local minima, especially when the initial values for $$\theta$$ are selected unfavourable. In Bayesian terms, it means that when our initial prior beliefs are chosen poorly, our posterior doesn't converge to its true value. A poorly chosen prior in this sense would select a high probability for an action that only produces a small reward. Consequently, our agent would rarely try any other potential high reward action because the small reward keeps bumping up the action.

To get an intuitive feel for this problem, imagine an agent that has the choice between two actions and two states: 
Either watch TV and get a small reward or go outside and play with the other agents and get an enormous reward. 

![TV & play outside MDP]({{ site.url }}{{ site.baseurl}}/assets/images/reinforce-tv-and-play-mdp.png)

The expressions $$R:1$$ and $$R:10$$ indicate a reward signal of 1 and 10 when entering the states.
{: .notice--info}

Now imagine our agent is a couch potato and initially already prone to watch TV. Because it has such a high chance of selecting watching TV giving it a small amount of pleasure it keeps binge-watching. The small rewards drown out the high rewards of playing outside. Hence it is very unlikely to select playing with the other agents. Moreover, even if it does, the value it assigns to it is comparatively small to the "watch TV" state, because it does it so rarely. How can we get our agent out of the house?

Well we can give the returns of rare action more weight when they are chosen. This can simply be done by deviding our gradient formula by the action probabilities:

$$E(X) = \sum_{i}^{n}\frac{\nabla_\theta\pi_\theta(a_i|s_i)*R_i}{\pi_\theta(a_i|s_i)}$$ 

Now when our agent selects an action that is four times as likely as another action, it only receives a quarter of its return compared to the other. In a sense, this way, the agent is more curious about actions it hasn't tried before because now it gets potentially a higher reward from it. Of course, negative rewards are also scaled that way. Meaning, when our agent tries a rare action and gets severely punished for it, it is very unlikely to try again, which is what we'd expect.

### The log-likelihood trick
There is one last bit of math missing from our formula, to arrive at the usual expression for REINFORCE. We can optimize it a bit by getting rid of the division to make it easier to compute in practice. To do so, we need to remember from calculus that the log of a gradient of a function is equal to the gradient divided by that function:

$$\nabla log f(x) = \frac{\nabla f(x)}{f(x)}$$

Now it is easy to see that we can simply replace the division by taking the logarithm of our policy gradient in our function:

$$E(X) = \sum_{i}^{n}\nabla_\theta log \pi_\theta(a_i|s_i)*R_i$$ 

This usage of the log is what is known as the [log-likelihood trick](https://math.stackexchange.com/questions/2554749/whats-the-trick-in-log-derivative-trick) in AI and machine learning.

## Conclusion
There it is, the expected return of the REINFORCE algorithm. We can also state it as a minimizable loss function when we want to plug it into one of the machine learning frameworks:

$$\mathcal{L} = -\sum_{i}^{n}\nabla_\theta log \pi_\theta(a_i|s_i)*R_i$$ 

In a future post, I'll talk about how you can implement this algorithm using one of the frameworks already mentioned.

