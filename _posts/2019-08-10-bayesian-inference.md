---
title: "Modelling AI Agents with Bayesian Inference"
date: 2019-08-10T06:34:30+02:00
categories:
  - blog
tags:
  - AI
  - Math
  - Statistics
  - Bayes' theorem
mathjax: true
---
I've teased in the last post, that I'd explain how Bayes' theorem is connected to AI and Reinforcement Learning. Today is that day. I'll show using a simple [k-armed bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit) example, how an agent might use Bayesian inference to model its belief about its environment. Subsequently, I'll show how the agent uses observations to update its belief.

## K-Armed Bandit Problem
First, let's recap the k-armed bandit problem. Imagine that our agent is faced with a selection of slot machine. For our simple example, we assume that each of them has a uniformly distributed chance to pay out a win. However, our agent doesn't know the specific chances to win on each slot machine. To figure out which one has the highest probability of winning, it needs to try out each of them and observe the results. Intuitively, the one where it sees the most wins is the one with the highest winning chance. However, we can use Bayes' theorem to model the belief of the winning probability of a particular slot machine. 

## Bayesian Inference
To keep things simple we'll model the winning chance of one particular slot machine. We'll represent winning with a result of 1 and loosing with a result of 0.

### Modelling the prior
To start off, we need to represent somehow our ignorance about the winning chances of the slot machine. We know that it is uniformly distributed, but we don't know the specific probability that it will pay out a win. So we start with an unknown uniformly distributed probability $$\theta$$. This is called our _"prior"_, as in our prior belief:

$$\theta \sim U[0, 1] f(\theta)=I_{0 \leq \theta \leq 1}$$

The $$I_{0 \leq \theta \leq 1}$$ is called the indicator function. It simply sais that it has the value $$\theta$$ between 0 and 1 and 0 otherwise.
{: .notice--info}

### Taking observations
Now we let our agent loose, and it will start pulling the lever of the slot machine. Let's say on its first pull the agent receives a win. How do we update our belief of the winning chances of this particular slot machine, using this new observation? First, we need to model the observation itself. What is the probability observing a win given the winning probability of the slot machine? Because we just have two outcomes, win or lose, we can model it using a simple binomial distribution:

$$f(x | \theta) = \theta^x(1-\theta)^{1-x}$$

The x, in this case, is the number of wins observed. When we set x=1, we can model the probability of observing one win.
{: .notice--info}


### Updating our belief with Bayes' theorem
So what we want to know is the **probability of** the slot machine producing **a win given our observation**. However, what we have is an observation of **a win given the winning probability** and the knowledge that the winning probability follows some unknown uniform distribution. Does this sound familiar? If this sounds like a use case for Bayes' theorem, then very good, because indeed this can be solved again by reversing the direction of the condition. In this case, we need the continuous version, as we want to know the unknown probability of a continuous uniform distribution:


$$f(\theta | x)=\frac{f(x | \theta)f(\theta)}{\int f(x | \theta)f(\theta)d\theta}$$

We call $$f(\theta \mid x)$$ the _"posterior"_, describing our updated beliefs following our observations. Now we can plug in our observation from before:

$$f(\theta | x=1) = \frac{\theta^1(1-\theta)^0 I_{0 \leq \theta \leq 1}}{\int \theta^1(1-\theta)^0I_{0 \leq \theta \leq 1}d\theta}$$

$$f(\theta | x=1) = \frac{\theta I_{0 \leq \theta \leq 1}}{\int^1_0 \theta d\theta} = \frac{\theta I_{0 \leq \theta \leq 1}}{\int^1_0 \frac{\theta^2}{2}}$$

$$f(\theta | x=1) = \frac{\theta I_{0 \leq \theta \leq 1}}{\frac{1}{2}} = 2\theta I_{0 \leq \theta \leq 1}$$

Our updated belief of the slot machines winning probability is now:

$$f(\theta | x=1) = 2\theta I_{0 \leq \theta \leq 1}$$ 

If we plot this as a function of $$\theta$$ we can intuitively see that now the winning distribution leans more toward 1. This means that given our observation, it is more likely that we will observe wins on this slot machine.

![Theta distribution graph]({{ site.url }}{{ site.baseurl }}/assets/images/bayes-inference-belief-update.jpg)

It also makes kind of sense because we just observed a win. Of course, this is not very accurate because this is only one observation. By letting the agent pull multiple times, taking various observations, we can continue to update our belief. With sufficient trials, our beliefs should converge to the actual winning probability of the slot machine.

## Selecting Actions
Now that we have modelled the agent's belief of the slot machine's winning probability, it needs to use it in some way to make decisions. There are various methods to do this. For instance, we can sample an action directly from our belief distribution. Or we can greedily select the belief distribution which has the maximum likelihood for success. There are different strategies all with their various benefits and downsides. However, I'll look into those in a future post. 

On a side note, in practice, it is often not necessary to calculate the integral of Bayes' theorem. It is just a constant factor and depending on what we want to achieve, we might ignore it. For instance, if we are only interested in how different distributions relate to each other, or some maxima or minima in our distribution, we can simply skip it. The integral in the denominator is just a normalizing factor (ensuring that probabilities add up to 1).

We can express this mathematically as $$f(\theta \mid x) \propto f(x \mid \theta)f(\theta)$$. Here $$\propto$$ means "proportional to".
{: .notice--info}

