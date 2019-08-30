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

## The Markov Decisions Process
Explain markoc decision process

## The Policy
Whats a policy, how is it modelled and the connection to Bayesian statistics. 

## Policy Gradient
Now we know the policy, how do we get better? We know from Baysian that improve estimate through data what's our data? Explain how we sample actions from it.

### Reward Signal
Explain the reward signal intuition and how it could be modelled, but leave details for other posts.

### Updating our Belief with Reward Signal
How do we update so we select better actions. Explain expected return and gradient. Use as loss function like in other ML. Explain why negative

### Stuck at Local Minima
Explain why we devide by probability density functions -> To award "curiosity" and weight unlikely actions more.

## Conclusion
Conclusion, and tease implementation for next post.
