---
title: "Bayesian Statistics - The Basis of AI"
date: 2019-08-03T15:34:30+02:00
categories:
  - blog
tags:
  - AI
  - Math
  - Statistics
  - Bayes' theorem
mathjax: true
---
In this post, I'll do my best to explain the basics of Bayesian statistics and how it relates to AI and Reinforcement Learning. Bayesian statistics is one of 3 mayor mathematical frameworks that deal with uncertainty. I'll briefly explain the general idea of all 3 and how the Bayesian framework relates to it.

## Classical 
In the classical framework, we deal with equally likely outcomes, like for instance fair dice rolls. A typical classical probability question would be 'what is the probability that two dice rolls sum up to 6'.

## Frequentist
The frequentist approach to probability is to hypothesize an infinite sequence of events and ask how often certain events occur. A good example would be modelling the package loss of a network connection. In a frequentist framework, you could say 1 in 100000 packages get lost during transit. However, this approach only works if you can define an infinite sequence of events.

## Bayesian
The Bayesian approach has a more concrete perspective. It takes into account information about the problem and forms its probability based on a belief. It means that different people can estimate different probabilities for the same event. For instance, let's say I want to estimate the probability of rain tomorrow. I might look outside, and if it is just the sunniest, cocktail enticing summer day, I'd probably estimate the chances quite low. However, a meteorologist might estimate the chance much higher, as she might have access to information that I don't have. For example, she most likely has much higher expertise in the ways of the weather and also could have access to comprehensive satellite imagery.

# Bayes' Theorem
With that out of the way, we can approach Bayes' theorem now. First, I'll introduce how to model conditional probability. Then I'll show you how we can use Bayes' theorem to reverse the direction of the condition. Using a short example, I'll demonstrate how we can use it to update our initial information. 

## Conditional Probability
Conditional probability forms the basis of Bayes' theorem. The math is quite simple. The probability that A happens given B is defined as the probability that A and B happen divided by the probability that B happens (Read "$$|$$" as "given" and "$$\cap$$" as "and"): 

$$P(A | B) = \frac{P(A \cap B)}{P(B)}$$

However, it is best explained using an example, and even better, I'll use Dungeons and Dragons for it.

### Are DnD players the better statisticians?
Let's say you suspect that DnD players with all the different dices and complex game mechanics might form an intuition for statistics. To proof this hypothesis, you gather 100 random people from your region and give them a standard statistics test which 20 people pass. Then you group them into DnD and non-DnD players and count how many of them passed or failed the test. Within your random sample, you managed to find 10 DnD players, of which 5 passed the test. Now using the formula from above you can calculate the probability that someone would pass the test given she or he is a DnD player:

$$P(\text{pass} \cap \text{DnD}) = 5/100 = 0.05$$

$$P(\text{DnD}) = 10/100 = 1/10 = 0.1$$

$$P(\text{pass} | \text{DnD}) = \frac{P(\text{pass} \cap \text{DnD})}{P(\text{DnD})} = \frac{0.05}{0.1} = 0.5$$

Indeed, you manage to find a probability of whooping 50% that a DnD player would pass the test as opposed to 20% of the general population of your region. This finding seems to show you that indeed DnD and mathematical prowess are connected.

## Reversing the Question - Bayes' theorem
Sometimes it is easier to ask a question in reverse because you have only data for specific conditional events. That is when you get Bayes' theorem out of your toolbox. The formula looks like this (The expression $$A^c$$ is the negation of the event $$A$$; meaning $$A$$ is not happening):

$$P(A | B) = \frac{P(B|A)P(A)}{P(B|A)P(A)+P(B|A^c)P(A^c)}$$

However, let's look at our example again, suppose excited by your previous findings you make them public. However, soon, you receive messages questioning the validity of your study.  They claim that your tests aren't accurate enough and that your sample size is too small. You look up the accuracy of the tests and find that the probability that someone passes by pure luck is just 5%. The test supposedly is also tough, because 25% fail even though they are good statisticians - maybe they are nervous. Given this new information, you can calculate the probability that someone passes the test even though she or he is a bad statistician. You can formulate the question as follows: 
$$P(\text{bad} | \text{good})\to$$ 
What is the probability that someone passes the test given that they are a bad statistician? You know the probability that someone in your region passes the test (20%) as you have obtained this information in your previous experiment. Now you can use Bayes' theorem to factor in the new information and answer this question: 

$$P(\text{pass} | \text{bad}) = 0.05$$

$$P(\text{fail} | \text{good}) = 0.25$$ 

$$P(\text{pass}|\text{good}) = 1 - P(\text{fail} | \text{good}) = 0.75$$

$$P(\text{bad}) = 1 - P(\text{good}) = 1 - 0.2 = 0.8$$

$$P(\text{bad} | \text{pass}) = \frac{P(\text{pass} | \text{bad})P(\text{bad})}{P(\text{pass} | \text{bad})P(\text{bad}) + P(\text{pass}|\text{good})P(\text{good})}$$

$$P(\text{bad} | \text{pass}) = \frac{0.05*0.8}{0.05*0.8+0.75*0.2} \approx 0.21$$

As we can see the probability that someone is a lousy statistician but passed the test is 21%! Is this surprising? It was to me when I first learned about this branch of math. The problem is that the false positives in a large population are quite significant, and our group of DnD players is comparatively small. Which means there is a good chance that we selected them from the portion of the population that just got lucky. It is the reason why a sufficient sample size is so important (!!!). 

## Updated Beliefs
As we have seen, we can use Bayes' theorem to update our belief of the probability of certain events occurring by factoring in new information. To those already familiar with Reinforcement Learning, this should ring a bell. Usually, RL agents try to maximize the expected value of a reward signal. The expected value is defined as the weighted sum of the values of all events with the weights being the probabilities of those events:

$$E(X) = \sum_{i=1}^{n}x_i*P(X=x_i)$$ 

When the agent is exploring the environment, it gathers new information and updates its belief of the probabilities of the events it might encounter. Now that we know the basics of Bayes' theorem, we can look into how exactly it connects to Reinforcement Learning, but I'll reserve this discussion for a future post. For those who want to dive deeper, there is a more [general formulation](https://en.wikipedia.org/wiki/Bayes'_theorem#Extended_form) of Bayes' theorem that takes into account multiple events. I can also recommend the Coursera course ["Bayesian Statistics: From Concept to Data Analysis"](https://www.coursera.org/learn/bayesian-statistics), which forms the basis of this post.

