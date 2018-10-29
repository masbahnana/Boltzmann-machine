## Boltzmann-machine

Boltzmann machines are unsupervised, energy-based probabilistic models (or generators). This means that they associate an energy for each configuration of the variables that one wants to model. Intuitively, learning in these models corresponds to associating more likely configurations to lower energy states. In these states there are units that we call visible, denoted by vv, and hidden units, denoted by hh. To make this more accurate, think of the Boltzmann Machine below as representing the possible states of a party. Each white dot corresponds to a person we know and each blue dot, the one we do not know. These points are assumed to be 1 if the person is a party and 0 if he is absent. The probability of a person going to the party depends on the likelihood of all other people going to the party.

We can think of connections as the relationship between people. Thus, the probability of a person going to the party depends on these connections, but only counts the connections of the people present (i.e. with 1 in the point). For example, let's say that the connection between v1v1 and v4v4 is negative, indicating that these people do not like each other. So, if one of them is the party, the probability of the other goes down. Note that some connections may be close to zero, indicating that people are indifferent to each other. In this case, the presence of one person at the party does not directly influence the probability of the presence of the other, but there may still be indirect influence, through other people. Lastly, there is a state of the party where most of the people present do not like each other. This state is quite voltage or energy and there is a tendency for it not to occur frequently. Boltzmann's machines capture this by putting little probability in states with a lot of energy.

With this example you may have realized that Boltzmann machines are extremely complicated. After all, to know the probability that a unit is connected (be 1), one must know the state of others, since there may be indirect relations. In fact, Boltzmann machines are so complicated that they have yet to prove practical utility. So we will have to restrict them in some way. Restricted Boltzmann Machines fulfill this role. They are Boltzmann Machines on the condition that there are no direct connections between the visible units nor between the hidden ones. This makes them simpler and more practical, but also less intuitive; our example of the party does not make much sense when only known people only interact directly with unknown people. Instead, unfortunately, I will have to provide a more abstract intuitive explanation.

Despite the restriction, Restricted Boltzmann Machines, in theory, can represent any phenomenon we want, as long as it has hidden units hh enough. The visible units in this case are the variables whose interaction with each other we want to understand. With the MBR, we forced the relation between the visible units to happen indirectly, through the hidden units. Thus, the more hidden units, the greater the ability of the MBR to capture complex interactions between variables. Despite the restriction, Restricted Boltzmann machines theoretically can represent any phenomenon we want, as long as it has hidden units hh enough. The visible units in this case are the variables whose interaction with each other we want to understand. With the MBR, we forced the relation between the visible units to happen indirectly, through the hidden units. Thus, the more hidden units, the greater the MBR's ability to capture complex interactions between variables.

## Mathematical Formulation

In statistical terms, MBR define a probability distribution:

#p (vv) = e-E (vv, hh) Z

#p (vv) = e-E (vv, hh) Z


in which ZZ is the normalizing factor, also called the partition function, Σv, he-E (v, hv, h) Σv, he-E (v, hv, h). The cost for optimization is then simply the negative of the loglog probability


#L (θθ) = - 1NΣi = 0Nlogp (vvi)

#L (θθ) = - 1NΣi = 0Nlog⁡p (vvi)


Training these models is equivalent to using downward stochastic gradient in empirical loglog probability and maximizing loglog likelihood. For the Restricted Boltzmann Machines, energy is given by


#E (hh, xx) = - bb⋅vv-cc⋅hh-hhTWWvv

#E (hh, xx) = - bb⋅vv-cc⋅hh-hhTWWvv


where bbbb and cccc are bias terms of the visible and hidden layers, respectively. Note how the energy is linear in the parameters, which gives us simple and efficient derivatives of computing. The constraint on MBRs relates to the fact that there are no connections between the hidden units nor between the visible units. As a consequence, the state of the hidden units is conditionally independent, given the visible state and the visible state is conditionally independent given the hidden state. In more intuitive terms, if we have the hidden state, we can withdraw from the visible state efficiently, since we do not have to worry about how different variables of that state interact with each other and vice versa.


#P (vv | hh) = Πp (hi | vv)

#P (vv | hh) = Πp (hi | vv)

#P (hh | vv) = Πp (vi | hh)

#P (hh | vv) = Πp (vi | hh)


Here, we will see Binary Restricted Boltzmann Machines. This means that each unit will be on or off and the probability of this is given by the sigmoid activation of each unit, or neuron:

#P (hi = 1 | vv) = σ (ci + wiwivv)

#P (hi = 1 | vv) = σ (ci + wiwivv)

#P (vi = 1 | hh) = σ (bi + wiwiThh)

#P (vi = 1 | hh) = σ (bi + wiwiThh)

With this binary specification, the loglog probability gradient takes on a particularly interesting form. It is not the purpose of this tutorial to derive this gradient, even because we will compute derivatives automatically, with self-differentiation. So, I'll just put the final result:

#∂θθ∂logp (vvn) = E [∂∂θ-E (vv, hh) ||vv = vvn] -E [∂∂θ-E (vv, hh)]

#∂θθ∂log⁡p (vvn) = E [∂∂θ-E (vv, hh) | vv = vvn] -E [∂∂θ-E (vv, hh)]


The first term of this derivative is called the positive phase because its role is to increase the probability of the data. You can think of it as the average of the energy derivative when samples of the data are coupled in place of the visible units. The second term is what we call the negative phase because its role is to reduce the probability of sample generated by the model. You can think of it as the average of the energy derivative when there are no coupled samples in place of the visible units. For those interested, the development of these derivatives can be found in these lecture notes of the University of Toronto course Introduction to Neural Networks and Machine Learning (CSC321, 2014).

Due to conditional independence, the first term relating to the negative phase can be computed directly, by simply putting samples of the data in vvvv and computing the probability of hhhh. The problem then is to compute the negative fear. It is simply the hope of all possible configurations of the XXXX data under the model distribution! Since this is usually greater than the estimated number of atoms in the universe, we will need some shortcut to compute the negative term.

Let's approximate that hope with MCMC (Monte Carlo Markov Chain), that is, we will initialize NN independent Markov Chains in the data and iteratively extract hhhh and vvvv samples. This iterative process is called Alternate Gibbs Sampling.

##CDK
Adapted from ResearchGate
Mathematically (below, superscript denotes iteration, not exponent),

#vv0n = xxnvvn0 = xxn

#hhkn~P (hh | vv = hhkn)

#hhnk~P (hh | vv = hhnk)

#vvkn~P (vv | hh = hhk-1n)

#vvnk~P (vv | hh = hhnk-1)

Then we substitute the average for the approximate

#E [∂∂θ-E (vv, hh)] ≈1NΣn = 0N∂∂θ-E (vv∞n, hh∞n)

#E [∂∂θ-E (vv, hh)] ≈1NΣn = 0N∂∂θ-E (vvn∞, hhn∞)


## Contrastive Divergence
We still need to solve a problem, which is to rotate the Markov Chain infinitely (or for a long time) to achieve the desired approximation. A rather surprising observation is that, in practice, performing only an alternating Gibbs sampling (i.e., an iteration of MCMC) is sufficient to achieve a good enough approach to the training. This one iteration is what we call Contrastive Divergence 1 or CD1. As the training occurs and we want updates of the most refined parameters, we can increase the number of iterations and train the MBR with CD3. The most common form of training is to start with CD1, then move to CD3, then CD5, and finally CD10.
