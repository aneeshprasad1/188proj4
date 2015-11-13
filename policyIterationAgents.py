# policyIterationAgents.py
# ------------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import numpy as np
import pdb

from learningAgents import ValueEstimationAgent

class PolicyIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PolicyIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs policy iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 20):
        """
          Your policy iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        states = self.mdp.getStates()
        # initialize policy arbitrarily
        self.policy = {}
        for state in states:
            if self.mdp.isTerminal(state):
                self.policy[state] = None
            else:
                self.policy[state] = self.mdp.getPossibleActions(state)[0]
        # initialize policyValues dict
        self.policyValues = {}
        for state in states:
            self.policyValues[state] = 0

        for i in range(self.iterations):
            # step 1: call policy evaluation to get state values under policy, updating self.policyValues
            self.runPolicyEvaluation()
            # step 2: call policy improvement, which updates self.policy
            self.runPolicyImprovement()

    def runPolicyEvaluation(self):
        """ Run policy evaluation to get the state values under self.policy. Should update self.policyValues.
        Implement this by solving a linear system of equations using numpy. """
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        numStates = len(states)

        T = np.zeros((numStates, numStates))
        R = np.zeros(numStates)
        stateOrder = {}
        ind = 0
        for state in states:
            stateOrder[state] = ind
            ind += 1

        for state in states:
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                reward = self.mdp.getReward(state)
                i = stateOrder[state]
                R[i] = reward                
                action = self.policy[state]
                transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                for transition in transitions:
                    transitionState = transition[0]
                    transitionProb = transition[1]
                    j = stateOrder[transitionState]
                    T[i][j] = transitionProb
                    
        I = np.eye(numStates)
        A = I - self.discount*T
        V = np.linalg.solve(A, R)
        for state in states:
            ind = stateOrder[state]
            self.policyValues[state] = V[ind]


    def runPolicyImprovement(self):
        """ Run policy improvement using self.policyValues. Should update self.policy. """
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for state in states:
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                if actions != ():
                    EUActions = {}
                    for action in actions:
                        EUAction = self.computeQValueFromValues(state, action)
                        EUActions[action] = EUAction
                    self.policy[state] = max(EUActions, key=EUActions.get)
                else:
                    self.policy[state] = None
            else:
                self.policy[state] = None

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.policyValues.
        """
        "*** YOUR CODE HERE ***"
        EUState = 0
        reward = self.mdp.getReward(state)
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for transition in transitions:
            transitionState = transition[0]
            transitionUtility = self.policyValues[transitionState]
            transitionProbability = transition[1]
            #EUTransition = transitionProbability*(reward + self.discount*transitionUtility)
            EUTransition = transitionProbability*transitionUtility
            EUState += EUTransition
        return reward + self.discount*EUState


    def getValue(self, state):
        return self.policyValues[state]

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

    def getPolicy(self, state):
        return self.policy[state]

    def getAction(self, state):
        return self.policy[state]
