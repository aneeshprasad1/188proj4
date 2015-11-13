# valueIterationAgents.py
# -----------------------
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

from learningAgents import ValueEstimationAgent
import collections
import time

import pdb
import numpy as np
class AsynchronousValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = collections.defaultdict(float)
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0

        "*** YOUR CODE HERE ***"
        for i in range(iterations):
            start = time.time()
            state = states[i%len(states)]
            if not mdp.isTerminal(state):
                reward = mdp.getReward(state)                
                actions = mdp.getPossibleActions(state)
                EUActions = []
                for action in actions:
                    EUAction = 0
                    transitions = mdp.getTransitionStatesAndProbs(state, action)
                    for transition in transitions:
                        transitionState = transition[0]
                        transitionUtility = self.values[transitionState]
                        transitionProbability = transition[1]
                        EUAction += transitionUtility*transitionProbability
                    EUActions.append(EUAction)
                maxEU = max(EUActions)
                updatedUtility = reward + discount*maxEU
                self.values[state] = updatedUtility
            print('Iteration: ' + str(i))
            print(sum(abs(value - 100) for state, value in self.values.items() if not self.mdp.isTerminal(state)))
            print('Time elapsed: ' + str(time.time()-start))

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        EUState = 0
        reward = self.mdp.getReward(state)
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for transition in transitions:
            transitionState = transition[0]
            transitionUtility = self.values[transitionState]
            transitionProbability = transition[1]
            #EUTransition = transitionProbability*(reward + self.discount*transitionUtility)
            EUTransition = transitionProbability*transitionUtility
            EUState += EUTransition
        return reward + self.discount*EUState

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        if actions == ():
            return
        EUActions = {}
        for action in actions:
            EUAction = self.computeQValueFromValues(state, action)
            EUActions[action] = EUAction

        return max(EUActions, key=EUActions.get)

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = collections.defaultdict(float)
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0

        "*** YOUR CODE HERE ***"
        # Compute predecessors of all states
        predecessors = {state: [] for state in states}
        for state in states:
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                for transition in transitions:
                    transitionState = transition[0]
                    transitionProb = transition[1]
                    if transitionProb > 0:
                        predecessors[transitionState].append(state)

        q = util.PriorityQueue()
        for state in states:
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                curr = self.values[state]
                qValues = [self.computeQValueFromValues(state, action) for action in actions]
                highQ = max(qValues)
                diff = abs(curr - highQ)
                q.update(state, -diff)

        times = []
        for i in range(iterations):
            #print(sum(abs(value - 100) for state, value in self.values.items() if not self.mdp.isTerminal(state)))
            start = time.time()
            if q.isEmpty():
                return
            s = q.pop()
            if not self.mdp.isTerminal(state):
                # update state
                actions = mdp.getPossibleActions(s)
                qValues = [self.computeQValueFromValues(s, action) for action in actions]
                highQ = max(qValues)
                self.values[s] = highQ
                
                for p in predecessors[s]:
                    actions = self.mdp.getPossibleActions(p)
                    curr = self.values[p]
                    qValues = [self.computeQValueFromValues(p, action) for action in actions]
                    highQ = max(qValues)
                    diff = abs(curr - highQ)
                    if diff > theta:
                        q.update(p, -diff)
            print('Iteration: ' + str(i))
            print(sum(abs(value - 100) for state, value in self.values.items() if not self.mdp.isTerminal(state)))
            elapsed = time.time()-start
            print('Time elapsed: ' + str(elapsed))
        




