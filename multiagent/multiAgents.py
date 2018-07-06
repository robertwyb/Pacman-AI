# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent
import sys

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best a
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        curFood = currentGameState.getFood().asList()
        maxint = 999999999
        minint = -999999999

        for i in newGhostStates:            # if next position is ghost position, return minimum value
            if i.getPosition() == newPos:
                return minint
        if action == Directions.STOP:
            return minint

        distance = maxint
        for food in curFood:                # find the food closest to pacman position
            mdistance = manhattanDistance(newPos, food)
            distance = min(distance, mdistance)
        distance += 1 
        return 1.0 / distance           # use one to divide the distance to get bigger value when distance is smaller

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def minimax(self, gameState, depth, index):
        action = "None"
        win, lose = gameState.isWin(), gameState.isLose()
        num = gameState.getNumAgents()
        if win or lose or depth >= self.depth:         # check if the state is terminal
            v = self.evaluationFunction(gameState)
            return v, action
        legal_actions = gameState.getLegalActions(index)
        if index == 0:
            value = -999999999 # initialize value for pacman agent
        else:
            value = 999999999  # initialize value for ghost agent
        for a in legal_actions:
            successor = gameState.generateSuccessor(index, a)
            new_index = (index + 1) % num
            new_depth = depth
            if new_index == 0:
                new_depth += 1 # get next agent index and depth of search. If next agent is pacman increase depth by 1
            next = self.minimax(successor, new_depth, new_index)
            new_value, new_action = next[0], next[1]
            if index != 0 and value > new_value or index == 0 and value < new_value:
                value, action = new_value, a
        return value, action


    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(index):
            Returns a list of legal a for an agent
            index=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(index, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        result = self.minimax(gameState,0,0)
        return result[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphabeta(self, alpha, beta, gameState, depth, index):
        action = "None"
        win, lose = gameState.isWin(), gameState.isLose()
        num = gameState.getNumAgents()
        if win or lose or depth >= self.depth:         # check if the state is terminal
            v = self.evaluationFunction(gameState)
            return v, action
        legal_actions = gameState.getLegalActions(index)
        if index == 0:
            value = -999999999 # initialize value for pacman agent
        else:
            value = 999999999  # initialize value for ghost agent
        for a in legal_actions:
            successor = gameState.generateSuccessor(index, a)
            new_index = (index + 1) % num
            new_depth = depth
            if new_index == 0:
                new_depth += 1   # get next agent index and depth of search. If next agent is pacman increase depth by 1
            nxt = self.alphabeta(alpha, beta, successor, new_depth, new_index)
            new_value, new_action = nxt[0], nxt[1]
            if index != 0 and value > new_value:   # check if current agent is ghost
                value, action = new_value, a
                if value <= beta:    # update beta and prune
                    beta = value
                if value <= alpha:
                    return value, action

            if index == 0 and value < new_value:     # check if current agent is pacman
                value, action = new_value, a
                if value >= alpha:    # update alpha and prune
                    alpha = value
                if value >= beta:
                    return value, action
        return value, action


    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(index):
            Returns a list of legal a for an agent
            index=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(index, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        result = self.alphabeta(-999999999, 999999999, gameState, 0, 0)
        return result[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        result = self.expectimax(gameState, 0, 0)
        return result[1]

    def expectimax(self, gameState, depth, index):
        action = "None"
        win, lose = gameState.isWin(), gameState.isLose()
        num = gameState.getNumAgents()
        if win or lose or depth >= self.depth:  # check if the state is terminal
            v = self.evaluationFunction(gameState)
            return v, action
        legal_actions = gameState.getLegalActions(index)
        action_num = len(legal_actions)
        if index == 0:
            value = -999999999  # initialize value for pacman agent
        else:
            value = 0  # initialize value for ghost agent
        for a in legal_actions:
            successor = gameState.generateSuccessor(index, a)
            new_index = (index + 1) % num
            new_depth = depth
            if new_index == 0:
                new_depth += 1  # get next agent index and depth of search. If next agent is pacman increase depth by 1
            nxt = self.expectimax(successor, new_depth, new_index)
            new_value, new_action = nxt[0], nxt[1]
            if index != 0:  # check if current agent is ghost
                avg = float(new_value) / float(action_num)
                value = value + avg
            if index == 0 and value < new_value:  # check if current agent is pacman
                value, action = new_value, a

        return value, action



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      The main idea of this evaluation function is eating all food as soon as possible.
      As explained in comment, we take number of food and capsule left to make up the evaluation value while
      number of capsule has more effect than number of food. At first I was going to write an aggressive evaluation
      function to let pacman eat ghost to get higher score. However, the pacman does not behave as expected. Then I
      decide to just put capsule to priority so that pacman can eat ghost if ghost is on pacman's moving direction.

    """
    "*** YOUR CODE HERE ***"

    large, small = 999999, -999999
    all_cap = currentGameState.getCapsules()
    all_ghost = currentGameState.getGhostStates()
    all_food = currentGameState.getFood().asList()
    num_cap, num_ghost, num_food = len(all_cap), len(all_ghost), len(all_food)
    cur_pos = currentGameState.getPacmanPosition()
    cur_score = currentGameState.getScore()

    min_food, min_cap, min_ghost = large, large, large
    if num_food > 0:             # get distance to closest food
        for f in all_food:
            d = math.sqrt((cur_pos[0] - f[0]) ** 2 + (cur_pos[1] - f[1]) ** 2)
            min_food = min(d, min_food)
    else:
        min_food = small

    if num_cap > 0:             # get distance to closest capsule
        for c in all_cap:
            d = math.sqrt((cur_pos[0] - c[0]) ** 2 + (cur_pos[1] - c[1]) ** 2)
            min_cap = min(d, min_cap)
    else:
        min_cap = small

    if num_ghost > 0:           # get distance to closest ghost
        for g in all_ghost:
            gp = g.getPosition()
            d = math.sqrt((cur_pos[0] - gp[0]) ** 2 + (cur_pos[1] - gp[1]) ** 2)
            min_ghost = min(d, min_ghost)
    else:
        min_ghost = small
    # the evaluation value will be smaller when there are no food and capsule left
    # capsule has more effects than food
    # return the negative value to get the correct evaluation
    value = 50*num_food + 500*num_cap + min_ghost + min_food + min_cap - cur_score
    return -value

# Abbreviation
better = betterEvaluationFunction

