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

from game import Agent

MAX_NUM = 1e6
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
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
        closestDist = MAX_NUM
        foodPos = newFood.asList() # returns food grid booleans as a list
        foodSize = currentGameState.getFood().count()

        if len(foodPos) == foodSize:
            for food in foodPos:
                if manhattanDistance(food, newPos) < closestDist:
                    closestDist = manhattanDistance(food, newPos)
        else:
            closestDist = 0
        for ghost in newGhostStates:
            closestDist += 2 ** (2-manhattanDistance(ghost.getPosition(), newPos))        
        return -closestDist

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

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"       
        def miniMaxFunction(state, depth, agentIndex):

          score = MAX_NUM
          if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          else:
            for action in state.getLegalActions(agentIndex):
              if agentIndex != gameState.getNumAgents()-1:
                temp = miniMaxFunction(state.generateSuccessor(agentIndex, action), depth, agentIndex+1)
                score = min(score, temp)
              else:
                temp = findMaxSc(state.generateSuccessor(agentIndex, action), depth-1)
                score = min(score, temp)
          return score

        def findMaxSc(state, depth):

          score = -MAX_NUM
          if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          else:
            for action in state.getLegalActions(0):
              temp = miniMaxFunction(state.generateSuccessor(0, action), depth, 1)
              score = max(score, temp)
          return score

        score = -MAX_NUM 
        for action in gameState.getLegalActions(0):
          temp = miniMaxFunction(gameState.generateSuccessor(0, action), self.depth, 1)
          if temp > score:
            score = temp
            finalActions = action
        return finalActions    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBetaFunction(state, depth, agentIndex, alpha, beta):
          
          score = MAX_NUM
          if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          else:
            for action in state.getLegalActions(agentIndex):
              if agentIndex == state.getNumAgents() - 1:
                temp = findMaxSc(state.generateSuccessor(agentIndex, action), depth-1, alpha, beta)
                score = min(score, temp)
              else:
                temp = alphaBetaFunction(state.generateSuccessor(agentIndex, action), depth, agentIndex+1, alpha, beta)
                score = min(score, temp)
              if score < alpha: 
                return score
              else:
                beta = min(beta, score)
          return score
          
        def findMaxSc(state, depth, alpha, beta):   

          score = -MAX_NUM
          if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          else:
            for action in state.getLegalActions(0):
              temp = alphaBetaFunction(state.generateSuccessor(0, action), depth, 1, alpha, beta)
              score = max(score,temp)
              if score > beta: 
                return score
              else:
                alpha = max(alpha, score)
          return score

        alpha = -MAX_NUM
        score = -MAX_NUM
        beta = MAX_NUM
        
        for action in gameState.getLegalActions(0):
          temp = alphaBetaFunction(gameState.generateSuccessor(0, action), self.depth, 1, alpha, beta)
          if temp > score:
            score = temp
            nextActions = action
          alpha = max(alpha, score)
        return nextActions

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

        def expectimaxFunction(state, depth, agentIndex):
          score = 0    
          if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          else:
            for action in state.getLegalActions(agentIndex):
              if agentIndex == state.getNumAgents() - 1:
                score += findMaxSc(state.generateSuccessor(agentIndex, action), depth-1)
              else:
                score += expectimaxFunction(state.generateSuccessor(agentIndex, action), depth, agentIndex+1)
          result = score / len(state.getLegalActions(agentIndex))
          return result

        def findMaxSc(state, depth):
          score = -MAX_NUM
          if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          else:
            for action in state.getLegalActions(0):
              temp = expectimaxFunction(state.generateSuccessor(0, action), depth, 1)
              score = max(score, temp)
          return score

        score = -MAX_NUM
        for action in gameState.getLegalActions(0):          
          temp = expectimaxFunction(gameState.generateSuccessor(0, action), self.depth, 1)
          if temp > score:
            score = temp
            nextActions = action
        return nextActions
        

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    If game state is win, return the MAX_NUM attribute (1e6).
    If game state is lose, return the -MAX_NUM attribute (-1e6).

    Increase score if pacman gets capsules/food.
    Increase score if pacman's scaredTimer is less than or equal to 0, decrease score otherwise.

    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
      return MAX_NUM
    if currentGameState.isLose():
      return -MAX_NUM
    currPos = currentGameState.getPacmanPosition()
    currGhost = currentGameState.getGhostStates()
    currCapsules = currentGameState.getCapsules()  
    score = currentGameState.getScore()
    currFood = currentGameState.getFood()

    capsuleScore, foodScore = [], [],
    capsuleScore += [50/manhattanDistance(currPos,capsule) for capsule in currCapsules]  
    foodScore += [10/manhattanDistance(currPos,food) for food in currFood]
    if len(foodScore) > 0:
      score += max(foodScore)
    if len(capsuleScore) > 0:
      score += max(capsuleScore)
    
    for ghost in currGhost:
      if ghost.scaredTimer > 0: 
          score -= 2 ** (4-manhattanDistance(ghost.getPosition(), currPos))
      else: 
          score += 2 ** (6-manhattanDistance(ghost.getPosition(), currPos))   
    return score

# Abbreviation
better = betterEvaluationFunction
