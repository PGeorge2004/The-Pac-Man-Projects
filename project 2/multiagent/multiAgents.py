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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        #Checks if the state is a winning state or a losing state
        if successorGameState.isWin() :
             return float('inf')
        if successorGameState.isLose():
            return -float('inf')
        
        #finds the manhattan distance between pacman and its closest food 
        newFood = newFood.asList()
        min = float('inf')
        for food in newFood:
            dis_pac_food = manhattanDistance(newPos,food)
            if dis_pac_food < min :
                min = dis_pac_food
        total_min = min
        
        #finds the manhattan distance between pacman and its closest ghost
        min = -1
        for ghost in newGhostStates :
            pos_ghost = ghost.getPosition()
            #if ghost is scared we do not care about its position
            if ghost.scaredTimer == 0 :
                dis_pac_ghost = manhattanDistance(pos_ghost,newPos)
                if dis_pac_ghost < 2 :
                    return -float('inf')
                if min == -1 :
                    min = dis_pac_ghost
                if dis_pac_ghost < min :
                    min = dis_pac_ghost
        #we add the 1/min to the total min because as the distance of the ghost increases the total min becomes smaller 
        #So the smaller the total min is and the bigger the min is the better the value is 
        total_min = 1.0/min + total_min  
        if total_min != 0 :
            #we add the score and 1/total min because the bigger the value returned the better the state is. 
            return successorGameState.getScore() + 1.0/total_min
        else:
            return float('inf')

def scoreEvaluationFunction(currentGameState: GameState):
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
    
    def getAction(self, gameState: GameState):
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
        return self.minimax(self.depth, gameState,0)[1]
        util.raiseNotDefined()
        
    def minimax(self,depth, currentGameState : GameState, NumOfAgent):
        #The base case
        if depth == 0 or currentGameState.isLose() or currentGameState.isWin():
            return (self.evaluationFunction(currentGameState), "NULL")
        #agent = 0 is pac man so the maximizer so we find the best move that he can take 
        # by calling recursively the minimax function
        if NumOfAgent == 0 :
            Max = -float('inf')
            Max_action = Directions.STOP
            for actions in currentGameState.getLegalActions(0):
                evaluation = self.minimax(depth,currentGameState.generateSuccessor(NumOfAgent,actions),1)
                if Max < evaluation[0]:
                    Max = evaluation[0]
                    Max_action = actions
            #I use tuples so that i can have the action that leads to that max
            #because i call the function from the root node so i do not have access to
            # to the move if i do not return it
            Max_tuple = (Max, Max_action)
            return Max_tuple
        else:
            #all other agents are ghosts so they are minimizers 
            Min = float('inf')
            #checks if the next agent is pacman so that we can decrease the depth and start the agent count from 0  
            if (NumOfAgent + 1) + 1 > currentGameState.getNumAgents():
                newAgent = 0 
                newDepth = depth - 1
            else:
                newAgent = NumOfAgent + 1
                newDepth = depth
            for actions in currentGameState.getLegalActions(NumOfAgent):
                evaluation = self.minimax(newDepth,currentGameState.generateSuccessor(NumOfAgent,actions),newAgent)
                if Min > evaluation[0]:
                    Min = evaluation[0]
                    Min_action = actions
            Min_tuple = (Min, Min_action)
            return Min_tuple
        
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimaxAB(self.depth, gameState,0,-float('inf'),float('inf'))[1]
        util.raiseNotDefined()
        
    def minimaxAB(self,depth, currentGameState : GameState, NumOfAgent,Alpha,Beta):
        if depth == 0 or currentGameState.isLose() or currentGameState.isWin():
            return (self.evaluationFunction(currentGameState), "NULL")
        if NumOfAgent == 0 :
            Max = -float('inf')
            Max_action = Directions.STOP
            for actions in currentGameState.getLegalActions(0):
                evaluation = self.minimaxAB(depth,currentGameState.generateSuccessor(NumOfAgent,actions),1,Alpha,Beta)
                if Max < evaluation[0]:
                    Max = evaluation[0]
                    Max_action = actions
                #The only difference from the minimax algorithm is that 
                #now we have the until now max(alpha) and min(beta)
                #so that if the expected values are less than alpha 
                #we prune them 
                Alpha = max(Alpha,evaluation[0])
                if Beta < Alpha:
                    break
            Max_tuple = (Max, Max_action)
            return Max_tuple
        else:
            Min = float('inf')
            if (NumOfAgent + 1) + 1 > currentGameState.getNumAgents():
                newAgent = 0 
                newDepth = depth - 1
            else:
                newAgent = NumOfAgent + 1
                newDepth = depth
            for actions in currentGameState.getLegalActions(NumOfAgent):
                evaluation = self.minimaxAB(newDepth,currentGameState.generateSuccessor(NumOfAgent,actions),newAgent,Alpha,Beta)
                if Min > evaluation[0]:
                    Min = evaluation[0]
                    Min_action = actions
                #Here because of the minimizer if the expected values are going to be less than alpha we just prune
                Beta = min(Beta,evaluation[0])
                if Beta < Alpha:
                    break
            Min_tuple = (Min, Min_action)
            return Min_tuple

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.Expectimax(self.depth, gameState,0)[1]
        util.raiseNotDefined()
        
    def Expectimax(self,depth, currentGameState : GameState, NumOfAgent):
        if depth == 0 or currentGameState.isLose() or currentGameState.isWin():
            return (self.evaluationFunction(currentGameState), "NULL")
        if NumOfAgent == 0 :
            Max = -float('inf')
            Max_action = Directions.STOP
            for actions in currentGameState.getLegalActions(0):
                evaluation = self.Expectimax(depth,currentGameState.generateSuccessor(NumOfAgent,actions),1)
                if Max <= evaluation[0] :
                    Max = evaluation[0]
                    Max_action = actions
            Max_tuple = (Max, Max_action)
            return Max_tuple
        else:
            #The only difference between minimax and expectimax is that 
            #the ghost instead of choosing the best for them move
            #they have equal chance to choose any move 
            #so we calculate the average between its children
            Total_evaluation = 0 
            i = 0
            if (NumOfAgent + 1) + 1 > currentGameState.getNumAgents():
                newAgent = 0 
                newDepth = depth - 1
            else:
                newAgent = NumOfAgent + 1
                newDepth = depth
            for actions in currentGameState.getLegalActions(NumOfAgent):
                evaluation = self.Expectimax(newDepth,currentGameState.generateSuccessor(NumOfAgent,actions),newAgent)
                Total_evaluation = evaluation[0] + Total_evaluation
                i = i +1 
            if i != 0 :
                Total_evaluation = Total_evaluation/i
            Min_tuple = (Total_evaluation, "Rand")
            return Min_tuple
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    # Useful information you can extract from a GameState (pacman.py)
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()

    #Checks if the state is a winning state or a losing state
    if currentGameState.isWin() :
        return float('inf')
    if currentGameState.isLose():
        return -float('inf')
    
    Food = Food.asList()
    min = float('inf')
    #finds the manhattan distance between pacman and its closest food 
    for food in Food:
        dis_pac_food = manhattanDistance(Pos,food)
        if dis_pac_food < min :
            min = dis_pac_food
    total_min = min
    
    chase_ghost = 0
    min = -1
    #finds the manhattan distance between pacman and its closest ghost
    for ghost in GhostStates :
        pos_ghost = ghost.getPosition()
        if ghost.scaredTimer == 0 :
            dis_pac_ghost = manhattanDistance(pos_ghost,Pos)
            if dis_pac_ghost < 2 :
                return -float('inf')
            if min == -1 :
                min = dis_pac_ghost
            if dis_pac_ghost < min :
                min = dis_pac_ghost
        else :
            #if the ghost is scared and close enough to pacman 
            #so it can eat it we promote this behavior by adding up the ghosts that can be eaten
            if ghost.scaredTimer - manhattanDistance(pos_ghost,Pos) > 0 :
                chase_ghost = chase_ghost + 1
    if min != -1 :
        total_min = 1/min + total_min   
              
    min = float('inf')
    #we also acount the distance of the closest capsule to the total min
    for capsules in currentGameState.getCapsules():
        dis_pac_Capsule = manhattanDistance(Pos,capsules)
        if dis_pac_Capsule < min or min == float('inf'):
            min = dis_pac_Capsule
    if min != float('inf'):
        total_min = min + total_min
    
    #we also add the ghost we can chase in the value that we return 
    if total_min != 0 :
        return currentGameState.getScore() + 1/float(total_min) + chase_ghost 
    else:
        return float('inf')
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
