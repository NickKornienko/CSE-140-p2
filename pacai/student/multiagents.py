from cmath import inf
import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.distance import manhattan, maze
from pacai.core.directions import Directions


class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best.
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # find distances to each ghost
        ghostDistances = []
        for ghost in newGhostStates:
            ghostDistances.append(manhattan(newPosition, ghost._position))

        # find the distance to the closest food and return it as eval
        # distance is negated since smaller numbers are defined as better options
        foodDistances = []
        for food in oldFood:
            if food:
                d = manhattan(newPosition, food)
                foodDistances.append(d)
        foodDistances.sort()
        eval = -foodDistances[0]

        # avoid running into ghosts at all costs
        for ghostDistance in ghostDistances:
            if ghostDistance < 2:
                eval = -999999

        return eval


class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using
        `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
        and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
        """

        # find max for pacman
        _, action = self.maxValue(gameState, 0, 0)
        return action

    def maxValue(self, gameState, agent, depth):
        """
        Maximize for pacman's turn
        """
        legalActions = gameState.getLegalActions(agent)

        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)

        # max depth reached or no valid actions left, return eval func
        if depth == self.getTreeDepth() or not legalActions:
            return self._evaluationFunction(gameState), None

        bestValue = -inf
        bestAction = None
        # find action with max value
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agent, action)
            nextAgent = (agent + 1) % gameState.getNumAgents()

            # call max/min depending on if next agent is pacman or ghost
            if nextAgent == 0:
                value, _ = self.maxValue(
                    successorGameState, nextAgent, depth + 1)
            else:
                value, _ = self.minValue(
                    successorGameState, nextAgent, depth)

            if value > bestValue:
                bestValue, bestAction = value, action

        return bestValue, bestAction

    def minValue(self, gameState, agent, depth):
        """
        Minimize for ghosts' turn
        """
        legalActions = gameState.getLegalActions(agent)
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)

        # max depth reached or no valid actions left, return eval func
        if depth == self.getTreeDepth() or not legalActions:
            return self._evaluationFunction(gameState), None

        bestValue = inf
        bestAction = None
        # find action with max value
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agent, action)
            nextAgent = (agent + 1) % gameState.getNumAgents()

            # call max/min depending on if next agent is pacman or ghost
            if nextAgent == 0:
                value, _ = self.maxValue(
                    successorGameState, nextAgent, depth + 1)
            else:
                value, _ = self.minValue(
                    successorGameState, nextAgent, depth)

            if value < bestValue:
                bestValue, bestAction = value, action

        return bestValue, bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using
        `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
        and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
        """

        # find max for pacman
        _, action = self.maxValue(gameState, 0, 0, -inf, inf)
        return action

    def maxValue(self, gameState, agent, depth, alpha, beta):
        """
        Maximize for pacman's turn with a-B pruning
        """
        legalActions = gameState.getLegalActions(agent)

        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)

        # max depth reached or no valid actions left, return eval func
        if depth == self.getTreeDepth() or not legalActions:
            return self._evaluationFunction(gameState), None

        bestValue = -inf
        bestAction = None
        # find action with max value
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agent, action)
            nextAgent = (agent + 1) % gameState.getNumAgents()

            # call max/min depending on if next agent is pacman or ghost
            if nextAgent == 0:
                value, _ = self.maxValue(
                    successorGameState, nextAgent, depth + 1, alpha, beta)
            else:
                value, _ = self.minValue(
                    successorGameState, nextAgent, depth, alpha, beta)

            if value > bestValue:
                bestValue, bestAction = value, action
                alpha = max(alpha, value)

            if value >= beta:
                return value, action

        return bestValue, bestAction

    def minValue(self, gameState, agent, depth, alpha, beta):
        """
        Minimize for ghosts' turn with a-B prunin
        """
        legalActions = gameState.getLegalActions(agent)
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)

        # max depth reached or no valid actions left, return eval func
        if depth == self.getTreeDepth() or not legalActions:
            return self._evaluationFunction(gameState), None

        bestValue = inf
        bestAction = None
        # find action with max value
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agent, action)
            nextAgent = (agent + 1) % gameState.getNumAgents()

            # call max/min depending on if next agent is pacman or ghost
            if nextAgent == 0:
                value, _ = self.maxValue(
                    successorGameState, nextAgent, depth + 1, alpha, beta)
            else:
                value, _ = self.minValue(
                    successorGameState, nextAgent, depth, alpha, beta)

            if value < bestValue:
                bestValue, bestAction = value, action
                beta = min(beta, value)

            if value <= alpha:
                return value, action

        return bestValue, bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using
        `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
        and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
        """

        legalActions = gameState.getLegalActions(0)
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)

        # return stop if there are no legal options (not reached in q4)
        if not legalActions:
            return Directions.STOP

        # find max for pacman
        bestValue = -inf
        bestAction = None

        # for each action find best expected score, return best action
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(0, action)
            expectedValue = self.expectimaxValue(successorGameState, 1, 0)

            if expectedValue > bestValue:
                bestValue, bestAction = expectedValue, action

        return bestAction

    def expectimaxValue(self, gameState, agent, depth):
        """
        Expected value for ghosts' turn
        """
        legalActions = gameState.getLegalActions(agent)
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)

        # max depth reached or no valid actions left, return eval func
        if depth == self.getTreeDepth() or not legalActions:
            return self._evaluationFunction(gameState)

        values = []
        expectedValue = None
        # find all values for legal moves
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agent, action)
            nextAgent = (agent + 1) % gameState.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth

            value = self.expectimaxValue(
                successorGameState, nextAgent, nextDepth)
            values.append(value)

        # return max value for pacman, expected for ghosts
        if agent == 0:
            expectedValue = max(values)
        else:
            expectedValue = (float(sum(values)) / float(len(legalActions)))

        return expectedValue


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: < write something here so we know what you did >
    """

    position = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    # find distances to each ghost
    ghostDistances = []
    for ghost in ghosts:
        ghostDistances.append(manhattan(position, ghost._position))

    # finished eating food, done
    if not foodList:
        return currentGameState.getScore()

    # find distances to capsules
    capsuleDistances = []
    for capsule in capsules:
        capsuleDistances.append(manhattan(position, capsule))

    # find the distance to the closest food and return it as eval
    # distance is negated since smaller numbers are defined as better options
    foodDistances = []
    for food in foodList:
        foodDistances.append(manhattan(position, food))

    # consider capsules first as they have a higher value
    if capsuleDistances:
        distances = capsuleDistances
    else:
        distances = foodDistances

    eval = -min(distances)

    # avoid running into ghosts at all costs
    for ghostDistance in ghostDistances:
        if ghostDistance < 1:
            eval = -999999

    return currentGameState.getScore() + eval


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either - - they'll usually
    just make a beeline straight towards Pacman(or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
