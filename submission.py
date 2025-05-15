from util import manhattanDistance
from game import Directions
import random, util
from typing import Any, DefaultDict, List, Set, Tuple

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
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState: GameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions(agentIndex):
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


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

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.
    """
    # Початковий виклик для Пак-Мена (агент 0)
    # Ми хочемо знайти дію для Пак-Мена, яка максимізує мінімаксне значення
    # після його ходу та подальших ходів привидів.

    legal_pacman_actions = gameState.getLegalActions(0)
    if not legal_pacman_actions:
        return Directions.STOP # Якщо немає дозволених ходів, зупиняємося

    best_score = -float('inf') # Ініціалізуємо найкращий рахунок як негативну нескінченність
    best_action = Directions.STOP # Ініціалізуємо найкращу дію

    # Для кожного можливого ходу Пак-Мена
    for action in legal_pacman_actions:
        # Генеруємо стан після ходу Пак-Мена
        successor_state = gameState.generateSuccessor(0, action)
        # Обчислюємо мінімаксне значення цього стану, починаючи з ходу першого привида (агент 1)
        # Глибина гри починається з 0 повних раундів, але ми вже зробили 1 хід (Пак-Мена),
        # тому наступний агент - привид 1, і ми ще не завершили повний раунд.
        # Ми будемо відстежувати кількість *індивідуальних* ходів, зроблених у дереві пошуку.
        # Після ходу Пак-Мена зроблено 1 індивідуальний хід.
        score = self.minimax_value(successor_state, 1, 1)

        # Якщо цей рахунок кращий за поточний найкращий
        if score > best_score:
            best_score = score
            best_action = action

    return best_action # Повертаємо найкращу дію для Пак-Мена

  def minimax_value(self, gameState: GameState, agentIndex: int, total_turns_made: int) -> float:
    """
      Рекурсивна допоміжна функція для обчислення мінімаксного значення стану.
      gameState: поточний стан гри
      agentIndex: індекс агента, чия черга ходити
      total_turns_made: загальна кількість індивідуальних ходів, зроблених від початку пошуку
    """
    num_agents = gameState.getNumAgents()
    # Визначаємо поточну глибину гри (кількість повних раундів)
    # Повний раунд завершується після ходу останнього привида
    current_game_depth = total_turns_made // num_agents

    # Базові випадки рекурсії:
    # 1. Кінцевий стан гри (перемога або програш)
    # 2. Досягнуто максимальної глибини пошуку
    # 3. У поточного агента немає дозволених ходів
    if gameState.isWin() or gameState.isLose() or \
       current_game_depth == self.depth or \
       not gameState.getLegalActions(agentIndex):
        # Оцінюємо кінцевий стан за допомогою self.evaluationFunction
        return self.evaluationFunction(gameState)

    # Отримуємо дозволені ходи для поточного агента
    legal_actions = gameState.getLegalActions(agentIndex)

    # Якщо хід Пак-Мена (Макс-гравець)
    if agentIndex == 0:
        max_value = -float('inf')
        for action in legal_actions:
            # Генеруємо стан-наступник
            successor_state = gameState.generateSuccessor(agentIndex, action)
            # Рекурсивно викликаємо для наступного агента (першого привида)
            # Збільшуємо лічильник зроблених ходів
            value = self.minimax_value(successor_state, 1, total_turns_made + 1)
            max_value = max(max_value, value)
        return max_value

    # Якщо хід привида (Мін-гравець)
    else:
        min_value = float('inf')
        for action in legal_actions:
            # Генеруємо стан-наступник
            successor_state = gameState.generateSuccessor(agentIndex, action)
            # Визначаємо індекс наступного агента
            next_agent_index = (agentIndex + 1) % num_agents
            # Рекурсивно викликаємо для наступного агента
            # Збільшуємо лічильник зроблених ходів
            value = self.minimax_value(successor_state, next_agent_index, total_turns_made + 1)
            min_value = min(min_value, value)
        return min_value


    # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
    You may reference the pseudocode for Alpha-Beta pruning here:
    en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 36 lines of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState: GameState) -> float:
  """
    Your extreme, unstoppable evaluation function (problem 4). Note that you can't fix a seed in this function.
  """

  # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
  raise Exception("Not implemented yet")
  # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction
