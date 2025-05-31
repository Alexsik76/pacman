from util import manhattanDistance
from game import Directions, Actions
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
      # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
      raise Exception("Not implemented yet")
  # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
    You may reference the pseudocode for Alpha-Beta pruning here:
    en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
  """
  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
      super().__init__(evalFn, depth)
      self.initial_evaluation_printed = False
  
  def getAction(self, gameState: GameState) -> str:
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 36 lines of code, but don't worry if you deviate from this)

    # Пак-Мен - це агент 0 (MAX-гравець)
    # Ініціалізуємо alpha (найкращий варіант для MAX на шляху до кореня)
    # та beta (найкращий варіант для MIN на шляху до кореня)
    alpha = -float('inf')
    beta = float('inf')

    best_score = -float('inf')
    best_action = Directions.STOP # Дія за замовчуванням, якщо немає можливих ходів

    legal_pacman_actions = gameState.getLegalActions(0)
    if not legal_pacman_actions:
        return Directions.STOP

    # Перебираємо можливі ходи Пак-Мена на кореневому рівні
    for action in legal_pacman_actions:
        successor_state = gameState.generateSuccessor(0, action)
        # Після ходу Пак-Мена (агент 0), настає черга першого привида (агент 1).
        # Поточна глибина ходів (ply depth) стає 1 (зроблено один хід).
        # Значення цього ходу визначається тим, що зробить MIN-гравець (привид).
        score = self._get_value(successor_state, 1, 1, alpha, beta) # agentIndex=1 (перший привид), ply_depth=1

        if score > best_score:
            best_score = score
            best_action = action

        # Оновлюємо alpha для кореневого MAX-вузла.
        # Це значення alpha потім використовується наступними викликами _get_value
        # для інших кореневих ходів, дозволяючи відсічення.
        alpha = max(alpha, best_score)
    if not self.initial_evaluation_printed:
        print(f"Мінімаксна оцінка для ПОЧАТКОВОГО СТАНУ ГРИ (глибина {self.depth}): {best_score}")
        # Позначаємо, що ми вже надрукували оцінку для початкового стану
        self.initial_evaluation_printed = True
    # Інколи на поточному ході агент не може обрати найкращий варіант, і вирішує зупинитись, у той час як це заборонено.
    # У таких випадках обирається випадковий напрямок.
    if best_action == Directions.STOP and Directions.STOP not in legal_pacman_actions and legal_pacman_actions:
       # print(f"ПОПЕРЕДЖЕННЯ: Агент хотів обрати STOP, але це нелегально. Поточні легальні дії: {legal_pacman_actions}. Фінальний best_score: {best_score}")
        best_action = random.choice(legal_pacman_actions)
    return best_action

  def _get_value(self,
                 gameState: GameState,
                 agentIndex: int,
                 current_ply_depth: int,
                 alpha: float,
                 beta: float) -> float:
    """
    Рекурсивна допоміжна функція для обчислення значення стану з використанням альфа-бета відсічення.
    - gameState: поточний стан гри.
    - agentIndex: індекс агента, чия черга ходити.
    - current_ply_depth: загальна кількість індивідуальних ходів, зроблених до цього моменту на поточному шляху пошуку.
    - alpha: найкраще значення, знайдене досі для MAX-гравця на шляху до цього стану.
    - beta: найкраще значення, знайдене досі для MIN-гравця на шляху до цього стану.
    """
    num_agents = gameState.getNumAgents()
    # current_game_depth - це кількість повних раундів (Пак-Мен + всі привиди), що завершилися.
    # self.depth - це максимальна кількість повних раундів для пошуку.
    current_game_depth = current_ply_depth // num_agents

    # Базові випадки для рекурсії:
    # 1. Кінцевий стан (перемога/програш)
    # 2. Досягнуто максимальної глибини пошуку для повних раундів
    # 3. Немає дозволених ходів для поточного агента
    if gameState.isWin() or gameState.isLose() or \
       current_game_depth == self.depth or \
       not gameState.getLegalActions(agentIndex):
        return self.evaluationFunction(gameState)

    legal_actions = gameState.getLegalActions(agentIndex)

    # Визначаємо, чий хід: Пак-Мена (MAX) чи привида (MIN)
    is_pacman_turn = (agentIndex == 0)

    if is_pacman_turn: # MAX-гравець (Пак-Мен)
        value = -float('inf')
        for action in legal_actions:
            successor_state = gameState.generateSuccessor(agentIndex, action)
            # Наступним агентом буде перший привид
            next_agent = (agentIndex + 1) % num_agents
            value = max(value, self._get_value(successor_state, next_agent, current_ply_depth + 1, alpha, beta))
            if value > beta: # Умова відсічення для MAX-гравця (β-відсічення)
                return value
            alpha = max(alpha, value) # Оновлюємо alpha
        return value
    else: # MIN-гравець (Привид)
        value = float('inf')
        for action in legal_actions:
            successor_state = gameState.generateSuccessor(agentIndex, action)
            # Наступним агентом може бути інший привид або Пак-Мен
            next_agent = (agentIndex + 1) % num_agents
            value = min(value, self._get_value(successor_state, next_agent, current_ply_depth + 1, alpha, beta))
            if value < alpha: # Умова відсічення для MIN-гравця (α-відсічення)
                return value
            beta = min(beta, value) # Оновлюємо beta
        return value
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

  pacmanPos = currentGameState.getPacmanPosition()
  # Рахунок гри - базовий компонент
  # Зауваж, що ми будемо додавати/віднімати з score, а не з currentGameState.getScore() напряму,
  # щоб не накопичувати помилки, якщо функція викликається для одного й того ж стану багато разів
  # (хоча для оцінки стану це менш критично, ніж для дії)
  # Тому краще почати з нуля і додати gameState.getScore() як один з компонентів.
  evaluationScore = 0
  currentScore = currentGameState.getScore()
  evaluationScore += currentScore

  # Перевірка на перемогу або програш
  if currentGameState.isWin():
      return float('inf') # Дуже велике позитивне число
  if currentGameState.isLose():
      return -float('inf') # Дуже велике негативне число (за модулем)

  # --- Фактори пов'язані з їжею ---
  foodList = currentGameState.getFood().asList()
  numFood = len(foodList)

  # 1. Штраф за кількість їжі, що залишилася
  # Чим більше їжі, тим гірша оцінка
  evaluationScore -= numFood * 10  # Вага: -10 за кожну одиницю їжі

  # 2. Бонус за близькість до найближчої їжі
  if numFood > 0:
      minFoodDist = float('inf')
      for foodPos in foodList:
          dist = manhattanDistance(pacmanPos, foodPos)
          minFoodDist = min(minFoodDist, dist)
      # Чим менша відстань, тим більший бонус (використовуємо 1/відстань)
      if minFoodDist > 0: # Уникаємо ділення на нуль
          evaluationScore += (1.0 / minFoodDist) * 5 # Вага: 5
  else: # Якщо їжі немає, але гра не виграна (наприклад, ще є привиди для з'їдання на спец. карті)
      pass # Вже оброблено isWin()

    # --- Фактори пов'язані з привидами ---
  ghostStates = currentGameState.getGhostStates()
    
    # 3. Вплив активних привидів
  sumDistToActiveGhosts = 0
  numActiveGhosts = 0
  closestActiveGhostDist = float('inf')

  for ghostState in ghostStates:
      ghostPos = ghostState.getPosition()
      distToGhost = manhattanDistance(pacmanPos, ghostPos)

      if ghostState.scaredTimer == 0: # Активний привид
          numActiveGhosts += 1
          sumDistToActiveGhosts += distToGhost
          closestActiveGhostDist = min(closestActiveGhostDist, distToGhost)

          if distToGhost <= 1: # Дуже небезпечно!
              evaluationScore -= 1000 # Великий штраф
          elif distToGhost <= 3:
              evaluationScore -= (1.0 / distToGhost) * 150 # Вага для близьких
          elif distToGhost <= 7: # Привиди на середній відстані
              evaluationScore -= (1.0 / distToGhost) * 70  # Менша вага
      
      else: # Наляканий привид
          # 4. Бонус за близькість до наляканих привидів (і можливість їх з'їсти)
          # Якщо встигаємо добігти, поки він наляканий
          if ghostState.scaredTimer > distToGhost + 1: # +1 як невеликий запас
              if distToGhost == 0: # Пакмен на клітинці з наляканим привидом
                  evaluationScore += 250 # Дуже великий бонус
              else:
                  evaluationScore += (1.0 / distToGhost) * 30 # Бонус за близькість
      if ghostState.scaredTimer == 0: # Ще раз переконуємося, що привид активний
          ghostDir = ghostState.getDirection()
          currentGhostPos = ghostState.getPosition()

          if ghostDir != Directions.STOP: # Немає сенсу аналізувати, якщо привид стоїть
              nextGhostPos = Actions.getSuccessor(currentGhostPos, ghostDir)
              currentDistToPacman = manhattanDistance(currentGhostPos, pacmanPos)
              nextDistToPacman = manhattanDistance(nextGhostPos, pacmanPos)
              if nextDistToPacman < currentDistToPacman:
                  # Привид рухається в напрямку Пакмена
                  # Визначаємо поріг відстані, на якій цей фактор стає важливим
                  # Наприклад, якщо привид в межах 5-7 клітинок і рухається на нас
                  threatDistanceForDirectionPenalty = 6
                  
                  if currentDistToPacman <= threatDistanceForDirectionPenalty:
                      # Величина штрафу може залежати від того, наскільки близько привид,
                      # або бути фіксованою.
                      # Цю вагу (W_HEADING_TOWARDS) потрібно буде підібрати.
                      penaltyWeight_HeadingTowards = 10 # Наприклад
                      
                      # Можна зробити штраф сильнішим, якщо привид дуже близько І рухається до нас
                      # (додатково до основного штрафу за близькість)
                      if currentDistToPacman <= 2: # Дуже близько і прямує до нас
                          evaluationScore -= penaltyWeight_HeadingTowards * 1.5 # Збільшений штраф
                      else:
                          evaluationScore -= penaltyWeight_HeadingTowards
                      
                      # Альтернативно, штраф може залежати від відстані:
                      # evaluationScore -= (1.0 / (currentDistToPacman + 0.1)) * penaltyWeight_HeadingTowards
                      # (додаємо 0.1, щоб уникнути ділення на нуль, якщо currentDistToPacman може бути 0,
                      # хоча в цьому блоці він, ймовірно, буде >0)
  # Якщо активні привиди дуже близько, це переважує інші фактори (крім смерті)
  # Це вже враховано вище штрафом у -1000

  # --- Фактори пов'язані з капсулами ---
  capsules = currentGameState.getCapsules()
  numCapsules = len(capsules)

  # 5. Невеликий штраф за кожну залишену капсулу (щоб стимулювати використання)
  evaluationScore -= numCapsules * 15 # Вага: -15

  # 6. Бонус за близькість до капсули, якщо є активні привиди і Пакмен не "в безпеці"
  if numActiveGhosts > 0 and numCapsules > 0 and closestActiveGhostDist < 5:
      minCapsuleDist = float('inf')
      for capsulePos in capsules:
          dist = manhattanDistance(pacmanPos, capsulePos)
          minCapsuleDist = min(minCapsuleDist, dist)
        
      if minCapsuleDist > 0: # Уникаємо ділення на нуль
          evaluationScore += (1.0 / minCapsuleDist) * 10 # Вага: 10

  return evaluationScore

better = betterEvaluationFunction
