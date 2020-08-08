from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint
import random, time, util
import game

def createTeam(firstIndex, secondIndex, isRed,
               first='QlearningOffensiveAgent', second='DefensiveReflexAgent'):

    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


###################
# Baseline Agents #
###################

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


#####################
# Q-learning Agents #
#####################


class QlearningOffensiveAgent(CaptureAgent):
    
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.startposition = gameState.getAgentPosition(self.index)
        self.qtable = {}
        self.candidates = util.Queue()
        self.inDanger = False
        self.height = gameState.data.layout.height
        self.width = gameState.data.layout.width
        self.foodEaten = 0
        self.dangerindex = 0
    
    def scaredTime(self,gameState):
        opponents = self.getOpponents(gameState)
        for opponent in opponents:
            if gameState.getAgentState(opponent).scaredTimer > 1:
             return gameState.getAgentState(opponent).scaredTimer
        return None
    
    def chooseAction(self, gameState):

        #------------ Initialisation ---------------#
        expanded = 0
        rate = 0.5 #learning rate
        gamma = 0.6 # discount factor
        actionList = ['North', 'South', 'West', 'East']
        visiting = util.Queue()
        visited = []
        currentState = self.getCurrentObservation()
        reward = 0
        finished = False
        visiting.push(currentState)

        while not finished and not visiting.isEmpty():
            currentState = visiting.pop()
            currentPosition = currentState.getAgentState(self.index).getPosition()
            visited.append(currentPosition)
            #print(currentPosition)
            #print (visited)
            legalactions = currentState.getLegalActions(self.index)
            expanded += 1

            if Directions.STOP in legalactions:
                legalactions.remove(Directions.STOP)
            
            for action in legalactions:
                self.candidates.push(action)
            
            #####################################
            if expanded > 50:
                finished = True

            if currentPosition not in self.qtable.keys():
                self.qtable[currentPosition] = [0,0,0,0]
            while not self.candidates.isEmpty():
                action = self.candidates.pop()
                indices = actionList.index(action)
                successor = currentState.generateSuccessor(self.index, action)
                currentFoodNum = len(self.getFood(currentState).asList())
                successorFoodNum = len(self.getFood(successor).asList())
                successorEval = [self.evaluate(successor, succ) for succ
                                        in successor.getLegalActions(self.index)]
                
                #if the successor has food, then eat is unless 
                if currentFoodNum - successorFoodNum == 1  and not self.inDanger:
                    reward = 999
                else:
                    reward = 0

                #update q table
                q_value = (1 - rate) * self.qtable[currentPosition][indices] + rate * (reward + gamma * max(successorEval))                
                self.qtable[currentPosition][indices] = q_value

                if successor.getAgentPosition(self.index) not in visited:
                    visiting.push(successor)
        
        currentPosition = gameState.getAgentState(self.index).getPosition()
        legalactions = gameState.getLegalActions(self.index)
        #give some chance to explore new direction
        if random.random()<0.01: 
            bestAction = random.choice(legalactions)
            indices = actionList.index(action)
        #otherwise exploit
        else:
            maxValue = -999
            for z in range(len(self.qtable[currentPosition])):
                if self.qtable[currentPosition][z] > maxValue and self.qtable[currentPosition][z] != 0:
                    maxValue = self.qtable[currentPosition][z]
            indices = self.qtable[currentPosition].index(maxValue)
            bestAction = actionList[indices]
        
        successor = gameState.generateSuccessor(self.index, bestAction)
        currentFoodNum = self.getFood(gameState).asList()
        successorFoodNum = self.getFood(successor).asList()
        successorPosition = successor.getAgentPosition(self.index)
        myPos = successor.getAgentState(self.index).getPosition()
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        selfState = successor.getAgentState(self.index).isPacman
        
        if len(ghosts) != 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
            if not self.inDanger:
                if self.scaredTime(gameState) == None:
                    if min(dists) <= 5 and selfState:
                        print("in Danger now, escaping!!!!")
                        self.inDanger = True
                    elif min(dists) > 5:
                        self.inDanger = False
                else:
                    if min(dists) <= 5 and selfState and self.scaredTime(gameState)<5:
                        print("scared time out now, quickly escaping!!!!")
                        self.inDanger = True
                    elif min(dists) > 5:
                        self.inDanger = False
        else:
            self.inDanger = False
        
        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            self.inDanger = True

        if myPos == self.startposition:
            self.inDanger = False
        
        if successorPosition in currentFoodNum and successorPosition not in successorFoodNum:
            self.qtable = {}

        if self.inDanger:
            self.qtable = {}
        return bestAction

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        # print successor
        pos = successor.getAgentState(self.index).getPosition()
        ## print pos
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        #print("this  is current feature", features)
        #print("this is current weights", weights)
        #print(features * weights)
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()

        # Compute distance to the nearest food
        if len(foodList) > 0: 
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(ghosts) > 0 :
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
            features['distanceToOpponent'] = min(dists)
        else:
            features['distanceToOpponent'] = 100
        features['distanceToStart'] = self.getMazeDistance(myPos, self.startposition)
        return features

    def getWeights(self, gameState, action):
        #print(self.scaredTime(gameState))
        if self.inDanger :
            # if self.index ==  0:
            self.dangerindex += 1
            print ("in danger", self.dangerindex)
            
            return {'distanceToFood': 0, 'distanceToOpponent': 30, 'distanceToStart': -10, 'distanceToScaredghosts': 0}
        else:
            #print ("start hunting")
            return {'distanceToFood': -1, 'distanceToOpponent': 0, 'distanceToStart': 0, 'distanceToScaredghosts': 0}



'''
def waStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has has the weighted (x 2) lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE FOR TASK 2 ***"
    from game import Directions
    stack = util.PriorityQueue() 
    closedList = []
    stack.push((problem.getStartState(),[],0),0 + heuristic(problem.getStartState(),problem)) #
    (state,toDirection,toCost) = stack.pop()
    print(str(state),str(toDirection),str(toCost))
    closedList.append((state,toCost + heuristic(problem.getStartState(),problem)))
    #keep getting successors until the goal is founds
    while not problem.isGoalState(state): 
        successors = problem.getSuccessors(state) 
        for son in successors:
            visitedExist = False
            total_cost = toCost + son[2]
            for (visitedState,visitedToCost) in closedList:
                #for question Q3, because the state actuallly has three components, the closelist will 
                #'reopen' because the state has changed
                if (son[0] == visitedState) and (total_cost >= visitedToCost): 
                    visitedExist = True
                    break

            if not visitedExist:        
                stack.push((son[0],toDirection + [son[1]],toCost + son[2]),toCost + son[2] + heuristic(son[0],problem)*2) 
                closedList.append((son[0],toCost + son[2])) 
        (state,toDirection,toCost) = stack.pop()

    return toDirection
'''

