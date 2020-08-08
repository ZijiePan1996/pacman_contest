# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import distanceCalculator
from util import nearestPoint
import random, time, util
from game import Directions
from game import Actions
import game
from util import PriorityQueue
from util import Queue
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'QlearningOffensiveAgent', second = 'DefAgent'):
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

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """
    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    self.startpos=gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.midwidth = gameState.data.layout.width / 2
    self.carryfoods = 0
    self.foodnum = len(self.getFood(gameState).asList())
    self.foods = self.getFood(gameState).asList()
    self.hisdefendfoods = self.getFoodYouAreDefending(gameState).asList()
    self.height = gameState.data.layout.height
    self.hispos = None
    initmap = InitMap(self,gameState)
    self.safefoodlist,self.dangerfoodlist = initmap.gainlist()
    self.deadends = initmap.gaindeadends() 
    self.indanger = False
    '''
    Your initialization code goes here, if you need any.
    '''
  def getsafefoods(self,gameState):
    foodlist = self.getFood(gameState).asList()
    safefoods = [food for food in foodlist if food in self.safefoodlist]
    return safefoods

  def getdangerfoods(self,gameState):
    foodlist = self.getFood(gameState).asList()
    dangerfoods = [food for food in foodlist if food in self.dangerfoodlist]
    return dangerfoods

  def getsafezone(self,gameState):
    zones = []
    if self.red:
      i = self.midwidth - 1
    else:
      i = self.midwidth + 1
    boundary = [(i,j) for j in range(self.height)]
    for pos in boundary:
      if not gameState.hasWall(int(pos[0]),int(pos[1])):
        zones.append(pos)
    return zones

  def gethomedist(self,gameState):
    selfpos = gameState.getAgentState(self.index).getPosition()
    zones = self.getsafezone(gameState)
    if len(zones) > 0:
      dists = [self.getMazeDistance(selfpos, a) for a in zones]
      return min(dists)
    else:
      return 9999


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """

    '''
    You should change this in your own agent.
    '''
    problem = foodsearchproblem(gameState,self)
    return self.astarsearch(problem,gameState,self.foodhuristic)[0]
  
  def getclosestgoast(self,gameState):
    selfpos = gameState.getAgentState(self.index).getPosition()
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ene = [e for e in enemies if not e.isPacman and e.getPosition() != None]
    dist = 9999
    ghost = None
    for e in ene:
      temp = self.getMazeDistance(selfpos,e.getPosition())
      if temp < dist:
        dist = temp
        ghost = e
    return (dist,ghost)

  def getclosestpacman(self,gameState):
    selfpos = gameState.getAgentState(self.index).getPosition()
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ene = [e for e in enemies if e.isPacman and e.getPosition() != None]
    dist = 9999
    pacman = None
    for e in ene:
      temp = self.getMazeDistance(selfpos,e.getPosition())
      if temp < dist:
        dist = temp
        pacman = e
    return (dist,pacman)
   
  def getclosestenemy(self,gameState,defence=False):
    selfpos = gameState.getAgentState(self.index).getPosition()
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    if defence:
      ene = [e for e in enemies if e.isPacman and e.getPosition() != None]
    else:
      ene = [e for e in enemies if not e.isPacman and e.getPosition() != None]
    if len(ene) > 0:
      dists = [self.getMazeDistance(selfpos, a.getPosition()) for a in ene]
      return min(dists)
    else:
      return 9999
  
  def renewhispos(self,gameState):
    if len(self.observationHistory) > 1:
      pstate = self.getPreviousObservation()
      pfoodlist = self.getFoodYouAreDefending(pstate).asList()
      curfoodlist = self.getFoodYouAreDefending(gameState).asList()
      if len(pfoodlist) != len(curfoodlist):
        for food in pfoodlist:
          if food not in curfoodlist:
            self.hispos = food

  def getopscaretime(self,gameState):
    ops = self.getOpponents(gameState)
    time = gameState.getAgentState(ops[0]).scaredTimer
    if time == None or time <= 0:
      time = 0
    return time

  def nullHeuristic(self,state,gameState):
    return 0

  def astarsearch(self,problem,gameState,heuristic=nullHeuristic):
    initstate = problem.getstartstate()
    statelist = PriorityQueue()
    closed = []
    h = heuristic(initstate,gameState)
    g = 0
    f = g + h
    initnode = (initstate,[],0)
    statelist.push(initnode,f)
    while not statelist.isEmpty():
      curnode = statelist.pop()
      curstate,curpath,curcost = curnode
      if problem.isGoalState(curstate):
        if len(curpath) == 0 :
          actions = gameState.getLegalActions(self.index)
          print('empty')
          return [random.choice(actions)] 
        return curpath
      if curstate not in closed:
        closed.append(curstate)
        succ = problem.getsucc(curstate)
        if len(succ)>0:
          for s in succ:
            succpath = curpath.copy()
            succpath.append(s[1])
            g = curcost+s[2]
            succnode = (s[0],succpath,g)
            h = heuristic(s[0],gameState)
            f = g+h
            statelist.push(succnode,f)
    actions = gameState.getLegalActions(self.index)
    print('random')
    return [random.choice(actions)]
  

  def avoidghosthuristic(self,state,gameState):
    ghost = self.getclosestgoast(gameState)[1]
    if ghost == None:
      return 0
    ghostdist = self.getMazeDistance(state,ghost.getPosition())
    if state in self.deadends:
      return 300
    if ghostdist > 5:
      return 0
    elif ghostdist < 4 :
      return (5-ghostdist)**5
    else :
      return 0
  
  def avoidpacmanhuristic(self,state,gameState):
    pacman = self.getclosestpacman(gameState)[1]
    if pacman == None:
      return 0
    pacdist = self.getMazeDistance(state,pacman.getPosition())
    if state in self.deadends:
      return 100
    if pacdist > 5:
      return 0
    elif pacdist < 4 :
      return (5-pacdist)**5
    else :
      return 0

  def capsulehuristic(self,state,gameState):
    capsules = self.getCapsules(gameState)
    if state in self.deadends:
      return 300
    if len(capsules) > 0:
      dists = [self.getMazeDistance(state,cap) for cap in capsules]
      dist = min(dists)
      return dist
    return 9999
  
  def foodhuristic(self,state,gameState):
    foods = self.getFood(gameState).asList()
    if state in self.deadends:
      return 50
    if len(foods) > 0:
      dists = [self.getMazeDistance(state,food) for food in foods]
      dist = min(dists)
      return dist
    return 9999
  
  def safefoodhuristic(self,state,gamestate):
    safefoods = self.getsafefoods(gamestate)
    if state in self.deadends:
      return 300
    if len(safefoods) > 0:
      dists = [self.getMazeDistance(state,food) for food in safefoods]
      dist = min(dists)
      return dist
    return 9999

  def homehuristic(self,state,gamestate):
    zones = self.getsafezone(gamestate)
    if state in self.deadends:
      return 300
    if len(zones) > 0:
      dists = [self.getMazeDistance(state,zone) for zone in zones]
      dist = min(dists)
      return dist
    return 9999

  def mixhuristic(self,state,gamestate):
    return min([float(self.homehuristic(state,gamestate)),float(self.capsulehuristic(state,gamestate))])

  def defendhuristic(self,state,gamestate):
    pacman = self.getclosestpacman(gamestate)[1]
    if pacman == None:
      return 9999
    pacdist = self.getMazeDistance(state,pacman.getPosition())
    return pacdist

  def hisposhuristic(self,state,gamestate):
    return self.getMazeDistance(state,self.hispos)

############################################
#              Attack Agent                #
############################################

class AtkAgent(DummyAgent):
  def chooseAction(self, gameState):
    nsfood = self.getsafefoods(gameState)
    nfood = self.getFood(gameState).asList()
    ndfood = self.getdangerfoods(gameState)
    nfoodnum = len(nfood)
    nsfoodnum = len(nsfood)
    ndfoodnum = len(ndfood)
    ngoastdist = self.getclosestenemy(gameState,False)
    carrynum = gameState.getAgentState(self.index).numCarrying
    remaintime = gameState.data.timeleft
    """
      if ngoastdist < 6:
      print('alart')
      print(carrynum)
    """
    capnum = len(self.getCapsules(gameState))
    scaretime = self.getopscaretime(gameState)
    if nfoodnum < 4 and carrynum != 0:
      problem = homeproblem(gameState,self)
      return self.astarsearch(problem, gameState, self.avoidghosthuristic)[0]
    if (remaintime - self.gethomedist(gameState)) < 70:
      print('timeup')
      problem = homeproblem(gameState,self)
      return self.astarsearch(problem, gameState, self.avoidghosthuristic)[0]
    if carrynum == 0 or scaretime > 5:
      self.indanger = False
    if nfoodnum == 0:
      problem = homeproblem(gameState,self)
      return self.astarsearch(problem, gameState, self.avoidghosthuristic)[0]
    if self.indanger:
      problem = safeproblem(gameState,self)
      return self.astarsearch(problem, gameState, self.avoidghosthuristic)[0]
    
    if scaretime > 20:
      problem = foodsearchproblem(gameState,self)
      return self.astarsearch(problem,gameState,self.foodhuristic)[0]
    if nsfoodnum > 0:
      if ngoastdist < 6 and scaretime < 4 and carrynum > 0:
        self.indanger = True
        problem = safeproblem(gameState,self)
        return self.astarsearch(problem, gameState, self.avoidghosthuristic)[0]
      if (ngoastdist - carrynum + 1) < 0 or carrynum >= 7:
        self.indanger = True
        problem = homeproblem(gameState,self)
        return self.astarsearch(problem, gameState, self.avoidghosthuristic)[0]
      else:
        problem = safefoodproblem(gameState,self)
        return self.astarsearch(problem, gameState, self.avoidghosthuristic)[0]

    if ndfoodnum > 0:
      if self.indanger:
        problem = safeproblem(gameState,self)
        return self.astarsearch(problem, gameState, self.avoidghosthuristic)[0]
      else:
        if capnum > 0 and carrynum == 0:
          problem = capsuleproblem(gameState,self)
          return self.astarsearch(problem, gameState, self.avoidghosthuristic)[0]
        if scaretime > 10 or ngoastdist > 6:
          problem = foodsearchproblem(gameState,self)
          return self.astarsearch(problem, gameState, self.avoidghosthuristic)[0]
        if carrynum == 0 :
          problem = foodsearchproblem(gameState,self)
          return self.astarsearch(problem, gameState, self.avoidghosthuristic)[0]
        if carrynum > 0 and (ngoastdist < 6 or scaretime <= 10):
          self.indanger = True
          problem = safeproblem(gameState,self)
          return self.astarsearch(problem, gameState, self.avoidghosthuristic)[0]
    else:
      actions = gameState.getLegalActions(self.index)
      print('random')
      return random.choice(actions)




############################################
#             Defence Agent                #
############################################

class DefAgent(DummyAgent):
  def chooseAction(self, gameState):
    nsfood = self.getsafefoods(gameState)
    nfood = self.getFood(gameState).asList()
    ndfood = self.getdangerfoods(gameState)
    nfoodnum = len(nfood)
    nsfoodnum = len(nsfood)
    ndfoodnum = len(ndfood)
    carrynum = gameState.getAgentState(self.index).numCarrying
    ngoastdist = self.getclosestenemy(gameState,False)
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invnum = len([e for e in enemies if e.isPacman])
    nearestinv = self.getclosestenemy(gameState,True)
    ghosttime = gameState.getAgentState(self.index).scaredTimer
    scaretime = self.getopscaretime(gameState)
    self.renewhispos(gameState)
    if invnum < 1:
      if carrynum < 3 and nfoodnum and (ngoastdist > 4 or scaretime > 7):
        problem = foodsearchproblem(gameState,self)
        if scaretime > 30:
           return self.astarsearch(problem,gameState,self.foodhuristic)[0]
        return self.astarsearch(problem, gameState, self.avoidghosthuristic)[0]
      else:
        problem = homeproblem(gameState,self)
        return self.astarsearch(problem, gameState, self.avoidghosthuristic)[0]
    else:
      if nearestinv < 6:
        problem = invaderproblem(gameState,self)
        print('chase')
        if ghosttime > 1:
          problem = keepdisproblem(gameState,self)
          return self.astarsearch(problem, gameState, self.avoidghosthuristic)[0]
        else:
          return self.astarsearch(problem, gameState, self.avoidghosthuristic)[0]
      elif self.hispos != None:
        problem = hisposproblem(gameState,self)
        print(self.hispos)
        return self.astarsearch(problem, gameState, self.hisposhuristic)[0]
      else:
        actions = gameState.getLegalActions(self.index)
        print('random')
        return random.choice(actions)


############################################
#             Search Problem               #
############################################
class possearchproblem:
  def __init__(self,gameState,agent):
    self.startstate = gameState.getAgentState(agent.index).getPosition()
    self.walls = gameState.getWalls()
    self.maxwidth = int(gameState.data.layout.width)
    self.maxheight = int(gameState.data.layout.height)

  def getstartstate(self):
    return self.startstate
    
  def isGoalState(self,state):
    util.raiseNotDefined()

  def getsucc(self, state):
    succ = []
    for act in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = state
      dx, dy = Actions.directionToVector(act)
      nx, ny = int(x + dx), int(y + dy)
      if not self.walls[nx][ny] and nx>0 and ny>0 and nx<self.maxwidth and ny<self.maxheight:
        ns = (nx,ny)
        cost = 1
        succ.append((ns,act,cost))
    return succ

class foodsearchproblem(possearchproblem):  
  def __init__(self,gameState,agent):
    super(foodsearchproblem,self).__init__(gameState,agent)
    self.foods = agent.getFood(gameState).asList()
    self.caps = agent.getCapsules(gameState)
  
  def isGoalState(self,state):
    return state in self.foods or state in self.caps

class safefoodproblem(possearchproblem):
  def __init__(self,gameState,agent):
    super(safefoodproblem,self).__init__(gameState,agent)
    self.sfoods = agent.getsafefoods(gameState)

  def isGoalState(self,state):
    return state in self.sfoods

class safeproblem(possearchproblem):
  def __init__(self,gameState,agent):
    super(safeproblem,self).__init__(gameState,agent)
    self.zones = agent.getsafezone(gameState)
    self.capsules = agent.getCapsules(gameState)
  
  def isGoalState(self,state):
    return state in self.capsules or state in self.zones

class homeproblem(possearchproblem):
  def __init__(self,gameState,agent):
    super(homeproblem,self).__init__(gameState,agent)
    self.zones = agent.getsafezone(gameState)
  
  def isGoalState(self,state):
    return state in self.zones

class capsuleproblem(possearchproblem):
  def __init__(self,gameState,agent):
    super(capsuleproblem,self).__init__(gameState,agent)
    self.capsules = agent.getCapsules(gameState)
  
  def isGoalState(self,state):
    return state in self.capsules

class hisposproblem(possearchproblem):
  def __init__(self,gameState,agent):
    super(hisposproblem,self).__init__(gameState,agent)
    self.hispos = agent.hispos

  def isGoalState(self,state):
    return state == self.hispos

class invaderproblem(possearchproblem):
  def __init__(self,gameState,agent):
    super(invaderproblem,self).__init__(gameState,agent)
    enemies = [gameState.getAgentState(agentIndex) for agentIndex in agent.getOpponents(gameState)]
    self.invaders = [enemy.getPosition() for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]

  def isGoalState(self,state):
    return state in self.invaders

class keepdisproblem(possearchproblem):
  def __init__(self,gameState,agent):
    super(keepdisproblem,self).__init__(gameState,agent)
    enem = [gameState.getAgentState(i) for i in agent.getOpponents(gameState)]
    self.inv = [enemy.getPosition() for enemy in enem if enemy.isPacman and enemy.getPosition() != None]
    self.agent = agent

  def isGoalState(self,state):
    for i in self.inv:
      if self.agent.getMazeDistance(state,i) == 2:
        return True
    return False


######################################
#          Map Initializer           #
######################################

class InitMap:
  def __init__(self,agent,gameState):
    self.foods = agent.getFood(gameState).asList()
    self.zones = agent.getsafezone(gameState)
    self.walls = gameState.getWalls()
    self.maxheight = gameState.data.layout.height
    self.maxwidth = gameState.data.layout.width

  def gainlist(self):
    safefoodlist = []
    dangerfoodlist = []
    for food in self.foods:
      if self.issafefood(food):
        safefoodlist.append(food)
      else:
        dangerfoodlist.append(food)
    return (safefoodlist,dangerfoodlist)

  def gaindeadends(self):
    deadendlist = []
    i = self.maxwidth/2
    for i in range(int(i),self.maxwidth):
      for j in range(self.maxheight):
        if not self.walls[int(i)][int(j)]:
          if not self.issafefood((int(i),int(j))):
            deadendlist.append((int(i),int(j)))
    return deadendlist

  def issafefood(self,foodpos):
    neighbour = []
    validneighbour = []
    counter = 0 
    x,y = foodpos
    neighbour.append((x+1,y+1))
    neighbour.append((x+1,y-1))
    neighbour.append((x-1,y+1))
    neighbour.append((x-1,y-1))
    for i in neighbour:
      nx , ny = i
      if nx>0 and nx<self.maxwidth and ny>0 and ny<self.maxheight:
        if not self.walls[nx][ny]:
           path = self.findpath(i)
           if len(path)>0:
             counter+=1
    return counter>1

  def isgoal(self,pos):
    return pos in self.zones
  
  def getsucc(self,pos):
    succ = []
    for act in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = pos
      dx, dy = Actions.directionToVector(act)
      nx, ny = int(x + dx), int(y + dy)
      if not self.walls[nx][ny] and nx> 0 and ny> 0 and nx<self.maxwidth and ny<self.maxheight:
        ns = (nx,ny)
        succ.append((ns,act))
    return succ

  def findpath(self,foodpos):
    initpos = foodpos
    nodes = Queue()
    nodes.push((initpos,[]))
    closed = []
    while not nodes.isEmpty():
      curnode,acts = nodes.pop()
      if curnode in closed:
        continue
      closed.append(curnode)
      if self.isgoal(curnode):
        if len(acts) == 0:
          return ['STOP'] 
        return acts
      succs = self.getsucc(curnode)
      for node in succs:
        newacts=acts.copy()
        newacts.append(node[1])
        nodes.push((node[0],newacts))
    return []


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

