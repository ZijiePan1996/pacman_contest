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
               first = 'AtkAgent', second = 'DefAgent'):
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
    op = self.getclosestgoast(gameState)[1]
    if op == None:
    	return 0
    time = op.scaredTimer
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
    ##decitiontree of atkagent

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

    ##decitiontree of defagent
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
    enemies = [gameState.getAgentState(i) for i in agent.getOpponents(gameState)]
    self.ene = [e.getPosition() for e in enemies if not e.isPacman and e.getPosition() != None and e.scaredTimer == 0]

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
      if not self.walls[nx][ny] and nx>0 and ny>0 and nx<self.maxwidth and ny<self.maxheight and (nx,ny) not in self.ene:
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
