"""
Q-Learning Implementation 
Written by Amanda Martin
Modified by Amanda Martin from wwagent.py:

    Modified from wwagent.py written by Greg Scott

    Modified to only do random motions so that this can be the base
    for building various kinds of agent that work with the wwsim.py 
    wumpus world simulation -----  dml Fordham 2019

    # FACING KEY:
    #    0 = up
    #    1 = right
    #    2 = down
    #    3 = left

    # Actions
    # 'move' 'grab' 'shoot' 'left' right'
"""

from random import randint
import copy
import random

qtable = [[ None for i in range(4) ] for j in range(16)]
print(qtable)
epsilon = .1
print(epsilon)

# This is the class that represents an agent
class WWAgent:

    def __init__(self):
        self.max=4 # number of cells in one side of square world
        self.stopTheAgent=False # set to true to stop th agent at end of episode
        self.position = (0, 3) # top is (0,0)
        self.directions=['up','right','down','left']
        self.facing = 'right'
        self.arrow = 1
        self.percepts = (None, None, None, None, None)
        self.map = [[ self.percepts for i in range(self.max) ] for j in range(self.max)]
        print("New agent created")
        # full symbols list
        self.symbols =  ['p03', 'p02', 'p01', 'p00', 'p10', 'p11', 'p12', 'p13', 'p23', 'p22', 'p21', 'p20', 'p30', 'p31', 'p32', 'p33',
                         'b03', 'b02', 'b01', 'b00', 'b10', 'b11', 'b12', 'b13', 'b23', 'b22', 'b21', 'b20', 'b30', 'b31', 'b32', 'b33',
                         's03', 's02', 's01', 's00', 's10', 's11', 's12', 's13', 's23', 's22', 's21', 's20', 's30', 's31', 's32', 's33',
                         'w03', 'w02', 'w01', 'w00', 'w10', 'w11', 'w12', 'w13', 'w23', 'w22', 'w21', 'w20', 'w30', 'w31', 'w32', 'w33']
        self.model = []
        self.kb = [] # conjunction of known propositions, either true (prop) or false ('n'+prop)
        self.alpha = [] # that target room is 100% safe (no wumpus or pit)
        #self.infer = [] # disjunction of propositions based on percepts
        self.visited = [self.position] # list of rooms that have already been visited
        self.unvisited = [] # list of rooms that have been determined to be safe but weren't yet explored
        self.hasMove = None # is move has been calculated but agent isn't yet facing the right direction
        self.path = [self.position]

        '''class attributes for probabilistic model checking'''
        self.n = 0 # count number of models that follow the KB
        self.m = 0 # count number of models that are safe
        self.ptable = [[ None for i in range(self.max) ] for j in range(self.max)] # holds probabilities for each room
        self.prevPos = (0,3)
        self.prevAction = None

        '''class attributes for backtracking to previously model checked rooms'''
        self.isBackTracking = False
        self.path2= []
        self.goalMove = None 
    
    # Add the latest percepts to list of percepts received so far
    # This function is called by the wumpus simulation and will
    # update the sensory data. The sensor data is placed into a
    # map structured KB for later use
    
    def update(self, percept):
        self.percepts=percept
        #[stench, breeze, glitter, bump, scream]
        if self.position[0] in range(self.max) and self.position[1] in range(self.max):
            self.map[ self.position[0]][self.position[1]]=self.percepts
        # puts the percept at the spot in the map where sensed

    # Since there is no percept for location, the agent has to predict
    # what location it is in based on the direction it was facing
    # when it moved

    def calculateNextPosition(self,action):
        if self.facing=='up':
            self.position = (self.position[0],max(0,self.position[1]-1))
        elif self.facing =='down':
            self.position = (self.position[0],min(self.max-1,self.position[1]+1))
        elif self.facing =='right':
            self.position = (min(self.max-1,self.position[0]+1),self.position[1])
        elif self.facing =='left':
            self.position = (max(0,self.position[0]-1),self.position[1])
        return self.position

    # and the same is true for the direction the agent is facing, it also
    # needs to be calculated based on whether the agent turned left/right
    # and what direction it was facing when it did
    
    def calculateNextDirection(self,action):
        if self.facing=='up':
            if action=='left':
                self.facing = 'left'
            else:
                self.facing = 'right'
        elif self.facing=='down':
            if action=='left':
                self.facing = 'right'
            else:
                self.facing = 'left'
        elif self.facing=='right':
            if action=='left':
                self.facing = 'up'
            else:
                self.facing = 'down'
        elif self.facing=='left':
            if action=='left':
                self.facing = 'down'
            else:
                self.facing = 'up'

    # calculate the best turn for the agent based on directions list
    # checks if the direction of move is one right turn away and returns right
    # else turns left (180 turns become two left turns)
    def calculateTurn(self, facing, direction):
        diff = self.directions.index(facing) - self.directions.index(direction)
        if diff == -1:
            return 'right'
        elif facing == 'left' and diff == 3:
            return 'right'
        elif diff == 1:
            return 'left'
        elif facing == 'up' and diff == -3:
            return 'left'
        else: 
            return 'left'

    # get the direction of the room you are moving to based on current room
    def getDirection(self, room):
        changex = room[1] -self.position[1]
        changey = room[0] -self.position[0]
        #print("changex: ", changex, " changey: ", changey)
        direction = None
        if changex == -1:
            direction = 'up'
        if changex == 1:
            direction = 'down'
        if changey == -1:
            direction = 'left'
        if changey == 1:
            direction = 'right'
        return direction

    # updates the KB with only known values - either 'n'+symbol if known to be false, or symbol if known to be true
    # known values include percepts in current room, no wumpus or pit in current room
    # no wumpus in adjacent rooms if no stench in current room
    # no wumpus in non adjacent rooms if stench in current room (because only one wumpus per model)
    # and no pit in adjacent room if no breeze in current room
    def updateKB(self):
        if self.position not in self.visited:
            temp = 'nw' + str(self.position[0]) + str(self.position[1])
            addToKB(temp, self.kb)
            temp = 'np' + str(self.position[0]) + str(self.position[1])
            addToKB(temp, self.kb)
            if 'stench' in self.percepts:
                addToKB('s' + str(self.position[0]) + str(self.position[1]), self.kb)
                rooms = getSurroundingRooms(self.position) # get rooms surrounding the stench
                for s in self.symbols:
                    if (int(s[1]), int(s[2])) not in rooms:
                        if s.startswith('w'):
                            addToKB('n'+ s, self.kb) #only one wumpus, so no wumpus in all rooms not
            else:
                addToKB('ns' + str(self.position[0]) + str(self.position[1]), self.kb)
                rooms = getSurroundingRooms(self.position) # get rooms surrounding current room
                for room2 in rooms:
                    temp = 'nw' + str(room2[0]) + str(room2[1])
                    addToKB(temp, self.kb)
            if 'breeze' in self.percepts:
                addToKB('b' + str(self.position[0]) + str(self.position[1]), self.kb)
                rooms = getSurroundingRooms(self.position) # get rooms surrounding the breeze
            else:
                addToKB('nb' + str(self.position[0]) + str(self.position[1]), self.kb)
                rooms = getSurroundingRooms(self.position) # get rooms surrounding current room
                for room2 in rooms:
                    temp = 'np' + str(room2[0]) + str(room2[1]) 
                    addToKB(temp, self.kb)
            self.visited.append(self.position)
        if self.position in self.unvisited:
            self.unvisited.remove(self.position)

    # function for steps taken once a valid 'move' has been found, given the new room coordinates
    def move(self, room):
        direction = self.getDirection(room)
        if self.facing == direction:
            self.hasMove = None
            action = 'move'
            self.calculateNextPosition(action)
            print("moving towards ", room)
            self.path.insert(0, room)
            self.prevPos = self.position
            self.prevAction = self.facing
        else:
            self.hasMove = room
            action = self.calculateTurn(self.facing, direction)
            #action = 'left'
            self.calculateNextDirection(action)
            print("turning ", action, " towards ", room)
        #print(self.kb)
        return action
    
    def move2(self, direction):
        room = self.calculateNextPosition(direction)
        if self.facing == direction:
            self.hasMove = None
            action = 'move'
            print("moving towards ", room)
            self.path.insert(0, room)
            self.prevPos = self.position
            self.prevAction = direction
        else:
            self.hasMove = room
            action = self.calculateTurn(self.facing, direction)
            #action = 'left'
            self.calculateNextDirection(action)
            print("turning ", action, " towards ", room)
        #print(self.kb)
        return action

    # modified action function
    # implements modelchecking through 'modelChecking' function
    def action(self):
        # test for controlled exit at end of successful gui episode
        if self.stopTheAgent:
            print("Agent exiting game")
            return 'exit' # will cause the episide to end
            
        #reflect action -- get the gold!
        if 'glitter' in self.percepts:
            print("Agent will grab the gold!")
            self.stopTheAgent=True
            updateQtable(self.directions, self.ptable, self.prevPos, self.prevAction, 1000, self.position)
            return 'grab'
        
        if self.hasMove != None:
            room = self.hasMove
            action = self.move(room)
            return action

        # backtracking to next best room to check
        if self.isBackTracking:
            if not self.path2:
                self.isBackTracking = False
            else:   
                theseRooms =  getSurroundingRooms(self.position)
                if self.goalMove in theseRooms:
                    action = self.move(self.goalMove)
                    self.path2 = [] # clear backtracking path
                    return action
                elif self.path2[0] in theseRooms:
                    action = self.move(self.path2[0])
                    self.path2.pop(0)
                    return action
                else: # case should never be met, in case of accidental infinite loop
                    print("error")
                    return 'exit'

        
        updateQtable(self.directions, self.ptable, self.prevPos, self.prevAction, -1, self.position)
        print(qtable)
        # update the KB with the knowledge you learn from current position
        self.updateKB()
        self.ptable[self.position[0]][self.position[1]] = 1.0 # if still alive, current square is 100% safe



        # add surrounding rooms to 'possiblemoves'
        possiblemoves = getSurroundingRooms(self.position)
        # model only cares for surrounding rooms plus the current room
        modelRooms = copy.deepcopy(possiblemoves)
        modelRooms.append(self.position)
        symbolsCleaned = []
        for rooms in modelRooms:
            room = str(rooms[0]) +str(rooms[1])
            for s in self.symbols:
                if s.endswith(room):
                    symbolsCleaned.append(s)
        action = None
        validMoves = []

        #You should pick a move based on the highest probability of being safe with
        #probability (1-e) with based on the Q table with e
        choice = random.choices(['ptable', 'qtable'], weights = [1-epsilon, epsilon])
        if choice == ['ptable']:
            # model check for each move
            for move in possiblemoves:
                # all visited rooms were previously model checked and are safe
                if move in self.visited:
                    validMoves.append(move)
                else:
                    if move in self.unvisited:
                        validMoves.insert(0, move)
                    # alpha is that wumpus is not in that room and pit is not in that room
                    self.alpha = [('w' + str(move[0]) + str(move[1]), False), ('p' + str(move[0]) + str(move[1]), False)]
                    # reset m and n for room
                    self.m = 0
                    self.n= 0
                    # call model check
                    value = self.modelcheck(symbolsCleaned, self.model, self.kb, self.alpha)
                    #calculate probability and update table
                    if self.n != 0:
                        prob = (self.m/self.n)
                        if self.ptable[move[0]][move[1]] != 0.0:
                            self.ptable[move[0]][move[1]] = prob
                        if prob == 1.0:
                            if move not in self.unvisited:
                                self.unvisited.insert(0, move)
                        else:
                            self.unvisited.append(move)
                    else:
                        print("room ", move, " safe in no models")
            if self.unvisited:
                for move in self.unvisited:
                    # move to 100% safe unvisited rooms first 
                    if self.ptable[move[0]][move[1]] == 1.0:
                        print("room ", move, "unvisited and safe")
                        if move in possiblemoves:
                            action = self.move(move)
                            return action
                        else:
                            print("backtracking to ", move)
                            self.isBackTracking = True
                            self.goalMove = move
                            goTo = getSurroundingRooms(move)
                            print("path: ", self.path)
                            for room in goTo:
                                if room in self.path:
                                    print(self.path)
                                    self.path2 = self.path[1:(self.path.index(room)+1)]
                                    if self.path2[0] in validMoves:
                                        action = self.move(self.path2[0])
                                        self.path2.pop(0)
                                        return action
                newMove = None              
                # if no more 100% safe rooms, find the next highest probability and move to that room
                for move in self.unvisited:
                    maxn = 0
                    if self.ptable[move[0]][move[1]] > maxn:
                        maxn = self.ptable[move[0]][move[1]]
                        newMove = move
                if newMove:
                    print("new move: ", newMove, " with prob: ", self.ptable[newMove[0]][newMove[1]])
                    if newMove in possiblemoves:
                        action = self.move(newMove)
                        return action
                    else:
                        self.isBackTracking = True
                        self.goalMove = newMove
                        goTo = getSurroundingRooms(newMove)
                        print("move: ", self.goalMove)
                        for room in goTo:
                            if room in self.path:
                                self.path2 = self.path[1:(self.path.index(room)+1)]
                                print(self.path2)
                                if self.path2[0] in validMoves:
                                    action = self.move(self.path2[0])
                                    self.path2.pop(0)
                                    return action
                if action == None and validMoves:
                    for move in validMoves:
                        if move in possiblemoves:
                            action = self.move(move)
                            return action
            print("no more safe rooms to explore")
            return 'exit'
        else:
            print("using q-learning, best move is ")
            possiblemoves = getSurroundingRooms(self.position)
            x, y = self.position
            state = ((x*4) + y)
            maxVal = 0
            maxMove = None
            for move in possiblemoves:
                newMove = self.directions.index(self.getDirection(move))
                if qtable[state][newMove] > maxVal:
                    maxVal = qtable[state][newMove]
                    maxMove = newMove
            print(maxMove)
            action = self.move2(maxMove)    
            return action


    # returns true if wumpus and pit are not in given room in the model
    def isSafe(self, alpha, model):
        isSafe = True
        for a in alpha:
            if a not in model:
                isSafe = False
        return isSafe

    # enumerates truth tables for models, and checks
    def modelcheck(self, symbols, model, KB, alpha):
        if len(symbols)==0:
            #print("-----------------------------------")
            #print(model)
            if isTrueRules(model, self.position): # checks if model obeys rules of the game
                if isTrueKB(model, self.kb): # check if model obeys senses
                    self.n+=1
                    if self.isSafe(alpha, model): # check if model is safe for room (given by alpha is true)
                        self.m+=1
                        return self.isSafe(alpha, model)
                    else:                         
                        return True
                else: # if model doesn't obey rules, model disregarded
                    return True
            else: # if model doesn't obey rules, model disregarded
                return True
        else:
            p = symbols[0]
            rest = list(symbols[1:len(symbols)])
            return self.modelcheck(rest,model+[(p,True)],KB,alpha) and self.modelcheck(rest,model+[(p,False)],KB,alpha)

def updateQtable(directions, ptable, state, action, reward, newState):
    if not state or not action:
        return
    x, y = state
    state1 = ((x*4) + y)
    action1 = directions.index(action)
    if qtable[state1][action1]: 
        x2,y2 = newState
        state2 = ((x2*4) + y2)
        print(qtable[state2])
        #maxMove = max(qtable[state2])
        qtable[state1][action1] = qtable[state1][action1] + (reward + (1*1) - qtable[state1][action1])
    else: #initialize qtable to probability value
        qtable[state1][action1] = ptable[x][y] 

# provided isTrue function
# checks validity of propositional phrases
# modified to include 'n'+symbol propositions
def isTrue(prop,model):
    #Check whether prop is true in model
    # assumes prop and model use the list/logic notation
    if isinstance(prop,str):
        if ((prop, True) not in model) and ((prop, False) not in model) and (('n'+prop, True) not in model) and (('n'+prop, False) not in model):
            return True
        elif prop.startswith('n'):
            return not isTrue(prop[1:len(prop)],model)
        else:
            return (prop,True) in model
    elif len(prop)==1:
        return isTrue(prop[0],model)
    elif prop[1]=='and':
        return isTrue(prop[0],model) and isTrue(prop[2],model)
    elif prop[1]=='or':
        return isTrue(prop[0],model) or isTrue(prop[2],model)
    elif prop[1]=='implies':
        return (not isTrue(prop[0],model)) or isTrue(prop[2],model)
    elif prop[1]=='iff':
        left = (not isTrue(prop[0],model)) or isTrue(prop[2],model)
        right= (not isTrue(prop[2],model)) or isTrue(prop[0],model)
        print(left,right)
        return (left and right)
    return False

# check if model follows all known values in the KB
def isTrueKB(model, kb):
    for prop, val in model:
        if val == False:
            if prop in kb:
                return False
        elif val == True:
            if 'n'+prop in kb:
                return False
    return True

# if alpha is more than two symbols, created nested alpha
# e.g. [ 's11', 'and', ['s12', 'and', 's01]]
def cleanAlpha(a, prop):
    if len(a) == 0:
        return a
    elif len(a) == 1:
        return a[0]
    elif len(a) == 2:
        return [a[0], prop, a[1]]
    else:
        return [a[0], prop, cleanAlpha(a[1:], prop)]
    
# gets all the adjacent rooms of 'currentRoom'
def getSurroundingRooms(currentRoom):
    nextPosMoves = []
    if currentRoom[0] > 0:
        nextPosMoves.append((currentRoom[0]-1, currentRoom[1]))
    if currentRoom[0] < 3:
        nextPosMoves.append((currentRoom[0]+1, currentRoom[1]))
    if currentRoom[1] > 0:
        nextPosMoves.append((currentRoom[0], currentRoom[1]-1))
    if currentRoom[1] < 3:
        nextPosMoves.append((currentRoom[0], currentRoom[1]+1))
    return nextPosMoves

# create an alpha list based on given params - example format: [ 's11', 'and', 's12']
def createAlpha(rooms, symbol, prop):
    alpha = []
    for room in rooms:
        temp = symbol + str(room[0]) + str(room[1])
        alpha.append(temp)
        alpha2 = cleanAlpha(alpha, prop)
    return alpha2

# appends symbol to the kb if prop not already in there
# and if not(prop) not already in there 
def addToKB(prop, kb):
    if prop not in kb:
        if ('n'+prop) not in kb:
            kb.append(prop)

# check model to see if there are breezes in rooms around pits, 
# and pits whenever breezes and ditto for wumpus/stenches
# and checks that there is max one wumpus in model
# returns false if any are false
def isTrueRules(model, position):
    isCorrect = True
    # check rules
    wumpusCount = 0
    for symbol, value in model:
        if isCorrect is False: # returns false after first false returned from isTrue
            return False
        if ('b' in symbol) and (value == True): # rooms where there is a breeze
            coords = (int(symbol[1]), int(symbol[2]))
            if coords == position:
                rooms = getSurroundingRooms(coords) # get rooms surrounding the breeze
                alpha = createAlpha(rooms, 'p', 'or')
                isCorrect = isTrue(alpha, model) # checks if there is at least one pit
        if ('p' in symbol) and (value == True): # rooms where there is a pit
            coords = (int(symbol[1]), int(symbol[2]))
            rooms = getSurroundingRooms(coords) # get rooms surrounding the pit
            alpha = createAlpha(rooms, 'b', 'and') # creates alpha for if there are breezes in all surrounding checked rooms
            isCorrect = isTrue(alpha, model) 
        if ('s' in symbol) and (value == True): # rooms where there is a stench
            coords = (int(symbol[1]), int(symbol[2]))
            if coords == position:
                rooms = getSurroundingRooms(coords) # get rooms surrounding the stench
                alpha = createAlpha(rooms, 'w', 'or') # creates alpha for if there is a wumpus in surrounding room
                isCorrect = isTrue(alpha, model) 
        if ('w' in symbol) and (value == True): # rooms where there is a wumpus
            wumpusCount += 1
            if wumpusCount > 1:
                return False # model is false is there is more than one wumpus in model
            coords = (int(symbol[1]), int(symbol[2]))
            rooms = getSurroundingRooms(coords) # get rooms surrounding the wumpus
            alpha = createAlpha(rooms, 's', 'and') # creates alpha for if there is a stench in all surrounding checked rooms
            isCorrect = isTrue(alpha, model)
    return isCorrect


