'''
Modified by Amanda Martin for Q-Learning Implementation
runs with wwagent_v3.py
'''
# Includes classes for the simulation and the Tkinter display of the simulation.
# Also includes code to process the command-line input and run the program accordingly.


import sys

from wwagent_v3 import *

from tkinter import *
from random import randint

# in your inner loop use it thus (just an example, I would probably use a named tuple)
#
#wwagent.update(percept) # update the agent with the current percept
#action = wwagent.action() # get the next action to take from the agent

# Global Constants
COLUMNS = 4
ROWS = 4
FONTTYPE = "Purisa"

# SET UP CLASS AND METHODS HERE
# Simulation class for running the underlying factors of the simulation
class Simulation:

    def __init__(self, rowSize, colSize, score):
        self.rowSize = rowSize
        self.colSize = colSize
        self.agent = WWAgent()
        self.score = score
        self.lastMove = 'None'
        self.lastPos = (3, 0)
        self.agentPos = (3, 0)
        self.agentFacing = 'right'
        self.arrow = 1
        self.wumpusAlive = True
        self.pits = {}
        self.percepts = {}
        for r in range(self.rowSize):
            for c in range(self.colSize):
                self.percepts['room'+str(r)+str(c)] = (None, None, None, None, None)
        self.wumpusLoc = (None, None)
        self.goldLocation = (None, None)
        self.hasGold = False
        self.endEpisode = False # self termination

    def set_percepts(self, r, c, item):
        if (item == 'gold'):
            p = self.percepts['room'+str(r)+str(c)]
            self.percepts['room'+str(r)+str(c)] = (p[0], p[1], 'glitter', p[3], p[4])
        if (item == 'wumpus'):
            p = self.percepts['room'+str(r)+str(c)]
            #self.percepts['room'+str(r)+str(c)] = ('stench', p[1], p[2], p[3], p[4])
            if ((r - 1) >= 0):
                p = self.percepts['room'+str(r-1)+str(c)]
                self.percepts['room'+str(r-1)+str(c)] = ('stench', p[1], p[2], p[3], p[4])
            if ((r + 1) < 4):
                p = self.percepts['room'+str(r+1)+str(c)]
                self.percepts['room'+str(r+1)+str(c)] = ('stench', p[1], p[2], p[3], p[4])
            if ((c - 1) >= 0):
                p = self.percepts['room'+str(r)+str(c-1)]
                self.percepts['room'+str(r)+str(c-1)] = ('stench', p[1], p[2], p[3], p[4])
            if ((c + 1) < 4):
                p = self.percepts['room'+str(r)+str(c+1)]
                self.percepts['room'+str(r)+str(c+1)] = ('stench', p[1], p[2], p[3], p[4])
        if (item == 'pit'):
            if ((r - 1) >= 0):
                p = self.percepts['room'+str(r-1)+str(c)]
                self.percepts['room'+str(r-1)+str(c)] = (p[0], 'breeze', p[2], p[3], p[4])
            if ((r + 1) < 4):
                p = self.percepts['room'+str(r+1)+str(c)]
                self.percepts['room'+str(r+1)+str(c)] = (p[0], 'breeze', p[2], p[3], p[4])
            if ((c - 1) >= 0):
                p = self.percepts['room'+str(r)+str(c-1)]
                self.percepts['room'+str(r)+str(c-1)] = (p[0], 'breeze', p[2], p[3], p[4])
            if ((c + 1) < 4):
                p = self.percepts['room'+str(r)+str(c+1)]
                self.percepts['room'+str(r)+str(c+1)] = (p[0], 'breeze', p[2], p[3], p[4])

    def generate_simulation(self):
        # Set wumpus location
        self.wumpusLoc = (randint(0, 3), randint(0, 3))
        while (self.wumpusLoc == (3, 0)):
            self.wumpusLoc = (randint(0, 3), randint(0, 3))
        # Set wumpus percepts
        self.set_percepts(self.wumpusLoc[0], self.wumpusLoc[1], 'wumpus')
        # Set gold location
        self.goldLocation = (randint(0, 3), randint(0, 3))
        while (self.goldLocation == (3, 0)) or (self.goldLocation == self.wumpusLoc):
            self.goldLocation = (randint(0, 3), randint(0, 3))
        # Set gold percepts
        self.set_percepts(self.goldLocation[0], self.goldLocation[1], 'gold')
        # Generate pits
        for r in range(self.rowSize):
            for c in range(self.colSize):
                if (randint(1, 5) == 3) and ((r != 3) or (c != 0)):
                    self.pits['room'+str(r)+str(c)] = True
                    # Set pit percepts
                    self.set_percepts(r, c, 'pit')
                else:
                    self.pits['room'+str(r)+str(c)] = False

    def reset_stats(self, newScore):
        self.agent = None
        self.score = newScore
        self.lastMove = 'None'
        self.lastPos = (3, 0)
        self.agentPos = (3, 0)
        self.agentFacing = 'right'
        self.arrow = 1
        self.hasGold = False
        self.wumpusAlive = True
        self.percepts = {}
        self.agent = WWAgent()
        self.endEpisode=False
        for r in range(self.rowSize):
            for c in range(self.colSize):
                self.percepts['room'+str(r)+str(c)] = (None, None, None, None, None)

    def agent_move(self, action):

        
        if (action == 'exit'):
            self.endEpisode=True
            return
        
        if (action == 'shoot'):
            self.score = self.score - 10
        else:
            self.score = self.score - 1
        r = self.agentPos[0]
        c = self.agentPos[1]
        if (action == 'move'):
            self.lastPos = self.agentPos
            bump = False
            if (self.agentFacing == 'right'):
                if ((c + 1) < 4):
                    self.agentPos = (self.agentPos[0], self.agentPos[1] + 1)
                else:
                    bump = True
            elif (self.agentFacing == 'up'):
                if ((r - 1) >= 0):
                    self.agentPos = (self.agentPos[0] - 1, self.agentPos[1])
                else:
                    bump = True
            elif (self.agentFacing == 'left'):
                if ((c - 1) >= 0):
                    self.agentPos = (self.agentPos[0], self.agentPos[1] - 1)
                else:
                    bump = True
            else:
                if ((r + 1) < 4):
                    self.agentPos = (self.agentPos[0] + 1, self.agentPos[1])
                else:
                    bump = True
            if (bump):
                p = self.percepts['room'+str(r)+str(c)]
                self.percepts['room'+str(r)+str(c)] = (p[0], p[1], p[2], 'bump', p[4])
            p = self.percepts['room'+str(r)+str(c)]
            self.percepts['room'+str(r)+str(c)] = (p[0], p[1], p[2], p[3], None)
            self.lastMove = 'Move Forward'
        elif (action == 'grab'):
            if (self.agentPos == self.goldLocation):
                self.hasGold = True
            self.lastMove = "Grab"
        elif (action == 'climb'):
            self.lastMove = 'Climb'
        elif (action == 'shoot'):
            if (self.arrow != 0):
                if (self.agentFacing == 'up'):
                    if (c == self.wumpusLoc[1]) and (r > self.wumpusLoc[0]):
                        self.wumpusAlive = False
                elif (self.agentFacing == 'right'):
                    if (r == self.wumpusLoc[0]) and (c < self.wumpusLoc[1]):
                        self.wumpusAlive = False
                elif (self.agentFacing == 'left'):
                    if (r == self.wumpusLoc[0]) and (c > self.wumpusLoc[1]):
                        self.wumpusAlive = False
                else:
                    if (c == self.wumpusLoc[1]) and (r < self.wumpusLoc[0]):
                        self.wumpusAlive = False
                self.arrow = 0
            if (self.wumpusAlive == False):
                p = self.percepts['room'+str(r)+str(c)]
                self.percepts['room'+str(r)+str(c)] = (p[0], p[1], p[2], None, 'scream')
            self.lastMove = 'Shoot'
        else:
            if (action == 'left'):
                if (self.agentFacing == 'right'):
                    self.agentFacing = 'up'
                elif (self.agentFacing == 'up'):
                    self.agentFacing = 'left'
                elif (self.agentFacing == 'left'):
                    self.agentFacing = 'down'
                else:
                    self.agentFacing = 'right'
                self.lastMove = 'Rotate Left'
            else:
                if (self.agentFacing == 'right'):
                    self.agentFacing = 'down'
                elif (self.agentFacing == 'down'):
                    self.agentFacing = 'left'
                elif (self.agentFacing == 'left'):
                    self.agentFacing = 'up'
                else:
                    self.agentFacing = 'right'
                self.lastMove = 'Rotate Right'
            p = self.percepts['room'+str(r)+str(c)]
            self.percepts['room'+str(r)+str(c)] = (p[0], p[1], p[2], None, None)
        #print 'S-position: ', self.agentPos

    def terminal_test(self):

        r = self.agentPos[0]
        c = self.agentPos[1]
        if (self.agentPos == self.wumpusLoc) and (self.wumpusAlive == True):
            resetAgent()
            return True
        elif (self.pits['room'+str(r)+str(c)]):
            resetAgent()
            return True
        elif (self.agentPos == (3, 0)) and self.lastMove.lower() == 'climb':
            return True
        else:
            return False

    def update_score(self):
        r = self.agentPos[0]
        c = self.agentPos[1]
        if (self.agentPos == self.wumpusLoc) and (self.wumpusAlive == True):
            self.score = self.score - 1000
        elif (self.pits['room'+str(r)+str(c)]):
            self.score = self.score - 1000
        elif (self.agentPos == (3, 0)) and self.lastMove.lower() == 'climb':
            if (self.hasGold):
                self.score = self.score + 1000

    def move(self):
        p = self.agentPos
        self.agent.update(self.percepts['room'+str(p[0])+str(p[1])])
        action = self.agent.action()
        #print "Sim action: ", action
        self.agent_move(action)



# Display class for running and modifying the GUI
class Display:

    score = None
    pastMove = None
    arrowStatus = None
    arrowStatusDis = None
    percepts = None
    agentDirection = None

    def set_room(self, r, c, sim):
        # Returns agent image
        if (sim.agentPos[0] == r) and (sim.agentPos[1] == c):
            if (sim.agentFacing.lower() == 'right'):
                return PhotoImage(file="Images/agent-right.gif")
            elif (sim.agentFacing.lower() == 'up'):
                return PhotoImage(file="Images/agent-up.gif")
            elif (sim.agentFacing.lower() == 'left'):
                return PhotoImage(file="Images/agent-left.gif")
            else:
                return PhotoImage(file="Images/agent-down.gif")
        # Returns start image
        elif (r == 3) and (c == 0):
            return PhotoImage(file="Images/start.gif")
        # Returns wumpus
        elif (r == sim.wumpusLoc[0]) and (c == sim.wumpusLoc[1]):
            if (sim.pits['room'+str(r)+str(c)]):
                return PhotoImage(file="Images/pit-wumpus.gif")
            else:
                return PhotoImage(file="Images/live-wumpus.gif")
        # Returns gold and pit or gold
        elif (r == sim.goldLocation[0]) and (c == sim.goldLocation[1]):
            if (sim.pits['room'+str(r)+str(c)]):
                return PhotoImage(file="Images/gold-pit.gif")
            else:
                return PhotoImage(file="Images/gold.gif")
        # Returns a pit
        elif (sim.pits['room'+str(r)+str(c)]):
            return PhotoImage(file="Images/pit.gif")
        # Returns an empty room
        else:
            return PhotoImage(file="Images/emptyroom.gif")

    def __init__(self, master, simulation):
        frame = Frame(master, width = 700, height = 500)
        frame.pack()
        self.grid = {}
        self.score = StringVar()
        self.pastMove = StringVar()
        self.arrowStatus = StringVar()
        self.percepts = StringVar()
        self.agentDirection = StringVar()
        self.score.set(str(0))
        self.pastMove.set('None')
        self.arrowStatus.set('Available')
        self.agentDirection.set('Right')
        self.percepts.set(str(simulation.percepts['room30']))
        theScoreDis = Label(master, font=(FONTTYPE, 16), text="Performance:")
        lastMoveDis = Label(master, font=(FONTTYPE, 16), text="Last Move:")
        performanceDis = Label(master, font=(FONTTYPE, 14), textvariable=self.score)
        pastMoveDis = Label(master, font=(FONTTYPE, 14), textvariable=self.pastMove)
        arrowTitle = Label(master, font=(FONTTYPE, 16), text="Arrow Status:")
        self.arrowStatusDis = Label(master, font=(FONTTYPE, 14), fg='Green', textvariable=self.arrowStatus)
        perceptsTitle = Label(master, font=(FONTTYPE, 16), text="Current Percepts:")
        perceptsDis = Label(master, font=(FONTTYPE, 14), textvariable=self.percepts)
        agentDirectionTitle = Label(master, font=(FONTTYPE, 16), text = "Agent Facing:")
        agentDirectionDis = Label(master, font=(FONTTYPE, 14), textvariable=self.agentDirection)
        self.goldStatus = Label(master, font=(FONTTYPE, 16), fg='Gold', text = "Agent has gold!")
        performanceDis.place(x = 420, y = 25)
        theScoreDis.place(x = 420, y = 0)
        arrowTitle.place(x = 420, y = 75)
        self.arrowStatusDis.place(x = 420, y = 100)
        lastMoveDis.place(x = 420, y = 150)
        pastMoveDis.place(x = 420, y = 175)
        perceptsTitle.place(x = 5, y = 420)
        perceptsDis.place(x = 5, y = 445)
        agentDirectionTitle.place(x = 420, y = 285)
        agentDirectionDis.place(x = 420, y = 312)

        #creating the initial grid
        for r in range(ROWS):
            for c in range(COLUMNS):
                tkimage = self.set_room(r, c, simulation)
                self.grid['room'+str(r)+str(c)] = Label(master, image = tkimage)
                self.grid['room'+str(r)+str(c)].image = tkimage
                self.grid['room'+str(r)+str(c)].place(x = c*100 + c*2, y = r*100 + r*2)

        #initializations
    def update_move(self, sim):
        self.score.set(str(sim.score))
        self.pastMove.set(sim.lastMove)
        self.agentDirection.set(sim.agentFacing.title())
        if (sim.arrow == 0):
            self.arrowStatus.set('Used')
            self.arrowStatusDis.config(fg='Red')
        if (sim.lastPos != sim.agentPos):
            r = sim.lastPos[0]
            c = sim.lastPos[1]
            tempImg = self.set_room(r, c, sim)
            self.grid['room'+str(r)+str(c)].config(image = tempImg)
            self.grid['room'+str(r)+str(c)].image = tempImg
        r = sim.agentPos[0]
        c = sim.agentPos[1]
        tempImg = self.set_room(r, c, sim) 
        self.grid['room'+str(r)+str(c)].config(image = tempImg)
        self.grid['room'+str(r)+str(c)].image = tempImg
        currentPercepts = sim.percepts['room'+str(sim.agentPos[0])+str(sim.agentPos[1])]
        self.percepts.set(str(currentPercepts))
        if (sim.hasGold):
            self.goldStatus.place(x = 500, y = 225)
        if (sim.arrow == 0):
            self.arrowStatus.set('Used')
        if (sim.wumpusAlive == False):
            loc = sim.wumpusLoc
            if (sim.agentPos != sim.wumpusLoc):
                temp = PhotoImage(file="Images/dead-wumpus.gif")
            else:
                temp = self.set_room(loc[0], loc[1], sim)
            self.grid['room'+str(loc[0])+str(loc[1])].config(image = temp)
            self.grid['room'+str(loc[0])+str(loc[1])].image = temp

    def reset_display(self, sim):
        for r in range(ROWS):
            for c in range(COLUMNS):
                tkimage = self.set_room(r, c, sim)
                self.grid['room'+str(r)+str(c)].config(image = tkimage)
                self.grid['room'+str(r)+str(c)].image = tkimage
        self.score.set(str(sim.score))
        self.pastMove.set(sim.lastMove)
        self.agentDirection.set(sim.agentFacing.title())
        self.arrowStatus.set('Available')
        self.arrowStatusDis.config(fg='Green')
        currentPercepts = sim.percepts['room'+str(sim.agentPos[0])+str(sim.agentPos[1])]
        self.percepts.set(str(currentPercepts))
        self.goldStatus.place_forget()
        
def resetAgent():
    sim.reset_stats(0)
    app.reset_display(sim)
    eaten.place_forget()
    fell.place_forget()
    climbOut.place_forget()
    makeMove.place(x = 420, y = 225)

# Interpret command-line call with arguments

arglist = sys.argv
if (len(sys.argv) == 2):
    if (arglist[1].lower() == '-gui'):
        print('Running GUI...')
        # RUN SIMULATION WITH GUI DISPLAY
        root = Tk()
        root.wm_title("Wumpus World Simulation")
        sim = Simulation(ROWS, COLUMNS, 0)
        sim.generate_simulation()
        app = Display(root, sim)

        # Updates the sim with each move
        def resetGame():
            sim.reset_stats(0)
            sim.generate_simulation()
            app.reset_display(sim)
            eaten.place_forget()
            fell.place_forget()
            climbOut.place_forget()
            makeMove.place(x = 420, y = 225)

        def updateSim():
            if (sim.endEpisode):
                resetGame()
                return
            sim.move()
            sim.update_score()
            if (sim.terminal_test() and sim.lastMove.lower() == 'climb'):
                climbOut.place(x = 420, y = 400)
                makeMove.place_forget()
            elif (sim.terminal_test()):
                if (sim.agentPos == sim.wumpusLoc) and (sim.wumpusAlive is True):
                    eaten.place(x = 420, y = 400)
                else:
                    fell.place(x = 420, y = 400)
                makeMove.place_forget()
            app.update_move(sim)

        # Methods for the buttons to operate the agent manually
        def movePlayer():
            sim.agent_move('move')
            sim.update_score()
            if (sim.terminal_test()):
                if (sim.agentPos == sim.wumpusLoc) and (sim.wumpusAlive is True):
                    eaten.place(x = 420, y = 400)
                else:
                    fell.place(x = 420, y = 400)
                makeMove.place_forget()
            app.update_move(sim)
        def moveLeft():
            sim.agent_move('left')
            sim.update_score()
            if (sim.terminal_test()):
                if (sim.agentPos == sim.wumpusLoc) and (sim.wumpusAlive is True):
                    eaten.place(x = 420, y = 400)
                else:
                    fell.place(x = 420, y = 400)
                makeMove.place_forget()
            app.update_move(sim)
        def moveRight():
            sim.agent_move('right')
            sim.update_score()
            if (sim.terminal_test()):
                if (sim.agentPos == sim.wumpusLoc) and (sim.wumpusAlive is True):
                    eaten.place(x = 420, y = 400)
                else:
                    fell.place(x = 420, y = 400)
                makeMove.place_forget()
            app.update_move(sim)
        def grab():
            sim.agent_move('grab')
            sim.update_score()
            if (sim.terminal_test()):
                if (sim.agentPos == sim.wumpusLoc) and (sim.wumpusAlive is True):
                    eaten.place(x = 420, y = 400)
                else:
                    fell.place(x = 420, y = 400)
                makeMove.place_forget()
            app.update_move(sim)
        def climb():
            sim.agent_move('climb')
            sim.update_score()
            if (sim.terminal_test()):
                if (sim.agentPos == sim.wumpusLoc) and (sim.wumpusAlive is True):
                    eaten.place(x = 420, y = 400)
                else:
                    fell.place(x = 420, y = 400)
                makeMove.place_forget()
            app.update_move(sim)
        def shoot():
            sim.agent_move('shoot')
            sim.update_score()
            if (sim.terminal_test()):
                if (sim.agentPos == sim.wumpusLoc) and (sim.wumpusAlive is True):
                    eaten.place(x = 420, y = 400)
                else:
                    fell.place(x = 420, y = 400)
                makeMove.place_forget()
            app.update_move(sim)
        # The move button
        makeMove = Button(root, text = "Move", font = (FONTTYPE, 14), command = updateSim)
        makeMove.place(x = 420, y = 225)

#       BELOW ARE BUTTONS FOR MANUALLY CONTROLLING THE AGENT
#       They can be used for testing the simulation runs properly
#       Uncomment the following lines to use them
#
        go = Button(root, text = "Go", font = (FONTTYPE, 14), command = movePlayer)
        go.place(x = 470, y = 350)
        left = Button(root, text = "Left", font = (FONTTYPE, 14), command = moveLeft)
        left.place(x = 420, y = 350)
        right = Button(root, text = "Right", font = (FONTTYPE, 14), command = moveRight)
        right.place(x = 515, y = 350)
        toGrab = Button(root, text = "Grab", font = (FONTTYPE, 14), command = grab)
        toGrab.place(x = 500, y = 435)
        toClimb = Button(root, text = "Climb", font = (FONTTYPE, 14), command = climb)
        toClimb.place(x = 570, y = 435)
        toShoot = Button(root, text = "Shoot", font = (FONTTYPE, 14), command = shoot)
        toShoot.place(x = 420, y = 390)

        reset = Button(root, text = "Reset", font = (FONTTYPE, 14), command = resetGame)
        eaten = Label(root, text = "WUMPUS ATE AGENT", fg = 'Red', font = (FONTTYPE, 16))
        climbOut = Label(root, text = "Player climbed out", fg = 'Green', font = (FONTTYPE, 18))
        fell = Label(root, text = "AGENT FELL IN PIT", fg = 'Red', font = (FONTTYPE, 16))
        


        reset.place(x = 420, y = 435)

        # Main simulation loop
        root.mainloop()
        #
    elif (arglist[1].lower() == '-nongui'):
        print('Running Non-GUI...')
        print('\n')
        # RUN SIMULATION WHILE WRITING TO standard output
        sim = Simulation(ROWS, COLUMNS, 0)
        sim.generate_simulation()
        wl = sim.wumpusLoc
        gl = sim.goldLocation
        pl = []
        for i in range(4):
            for j in range(4):
                if sim.pits['room'+str(i)+str(j)] is True:
                    pl.append((i, j))
        moveCount = 0

        # Print the steps
        print('START OF SIMULATION')
        while (sim.terminal_test() is not True) and (sim.endEpisode is not True ):
            print('------------------------------------------------------------------')
            print ('Move: ', moveCount)
            print ('Last Action: ', sim.lastMove)
            print('\n')
            print('Wumpus World Item Locations:')
            print ('Wumpus Location: ', wl, '   Gold Location: ', gl)
            print ('Pit Locations: ', str(pl))
            print('\n')
            print('Agent Info:')
            print ('Position: ', sim.agentPos, '   Facing: ', sim.agentFacing)
            print ('Has Gold: ', str(sim.hasGold), '   Arrow: ', sim.arrow)
            print('\n')
            print('Simlulation Current States:')
            print ('Wumpus Alive: ', str(sim.wumpusAlive), '   Performance: ', sim.score)
            print ('Current Percepts: ', str(sim.percepts['room'+str(sim.agentPos[0])+str(sim.agentPos[1])]))
            # Prompt agent to move
            sim.move()
            sim.update_score()
            moveCount = moveCount + 1
        # Print final result
        print('------------------------------------------------------------------')
        print ('Last Action: ', sim.lastMove)
        print('GAME OVER')
        print('\n')
        if (sim.endEpisode):
            print("Agent acquired the gold.")
        elif sim.lastMove.lower() == 'climb':
            print('Agent has climbed out of cave.')
        elif sim.agentPos == sim.wumpusLoc:
            print('Agent was eaten by the wumpus and died!')
        else:
            print('Agent fell into pit and died!')
        print('\n')
        print ('Final Performance: ', sim.score)

    elif (arglist[1].lower() == '-help'):
        print('------------------------------------------------------------------')
        print('This python program runs a simulation of Wumpus World.')
        print('\n')
        print('To run the GUI represented version, run the following command:')
        print('>\tpython wwsim.py -gui')
        print('\n')
        print('To run the Non-GUI version, run the following command:')
        print('>\tpython wwsim.py -nongui')
        print('------------------------------------------------------------------')
    else:
        raise Exception('Invalid command-line argument. Run \'python wwsim.py -help\' for help.');
else:
    raise Exception('Invalid command-line call. Run \'python wwsim.py -help\' for help.');
