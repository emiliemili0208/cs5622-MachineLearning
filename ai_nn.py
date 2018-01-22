import population
import numpy as np
import random

WIDTH = 6
HEIGHT = 20

class AI_NN( object ):

    def __init__( self, grid, score, grapher, model):
        self.grid = grid
        self.score = score
        self.population = population.Population( )
        self.currentGeneration = 0
        self.currentGenome = 0
        self.old = 0
        self.grapher = grapher
        self.backupGrid = np.zeros( [ WIDTH, HEIGHT ], dtype=np.uint8 ) #change width #change height
        self.backupTile = [ 0, 0, 0 ]
        # ===================================================================== 
        
        # Q learning state/action
        self.state = None
        self.count = 0
        self.actionTable = []
        for move in range( -int(WIDTH/2), int(WIDTH/2) ):
            for rotate in range( 0, 4 ):
                self.actionTable.append((move,rotate))
        # NN model
        self.model = model
        self.expBuffer = 10000
        self.exp = []
        self.bufInd = 0
        self.batchSize = 100
        # Param
        self.gamma = 0.8
        self.alpha = 0.2
        self.epsilon = 0
        # =====================================================================

    # =====================================================================

    def play(self, tile):
        bestMove, bestRotate = self.chooseBestAction(tile)
        curState, action, reward, nextState = self.update(tile, bestMove, bestRotate)
        #print("chose action ", action)
        self.train(curState, action, reward, nextState)
    
    def chooseBestAction(self, tile):
        self.state = self.calculateState()
        curStateWithTile = self.state + [tile.identifier]
        value1 = self.fitness()
        #print(curStateWithTile)
        Q = self.model.predict(np.asarray(curStateWithTile).reshape(1,6), batch_size = 1)
        
        if random.random() < self.epsilon:
            action = np.random.randint(0,24)
        else:
            action = np.argmax(Q)

        return self.actionTable[action][0], self.actionTable[action][1]

    def update(self, tile, bestMove, bestRotate):
        self.grid.realAction = True
        
        # Get current state and fitness func
        value1 = self.fitness()
        self.state = self.calculateState()
        curStateWithTile = self.state + [tile.identifier]
       
        # make move and update grid
        for i in range( 0, bestRotate ):
            tile.rotCW( )
        if bestMove<0:
            for i in range( 0, -bestMove ):
                tile.decX( )
        if bestMove>0:
            for i in range( 0, bestMove ):
                tile.incX( )
        tile.drop( )
        tile.apply( )
        self.grid.removeCompleteRows( )

        #change epsilon
        self.changeEpsilon(bestMove)

        # Get New State
        value2 = self.fitness()
        nextState = self.calculateState()
        nextStateWithTile = nextState + [tile.identifier]
     
        # Calculate reward
        reward = self.getReward(value1, value2)
        # print(reward)
        for index, a in enumerate(self.actionTable):
            if bestMove == a[0] and bestRotate == a[1]:
                action = index
        return curStateWithTile, action, reward, nextStateWithTile

    def train(self, curState, action, reward, nextState):
        if (len(self.exp) < self.expBuffer): #if buffer not filled, add to it
            self.exp.append((curState, action, reward, nextState))
        else: #if buffer full, overwrite old values
            if self.bufInd < (self.expBuffer-1):
                self.bufInd += 1
            else:
                self.bufInd = 0
            
            self.exp[self.bufInd] = (curState, action, reward, nextState)
            #randomly sample our experience memory
            minibatch = random.sample(self.exp, self.batchSize)
            X_train = []
            y_train = []
            for mem in minibatch:
                #Get max_Q(S',a)
                old_state, action, mem_reward, next_state = mem
                oldQ = self.model.predict(np.asarray(old_state).reshape(1,6), batch_size=1)
                newQ = self.model.predict(np.asarray(next_state).reshape(1,6), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,24))
                y[:] = oldQ[:]
                update = (mem_reward + (self.gamma * maxQ))
                
                y[0][action] = update
                X_train.append(np.asarray(old_state).reshape(6,))
                y_train.append(y.reshape(24,))

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            self.model.fit(X_train, y_train, batch_size=self.batchSize, nb_epoch=1, verbose=0)
            #print("train!!!!")

    def getReward(self, value1, value2):
        return value2-value1

    def fitness(self):
        # value = self.grid.lastRowsCleared * 0.76
        # value += self.grid.lastSumHeight / 10 * -0.51
        # value += self.grid.lastAmountHoles * -0.36
        # value += self.grid.lastRoughness * -0.18
        value = 10
        value += self.grid.lastRowsCleared * 0.76
        value += self.grid.lastMaxHeight * -0.51
        #value += self.grid.lastRelativeHeight * -0.5
        value += self.grid.lastAmountHoles * -0.36
        value += self.grid.lastRoughness * -0.18
        return value

    def calculateState(self):
        state = []
        for item in self.grid.grid:
            # print(item)
            # input()
            state_count = 0
            while state_count < HEIGHT and item[state_count] == 0:
                state_count += 1
                pass
            state += [HEIGHT - state_count]
        
        state_diff=[]
        for i in range(len(state)-1):
            state_diff.append(state[i+1] - state[i])
            # if state[i+1] - state[i] > 0:
            #     state_diff.append(1)
            # elif state[i+1] - state[i] < 0:
            #     state_diff.append(-1)
            # else:
            #     state_diff.append(0)

        return state_diff

    def changeEpsilon(self, move):
        if move < 0 and self.epsilon > 0:
            if self.epsilon < 0.1:
                self.epsilon = 0
            else:
                self.epsilon -= 0.1
        elif move > 0 and self.epsilon < 1:
            if self.epsilon > 0.9:
                self.epsilon = 1
            else:
                self.epsilon += 0.1