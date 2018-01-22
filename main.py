# main.py


import timeController
import scoreController
import gridController
import tileController

import grapher

import ai
import time
import viewController
import NeuralNetwork
import ai_nn

timeStart = time.time()

timeController = timeController.TimeController( 1 )
scoreController = scoreController.ScoreController( )
gridController = gridController.GridController( scoreController )
tileController = tileController.TileController( gridController )

grapher = grapher.Grapher( scoreController )

#ai = ai.AI( gridController, scoreController, grapher, {}, 20)

nn = NeuralNetwork.NeuralNetwork()
model = nn.create_model()
ai = ai_nn.AI_NN(gridController, scoreController, grapher, model)

viewController = viewController.ViewController( gridController, timeController, scoreController, ai, grapher )

index = 0

# =====================================================================
cTile = tileController.getTile( index )
index += 1
nTile = tileController.getTile( index )
# =====================================================================


# cTile = tileController.getRandomTile( )
# nTile = tileController.getRandomTile( )
viewController.setTile( cTile, nTile )

"""
while not viewController.abort:
    if timeController.timeEvent( ):
        if viewController.aiState:
            #move, rotate, rating =  ai.makeMove( cTile )
            ai.train(cTile)
        if not cTile.incY( ):
            cTile.apply( )

            if not gridController.checkForGameOver( ):
                scoreController.tileReleased( )
                cTile = nTile
                index += 1
                nTile = tileController.getTile( index )
                viewController.setTile( cTile, nTile )
            else:
                index = 0
                cTile = tileController.getTile( index )
                index += 1
                nTile = tileController.getTile( index )
                viewController.setTile( cTile, nTile )
    viewController.updateEverything( )
"""

while True:
    times = 0
    index = 0
    scoreController.reset()
    gridController.reset()
    # input()
    for i in range(300):
        if timeController.timeEvent( ):
            if viewController.aiState:
                #move, rotate, rating =  ai.makeMove( cTile )
                ai.play(cTile)
            if not cTile.incY( ):
                cTile.apply( )

                if not gridController.checkForGameOver( ):
                    scoreController.tileReleased( )
                    cTile = nTile
                    index += 1
                    nTile = tileController.getTile( index )
                    viewController.setTile( cTile, nTile )
                else:
                    index = 0
                    times += 1
                    cTile = tileController.getTile( index )
                    index += 1
                    nTile = tileController.getTile( index )
                    viewController.setTile( cTile, nTile )
        viewController.updateEverything( )
    time_cost = time.time()
    print('Evaluation : %f, gameover %d times, use %d old exp, cost %f sec' % (scoreController.score, times, ai.old, time_cost-timeStart))
    #ai.printQTable()
timeEnd = time.time()
print("It cost %f sec" % (timeEnd - timeStart))