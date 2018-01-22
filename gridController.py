# gridController.py


import numpy as np
import math

WIDTH = 6
HEIGHT = 20

class GridController( object ):

    def __init__( self, score ):
        self.width = WIDTH #change width
        self.height = HEIGHT #change height
        self.score = score
        self.grid = np.zeros( [ self.width, self.height ], dtype=np.uint8 ) #change height
        self.realAction = True
        self.lastRowsCleared = 0
        self.lastMaxHeight = 0
        self.lastSumHeight = 0
        self.lastRelativeHeight = 0
        self.lastRoughness = 0
        self.lastAmountHoles = 0


    def checkField( self, psX, psY ):
        if psX < 0 or psX > self.width-1 or psY > self.height-1 or psY < 0: #change height
            return False
        if self.grid[ psX, psY ] != 0:
            return False
        return True

    def apply( self, posX, posY, identifier ):
        self.grid[ posX, posY ] = identifier

    def removeCompleteRows( self ):
        rows = 0
        for y in range( HEIGHT-1, -1, -1 ): #change height
            while np.amin( self.grid.T[ y ] ) != 0:
                rows += 1
                for y2 in range( y, 0, -1 ):
                    for x in range( self.width ):
                        self.grid[ x, y2 ] = self.grid[ x, y2-1 ]
        self.lastRowsCleared = rows
        heightData = [ ]
        for column in self.grid:
            counter = HEIGHT
            for i in range(HEIGHT-1, -1, -1):
                if column[i] != 0:
                    counter = i
            heightData.append( HEIGHT-1-counter )
        self.lastMaxHeight = np.amax( heightData )
        self.lastSumHeight = np.sum( heightData )
        self.lastRelativeHeight = self.lastMaxHeight - np.amin( heightData )
        self.lastRoughness = 0
        for x in range( self.width-1 ):
            self.lastRoughness += abs( heightData[ x ] - heightData[ x-1 ] )
        self.lastAmountHoles = 0
        for x in range( self.width ):
            for y in range( HEIGHT-1, 1, -1 ): #change height
                if self.grid[ x, y ] == 0 and self.grid[ x, y-1 ] != 0:
                    self.lastAmountHoles += 1
        if self.realAction:
            self.score.rowsCleared( rows )

    def checkForGameOver( self ):
        for y in range( 4 ):
            if np.amax( self.grid.T[ y ] ) != 0:
                if self.realAction:
                    self.reset( )
                return True
        return False

    def reset( self ):
        self.grid = np.zeros( [ self.width, self.height ], dtype=np.uint8 )#change width #change height
        if self.realAction:
            # self.score.reset( )
            self.lastRowsCleared = 0
            self.lastMaxHeight = 0
            self.lastSumHeight = 0
            self.lastRelativeHeight = 0
            self.lastRoughness = 0
            self.lastAmountHoles = 0
