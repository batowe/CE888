# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai

from math import *
import random


statez = ""

class OXOState:
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
    """
    def __init__(self):
        self.playerJustMoved = 2 # At the root pretend the player just moved is p2 - p1 has the first move
        self.board = [0,0,0,0,0,0,0,0,0] # 0 = empty, 1 = player 1, 2 = player 2
        
    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OXOState()
        st.playerJustMoved = self.playerJustMoved
        st.board = self.board[:]
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        assert move >= 0 and move <= 8 and move == int(move) and self.board[move] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[move] = self.playerJustMoved
        
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return [i for i in range(9) if self.board[i] == 0]
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        for (x,y,z) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
            if self.board[x] == self.board[y] == self.board[z]:
                if self.board[x] == playerjm:
                    return 1.0
                else:
                    return 0.0
        if self.GetMoves() == []: return 0.5 # draw
        assert False # Should not be possible to get here

    def __repr__(self):
        s= ""
        for i in range(9):
            s += "012"[self.board[i]]
            #s += ".XO"[self.board[i]]
            #if i % 3 == 2: s += "\n"
        return s

class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves() # future child nodes
        self.playerJustMoved = state.playerJustMoved # the only part of the state that the Node needs later
        
    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + sqrt(2*log(self.visits)/c.visits))[-1]
        return s
    
    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move = m, parent = self, state = s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n
    
    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        bum =  "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"
        return ""
    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
             s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
             s += str(c) + "\n"
        return s


def UCT(rootstate, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves) 
            state.DoMove(m)
            node = node.AddChild(m,state) # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.GetMoves() != []: # while state is non-terminal
            state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose): print(rootnode.TreeToString(0))
    #else: print(rootnode.ChildrenToString())

    return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited


def UCTPlayGame():
    """ Play a sample game between two UCT players where each player gets a different number 
        of UCT iterations (= simulations = tree nodes).
    """
    # Create a game data .csv file
    f = open("gameData.csv", "w+")
    f.write("p1,p2,p3,p4,p5,p6,p7,p8,p9,win\n")
    global statez
    play = False
    numGames = 500
    print("Running simulation...")
    # Play a set number of games
    for k in range(0, numGames):
        win = False
        #print("THIS IS STATES BEFORE: " + statez)
        statez = ""
        #print("Game=" + str(k) + "\n")
        state = OXOState()
        # Play the game while there are moves or no player has won
        while (state.GetMoves() != []):
            # Print board state on each move
            if play:
                print(str(state.board[0:3]))
                print(str(state.board[3:6]))
                print(str(state.board[6:9]))
                # Let us choose a move
                tmp = 0
                moves = []
                for i in state.GetMoves():
                    moves.append(i+1) # Normalise numbers to 1-9 range for better comprehension
                print("Available moves: " + str(moves))
                while((tmp-1) not in state.GetMoves()):
                    tmp = int(input("Please make a move: "))
                
                state.DoMove(int(tmp)-1)
                print("You moved")
            if state.playerJustMoved == 1:
                m = UCT(rootstate = state, itermax = 1, verbose = False) # play with values for itermax and verbose = True
            else:
                m = UCT(rootstate = state, itermax = 1, verbose = False)
            #print(str(state.GetMoves())) # List valid moves at each step
            #print "Best Move: " + str(m) + "\n"
            state.DoMove(m)
            #print("Computer moves")
        
        if state.GetResult(state.playerJustMoved) == 1.0:
            print("Player " + str(state.playerJustMoved) + " wins!")
            win = True
        elif state.GetResult(state.playerJustMoved) == 0.0:
            print("Player " + str(3 - state.playerJustMoved) + " wins!")
        else:
            print("Nobody wins!")
        print(str(state.board[0:3]))
        print(str(state.board[3:6]))
        print(str(state.board[6:9]))
        tmpPrint = ""
        for i in state.board:
            tmpPrint = tmpPrint + str(i) + ","
        tmpPrint = tmpPrint + str(win) + "\n"
        print(tmpPrint)
        f.write(tmpPrint)
    f.close()
    print("Done!")
if __name__ == "__main__":
    """ Play a single game to the end using UCT for both players. 
    """
    UCTPlayGame()

            
                          
            

