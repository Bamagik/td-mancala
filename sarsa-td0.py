import numpy as np
import copy
import mancala

ALPHA = 0.1
EPSILON = 0.1
GAMMA = 0.1

class MancalaAI():
    def __init__(self, player, W=np.random.random((14)), epsilon=EPSILON):
        self.player = player
        self.W = W
        self.epsilon = epsilon

    def choose_action(self, board):
        best_action = None
        best_av = np.inf
        for i in range(6):
            if board.pits[i + (self.player-1)*6] != 0:
                new_board = copy.deepcopy(board)
                
                new_board.sow_seeds(i + (self.player-1)*6, self.player)

                av = np.dot(self.W, new_board.vectorize(self.player))

                if av < best_av:
                    best_action = i + (self.player-1)*6
                    best_av = av
            
        return best_action  

    def move(self, board):
        if np.random.random() >= self.epsilon:
            action = self.choose_action(board)
        else:
            action = np.random.randint(6)
            while board.pits[action + (1-self.player)*6] == 0:
                action = np.random.randint(6)
        return action


def sarsa():
    """
    v^ here is defined as X*W
    """
    policy = None
    w = np.random.random((14,))
    
    for i in range(1000):
        board = mancala.Board()

        player1 = MancalaAI(1, w)
        player2 = MancalaAI(2, w)
        players = [player1, player2]

        ai_player = 1
        current_player = 1
        while not board.is_end():
            if ai_player == current_player:
                # save current state
                state = board.vectorize(current_player)

                # choose action
                action = players[current_player].move(board)

                # take action, observe impact
                play_again = board.sow_seeds(action, current_player)
                if board.is_end() and np.argmax(board.banks)+1 == current_player:
                    reward = 1
                elif board.is_end():
                    reward = -1
                else:
                    reward = 0
                
                w = w + ALPHA * (reward + GAMMA*np.dot(w, board.vectorize(current_player)) - \
                        np.dot(w, state)) * state
                players[current_player] = MancalaAI(current_player, w)
            else:
                board.sow_seeds(players[current_player-1].move(board))

    print(w)


if __name__ == "__main__":
    sarsa()