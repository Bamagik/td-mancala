import numpy as np
import copy
import mancala

ALPHA = 10e-4 
EPSILON = 0.1
GAMMA = 0.1

class MancalaAI():
    def __init__(self, player, W=np.random.random((14)), epsilon=EPSILON):
        self.player = player
        self.W = W
        self.epsilon = epsilon

    @staticmethod
    def load_AI(path, player):
        w = []
        with open(path, 'r') as file:
            for line in file.readlines():
                w.append(float(line))
        return MancalaAI(player, W=np.array(w))
    
    def choose_action(self, board, lookahead=True):
        best_action = None
        best_av = np.inf
        for i in range(6):
            if board.pits[i + (self.player-1)*6] != 0:
                new_board = copy.deepcopy(board)
                
                play_again = new_board.sow_seeds(i + (self.player-1)*6, self.player)

                # generalize what the AI would think the new future would be
                if lookahead and not play_again:
                    new_board.sow_seeds(self.choose_action(board, lookahead=False), self.player)

                av = np.dot(self.W, new_board.vectorize(self.player))

                if av < best_av:
                    best_action = i + (self.player-1)*6
                    best_av = av
        if best_action is None:
            board.print_board()
            
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
    w = np.random.random((19,))
    
    for i in range(10000):
        board = mancala.Board()

        player1 = MancalaAI(1, w)
        player2 = MancalaAI(2, w)
        players = [player1, player2]

        ai_player = 1
        current_player = 1

        state = board.vectorize(current_player)
        action = None
        reward = 0
        while not board.is_end():
            play_again = False
            if ai_player == current_player:
                w = w + ALPHA * (reward - GAMMA*np.dot(w, board.vectorize(current_player)) - \
                        np.dot(w, state)) * state
                players[current_player] = MancalaAI(current_player, w)

                # save current state
                state = board.vectorize(current_player)

                # choose action
                action = players[current_player].move(board)

                previous_bank_val = board.banks[current_player-1]
                # take action, observe impact
                play_again = board.sow_seeds(action, current_player)

                reward = board.banks[current_player-1] - previous_bank_val

                if board.is_end() and np.argmax(board.banks)+1 == current_player:
                    reward = 100
                elif board.is_end():
                    reward = -100
                else:
                    reward = 0
            else:
                play_again = board.sow_seeds(players[current_player-1].move(board), current_player)

            if not play_again:
                current_player = 2 - current_player

        w = w + ALPHA * (reward - np.dot(w, state)) * state

    return w


if __name__ == "__main__":
    weights = sarsa()

    print(weights)

    with open('trained_weights.txt', 'w') as file:
        file.writelines([f"{w}\n" for w in weights])