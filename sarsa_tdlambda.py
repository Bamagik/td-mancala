import numpy as np
import copy
import mancala
import time

ALPHA = 10e-6
EPSILON = 0.1
GAMMA = 0.5
LAMBDA = 0.1

NUM_WEIGHTS = 98

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
            action = i + (self.player-1)*6
            if board.pits[action] != 0:
                av = self.calculate_future_state(action, board, lookahead)

                if av < best_av:
                    best_action = action
                    best_av = av
        if best_action is None:
            board.print_board()
            
        return best_action, best_av

    def calculate_future_state(self, action, board, lookahead=True):
        new_board = copy.deepcopy(board)

        play_again = new_board.sow_seeds(action, self.player)

        # generalize what the AI would think the new future would be
        if lookahead and not play_again:
            new_board.sow_seeds(self.choose_action(board, lookahead=False)[0], 2 - self.player)

        av = np.dot(self.W, new_board.vectorize(self.player))

        return av

    def move(self, board):
        if np.random.random() >= self.epsilon:
            action, q = self.choose_action(board)
        else:
            action = np.random.randint(6) + (self.player-1)*6
            while board.pits[action] == 0:
                action = np.random.randint(6) + (self.player-1)*6
        return action, self.calculate_future_state(action, board)


def sarsa():
    """
    v^ here is defined as X*W
    """
    policy = None
    w = np.random.random((NUM_WEIGHTS,))
    
    for i in range(10000):
        board = mancala.Board()

        player1 = MancalaAI(1, w)
        player2 = MancalaAI(2, w)
        players = [player1, player2]

        ai_player = 1
        current_player = 1

        state = board.vectorize(current_player)

        Qold = 0
        z = np.zeros((NUM_WEIGHTS,))
        delta = 0
        action = None

        turn = 0

        last_banks = None

        while not board.is_end():
            play_again = False
            if ai_player == current_player:
                reward = 0
                new_state = board.vectorize(current_player)

                if action is None:
                    # previous_bank_val = board.banks[current_player-1]
                    action, av = players[current_player].move(board)

                    Q = w.T @ state
                    Q_prime = w.T @ new_state

                    delta = reward + GAMMA * Q_prime - Q
                    z = GAMMA * LAMBDA * z + (1 - ALPHA * GAMMA * LAMBDA * z.T @ state) * state

                    w = w + ALPHA * (delta + Q - Qold) * z - \
                        ALPHA * (Q - Qold) * state
                    
                    players[current_player].W = w

                    Qold = Q 
                else:
                    action, av = players[current_player].move(board)
                    
                state = new_state
                play_again = board.sow_seeds(action, current_player)
            else:
                play_again = board.sow_seeds(players[current_player-1].move(board), current_player)

            if not play_again:
                current_player = 2 - current_player

            turn += 1

            if board.is_end() and np.argmax(board.banks)+1 == ai_player:
                reward = 10
            elif board.is_end():
                reward = -10

        z = GAMMA * LAMBDA * z + (1 - ALPHA * GAMMA * LAMBDA * z.T @ state) * state
        w = w + ALPHA * (reward - Qold) *  - ALPHA * (Q - Qold) * state

    return w


if __name__ == "__main__":
    start = time.time()

    weights = sarsa()

    print('Took', time.time() - start, 'seconds')

    print(weights)

    with open('trained_weights.txt', 'w') as file:
        file.writelines([f"{w}\n" for w in weights])