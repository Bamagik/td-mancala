# Mancala
# Quintin Reed
# Mancala actions and play for the game

from itertools import cycle

PIT_COUNT = 12

def player_pits(player):
    return range( 0+(6*(player-1)), 6+(6*(player-1)) )

class Board():
    banks = [0] * 2
    pits = [4] * PIT_COUNT

    def check_capture(self, last_pit: int, player: int):
        return self.pits[last_pit] == 1 and last_pit in player_pits(player)

    def is_end(self):
        if sum(self.pits[0:6]) == 0 or sum(self.pits[6:12]) == 0:
            banks[0] += sum(self.pits[0:6])
            banks[1] += sum(self.pits[6:12])
            return True
        return False

    def print_board(self):
        print(" ", " ".join([f"{p}" for p in player_pits(2)[::-1]]), " ", sep="")
        print("-"*20)
        print(" ", " ".join([f"{p}" for p in self.pits[11:5:-1]]), " ")
        print(self.banks[1], " "*11, self.banks[0])
        print(" ", " ".join([f"{p}" for p in self.pits[0:6]]), " ")
        print("-"*20)
        print(" ", " ".join([f"{p}" for p in player_pits(1)]), " ")

    def sow_seeds(self, pit_idx: int, player: int):
        seeds = self.pits[pit_idx]
        self.pits[pit_idx] = 0
        previous_pit = pit_idx
        pit_idx += 1

        while seeds != 0:
            if pit_idx == PIT_COUNT:
                # to make it a cycle loop back to 0
                pit_idx = 0
            if pit_idx == 0 and previous_pit == 11 and player == 2:
                self.banks[player-1] += 1
                previous_pit = pit_idx
            elif pit_idx == 6 and previous_pit == 5 and player == 1:
                self.banks[player-1] += 1
                previous_pit = pit_idx
            else:
                self.pits[pit_idx] += 1
                # only update pit_idx when dropping in a pit
                previous_pit = pit_idx
                pit_idx += 1
            seeds -= 1

        # on capture at end of sowing
        if self.check_capture(previous_pit, player):
            self.banks[player-1] += self.pits[previous_pit]
            self.pits[previous_pit] = 0

            op_side_pit = (PIT_COUNT - 1) - previous_pit
            self.banks[player-1] += self.pits[op_side_pit]
            self.pits[op_side_pit] = 0
    
        # return if end of sowing was a bank
        return previous_pit == pit_idx


if __name__ == "__main__":
    # play the game
    board = Board()

    # player names
    player_turns = cycle([1,2])
    whos_turn = next(player_turns)

    while not board.is_end():
        board.print_board()
        print()
        pit = int(input(f"Player {whos_turn} move:"))
        if pit not in player_pits(whos_turn):
            print("Pit is not the player's, choose again")
            continue
        else:
            play_again = board.sow_seeds(pit, whos_turn)
            if play_again:
                print(f"Player {whos_turn} goes again")
                continue
            whos_turn = next(player_turns)