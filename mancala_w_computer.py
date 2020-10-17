import sarsa_td0 as sarsa
from mancala import *
from itertools import cycle

if __name__ == "__main__":
    # play the game
    board = Board()

    # set computer
    computer_turn = int(input("What player is the computer? [1 or 2]: "))
    computer = sarsa.MancalaAI.load_AI('trained_weights.txt', computer_turn)

    # player names
    player_turns = cycle([1,2])
    whos_turn = next(player_turns)

    while not board.is_end():
        board.print_board()        
        print()

        play_again = False
        if whos_turn == computer_turn:
            comp_action = computer.choose_action(board)
            print(f"Computer's Move, {comp_action}")
            play_again = board.sow_seeds(comp_action, whos_turn)
        else:
            pit = int(input(f"Player {whos_turn} move: "))
            if pit not in player_pits(whos_turn):
                print("Pit is not the player's, choose again")
                continue
            elif board.pits[pit] == 0:
                print("Pit is empty, choose another")
                continue
            else:
                play_again = board.sow_seeds(pit, whos_turn)
        if play_again:
            print(f"Player {whos_turn} goes again")
        else:
            whos_turn = next(player_turns)

    print("Player", np.argmax(board.banks)+1, "won!")