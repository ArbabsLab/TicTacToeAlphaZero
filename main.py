import numpy as np

class TicTacToe:
    def __init__(self) -> None:
        self.row = 3
        self.col = 3
        self.board = self.row * self.col

    def get_initial_state(self):
        return np.zeros((self.row, self.col))
    
    def get_next_state(self, state, action, player):
        row = action // self.col
        col = action % self.col
        state[row, col] = player
        return state
    
    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        row = action // self.col
        col = action % self.col
        player = state[row, col]

        return (
            np.sum(state[row, :]) == player * self.col
            or np.sum(state[:, row]) == player * self.row
            or np.sum(np.diag(state)) == player * self.row
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row
        )
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
tictactoe = TicTacToe()
player = 1

state = tictactoe.get_initial_state()

while True:
    print(state)
    valid_moves = tictactoe.get_valid_moves(state)
    print("valid moves", [i for i in range(tictactoe.board) if valid_moves[i] == 1])
    action = int(input(f"{player}:"))

    if valid_moves[action] == 0:
        print("Not Valid")
        continue

    state = tictactoe.get_next_state(state, action, player)
    value, is_terminal = tictactoe.get_value_and_terminated(state, action)

    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break

    player = tictactoe.get_opponent(player)