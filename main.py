import numpy as np
import mcts

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
        if action == None:
            return False
        
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
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        return encoded_state
    


