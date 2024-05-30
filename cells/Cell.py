from numpy import ndarray, array

class Cell(object):
    neighbors: ndarray = None
    state = 0

    def __init__(self, state_func: callable):
        self.state_func = state_func

    def set_neighbors(self, *neighbors):
        self.neighbors = array(neighbors)

    def new_state(self):
        self.state = self.state_func(self.neighbors, self.state)

    def set_state(self, new_state: int):
        self.state = new_state

    def get_state(self):
        return self.state
