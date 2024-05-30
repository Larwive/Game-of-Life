import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]], dtype=np.float32)


class GameOfLifeLayer(nn.Module):
    def __init__(self):
        super(GameOfLifeLayer, self).__init__()

    def forward(self, neighbors, current_state):
        new_state = torch.where(
            (current_state == 1) & ((neighbors == 2) | (neighbors == 3)),
            torch.tensor(1.0, device=current_state.device),
            torch.where(
                (current_state == 0) & (neighbors == 3),
                torch.tensor(1.0, device=current_state.device),
                torch.tensor(0.0, device=current_state.device)
            )
        )
        return new_state.unsqueeze(1)

class GameOfLife2Layer(nn.Module):
    def __init__(self):
        super(GameOfLife2Layer, self).__init__()

    def forward(self, neighbors, current_state):
        new_state = torch.where(
            (current_state == 0) & ((neighbors == 2) | (neighbors == 3)),
            torch.tensor(1.0, device=current_state.device),
            torch.where(
                (current_state == 0) & (neighbors == 3),
                torch.tensor(1.0, device=current_state.device),
                torch.tensor(0.0, device=current_state.device)
            )
        )
        return new_state.unsqueeze(1)

class GameOfLife3Layer(nn.Module):
    def __init__(self):
        super(GameOfLife3Layer, self).__init__()

    def forward(self, neighbors, current_state):
        new_state = torch.where(
            (current_state == 1) & ((neighbors == 1) | (neighbors == 3)),
            torch.tensor(1.0, device=current_state.device),
            torch.where(
                (current_state == 0) & (neighbors == 3),
                torch.tensor(1.0, device=current_state.device),
                torch.tensor(0.0, device=current_state.device)
            )
        )
        return new_state.unsqueeze(1)


class GameOfLife4Layer(nn.Module):
    def __init__(self):
        super(GameOfLife4Layer, self).__init__()

    def forward(self, neighbors, current_state):
        new_state = torch.where(
            (current_state == 0) & ((neighbors == 2) | (neighbors == 3)),
            torch.tensor(1.0, device=current_state.device),
            torch.where(
                (current_state == 0) & (neighbors == 3),
                torch.tensor(1.0, device=current_state.device),
                torch.tensor(0.0, device=current_state.device)
            )
        )
        return new_state.unsqueeze(1)

class Var3Layer(nn.Module):
    def __init__(self):
        super(Var3Layer, self).__init__()

    def forward(self, neighbors, current_state):
        new_state = torch.where(
            (current_state == 1),
            torch.where((torch.Tensor.bool(neighbors % 2)), torch.tensor(1.0, device=current_state.device),
                        torch.tensor(0.0, device=current_state.device)),
            torch.where(
                (neighbors == 1),
                torch.tensor(1.0, device=current_state.device),
                torch.tensor(0.0, device=current_state.device),
            )
        )
        return new_state.unsqueeze(1)


class Var4Layer(nn.Module):
    def __init__(self):
        super(Var4Layer, self).__init__()

    def forward(self, neighbors, current_state):
        new_state = torch.where(
            (current_state == 1),
            torch.where((torch.Tensor.bool((neighbors + 1) % 2)), torch.tensor(1.0, device=current_state.device),
                        torch.tensor(0.0, device=current_state.device)),
            torch.where(
                (neighbors == 1),
                torch.tensor(1.0, device=current_state.device),
                torch.tensor(0.0, device=current_state.device),
            )
        )
        return new_state.unsqueeze(1)


class GameOfLifeModel(nn.Module):
    def __init__(self, kernel, rules):
        super(GameOfLifeModel, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv.weight.data = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)
        self.gol = rules()

    def forward(self, x):
        neighbors = self.conv(x)
        new_state = self.gol(neighbors.squeeze(1), x.squeeze(1))
        return new_state


def animate_with_model(model, size=100, nb_iter=200, delay=.000001, device=torch.device('mps')):
    grid = torch.randint(0, 2, (1, 1, size, size), dtype=torch.float32).to(device)
    plt.imshow(grid.cpu().numpy()[0, 0], cmap='gray')
    plt.axis('off')
    plt.pause(delay)

    for _ in range(nb_iter):
        grid = model(grid)
        plt.clf()
        plt.imshow(grid.cpu().detach().numpy()[0, 0], cmap='gray')
        plt.axis('off')
        plt.pause(delay)
    plt.show()


size = 200
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using {device}.")
# model = GameOfLifeModel(kernel, GameOfLifeLayer).to(device)
#model = GameOfLifeModel(kernel, Var3Layer).to(device)
model = GameOfLifeModel(kernel, GameOfLife3Layer).to(device)
animate_with_model(model, size, delay=0.01, nb_iter=600, device=device)
