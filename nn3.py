import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define the convolutional kernel for counting neighbors
kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]], dtype=np.float32)


class GameOfLifeLayer(nn.Module):
    def __init__(self):
        super(GameOfLifeLayer, self).__init__()

    def forward(self, neighbors, current_state):
        # Apply the Game of Life rules
        new_state = torch.where(
            (current_state == 1) & ((neighbors == 2) | (neighbors == 3)),
            torch.tensor(1.0, device=current_state.device),
            torch.where(
                (current_state == 0) & (neighbors == 3),
                torch.tensor(1.0, device=current_state.device),
                torch.tensor(0.0, device=current_state.device)
            )
        )
        return new_state.unsqueeze(1)  # Add channel dimension back


class GameOfLifeModel(nn.Module):
    def __init__(self, kernel):
        super(GameOfLifeModel, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv.weight.data = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)
        self.gol = GameOfLifeLayer()

    def forward(self, x):
        neighbors = self.conv(x)
        new_state = self.gol(neighbors.squeeze(1), x.squeeze(1))
        return new_state


def animate_with_model(model, size=100, nb_iter=200, delay=.00001, device='mps'):
    grid = torch.randint(0, 2, (1, 1, size, size), dtype=torch.float32).to(device)
    changes = torch.ones_like(grid, dtype=torch.bool)  # Initially, consider all cells as changed
    plt.imshow(grid.cpu().numpy()[0, 0], cmap='gray')
    plt.axis('off')
    plt.pause(delay)

    for _ in range(nb_iter):
        # Get the indices of changed cells
        change_indices = changes.nonzero(as_tuple=True)

        # Create a mask of neighbors to be updated
        update_mask = torch.zeros_like(grid, dtype=torch.bool)
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                shift_i, shift_j = i - 1, j - 1
                update_mask[:, :, max(0, -shift_i):min(size, size - shift_i),
                max(0, -shift_j):min(size, size - shift_j)] |= \
                    changes[:, :, max(0, shift_i):min(size, size + shift_i), max(0, shift_j):min(size, size + shift_j)]

        # Update only the regions indicated by the update mask
        updated_grid = grid.clone()
        updated_grid[update_mask] = model(grid)[update_mask]

        # Determine which cells have changed
        changes = (grid != updated_grid)
        grid = updated_grid

        # Plot the grid
        plt.clf()
        plt.imshow(grid.cpu().detach().numpy()[0, 0], cmap='gray')
        plt.axis('off')
        plt.pause(delay)


# Example usage
size = 10000
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = GameOfLifeModel(kernel).to(device)
animate_with_model(model, size, nb_iter=200, device=device)
