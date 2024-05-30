import os
from numpy import ndarray, full, vectorize, average, array
from random import random
from cells.Cell import Cell
import matplotlib.pyplot as plt
from typing import Optional

from concurrent.futures import ThreadPoolExecutor, as_completed

vect_create_cell: ndarray[Cell] = vectorize(lambda func: Cell(func))
vect_get_state = vectorize(lambda cell: cell.get_state())
vect_new_state = vectorize(lambda cell: cell.new_state())


def mean(neighbors: ndarray, current_state: Optional[int] = None) -> int:
    return average(vect_get_state(neighbors))


def original(neighbors: ndarray, current_state: Optional[int] = None) -> int:
    s = sum(vect_get_state(neighbors))
    if current_state == 1:
        return s == 2 or s == 3
    return s == 3


def variante(neighbors: ndarray, current_state: Optional[int] = None) -> int:
    s = sum(vect_get_state(neighbors))
    if current_state == 1:
        return s % 2
    return s == 3


def variante2(neighbors: ndarray, current_state: Optional[int] = None) -> int:
    s = sum(vect_get_state(neighbors))
    if current_state == 1:
        return s % 2
    return (s + 1) % 2


def variante3(neighbors: ndarray, current_state: Optional[int] = None) -> int:
    s = sum(vect_get_state(neighbors))
    if current_state == 1:
        return s % 2
    return s == 1


def variante4(neighbors: ndarray, current_state: Optional[int] = None) -> int:
    s = sum(vect_get_state(neighbors))
    if current_state == 1:
        return (s + 1) % 2
    return s == 1


def activation(i: int, j: int) -> int:
    return i % 2 and j % 2

def initialize_cell_neighbors(test_map, sx, sy):
    for i in range(sx):
        for j in range(sy):
            im, ip, jm, jp = (i - 1) % sx, (i + 1) % sx, (j - 1) % sy, (j + 1) % sy
            test_map[i, j].set_neighbors(
                test_map[im, jp], test_map[i, jp], test_map[ip, jp], test_map[im, j],
                test_map[ip, j], test_map[im, jm], test_map[i, jm], test_map[ip, jm]
            )
            if True: #activation(i, j):
                test_map[i, j].set_state(random() > .5)
    return test_map

def update_states(test_map, chunk):
    for i, j in chunk:
        test_map[i, j].new_state()

def get_chunks(sx, sy, num_chunks):
    chunk_size = (sx * sy) // num_chunks
    chunks = []
    for n in range(num_chunks):
        chunk = [(i, j) for i in range(sx) for j in range(sy)][n * chunk_size:(n + 1) * chunk_size]
        chunks.append(chunk)
    return chunks

def animate(func: callable = original, size: int = 100, nb_iter: int = 200, delay: float = .0000001):
    """"#test_map: ndarray[Cell] = vect_create_cell(full((size, size), fill_value=func))
    #test_map: da.Array = da.full((1000, 1000), fill_value=func, chunks=(100, 100)).map_blocks(vect_create_cell, dtype=Cell).compute()
    test_map = create_cell_grid(func, size).compute()
    sx, sy = test_map.shape
    for i in range(sx):
        for j in range(sy):
            im, ip, jm, jp = (i - 1) % sx, (i + 1) % sx, (j - 1) % sy, (j + 1) % sy
            test_map[i, j].set_neighbors(test_map[im, jp], test_map[i, jp], test_map[ip, jp], test_map[im, j],
                                         # test_map[i, j],
                                         test_map[ip, j], test_map[im, jm], test_map[i, jm],
                                         test_map[ip, jm])
            if True: #activation(i, j):
                test_map[i, j].set_state(random() > .5)

    im_map: ndarray = vect_get_state(test_map)
    plt.imshow(im_map)
    plt.axis('off')
    # ig('test_map.png', bbox_inches='tight', pad_inches=0)
    plt.pause(delay)"""
    test_map = vect_create_cell(full((size, size), fill_value=func))

    sx, sy = test_map.shape
    test_map = initialize_cell_neighbors(test_map, sx, sy)

    im_map = vect_get_state(test_map)
    plt.imshow(im_map)
    plt.axis('off')
    plt.pause(delay)

    num_workers = os.cpu_count()
    print(f"Using {num_workers} workers.")
    chunks = get_chunks(sx, sy, num_workers)

    for _ in range(nb_iter):
        #vect_new_state(test_map)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(update_states, test_map, chunk) for chunk in chunks]
            for future in as_completed(futures):
                future.result()
        im_map = vect_get_state(test_map)
        plt.clf()
        plt.imshow(im_map)
        plt.axis('off')
        # plt.savefig('test_map{}.png'.format(_), bbox_inches='tight', pad_inches=0)

        plt.pause(delay)


animate(original, size=500, nb_iter=300)
