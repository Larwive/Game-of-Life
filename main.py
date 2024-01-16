from numpy import ndarray, random, zeros
import matplotlib.pyplot as plt

couples: tuple = ((-1, 1), (0, 1), (1, 1), (-1, 0), (0, 0), (1, 0), (-1, -1), (0, -1), (1, -1))
weights: tuple = (0.125, 0.125, 0.125, 0.125, 1, 0.125, 0.125, 0.125, 0.125)

def mean(arr:ndarray, x:int, y:int, coords=couples, weights=weights):
    s: int = 0
    for c, w in zip(coords, weights):
        dx, dy = c
        s+=arr[(x+dx)%arr.shape[0], (y+dy)%arr.shape[1]]*w
    arr[x, y] = s/sum(weights)

test_map: ndarray = random.rand(100, 100)

plt.imshow(test_map)
plt.axis('off')
plt.show()
plt.pause(1)
for i in range(100):
    for j in range(100):
        mean(test_map, i, j)
plt.imshow(test_map)
plt.axis('off')
plt.show()