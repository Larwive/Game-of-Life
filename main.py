from numpy import ndarray, random
import matplotlib.pyplot as plt

test_map: ndarray = random.rand(100, 100)

plt.imshow(test_map)
plt.axis('off')
plt.show()

