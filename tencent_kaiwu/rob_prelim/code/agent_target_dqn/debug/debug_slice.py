import numpy as np

x = np.zeros((5, 5, 3))
y = x.copy()
y[2, 2] = 1.0
print(x, y, sep='\n', end='\n\n')
x[y > 0] = np.array([1,1,1]) * y[y > 0]
print(x, y, sep='\n')
