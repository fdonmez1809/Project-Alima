import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

def generate_shape(num_points=5):
    angles = np.linspace(0, 2 * np.pi, num_points) # generates 'num_points' number of angles between 0 and 2π
    radius = 1 + 0.3 * np.random.randn(num_points) # creates 'num_points' random values from a normal distribution.
    x = radius * np.sin(angles) # x-coordination
    y = radius * np.cos(angles) # y-coordination
    return x, y

# x, y = generate_shape(10)  


# x = np.append(x, x[0])
# y = np.append(y, y[0])

# plt.figure(figsize=(5, 5))
# plt.plot(x, y, 'o-', linewidth=2)
# plt.fill(x, y, alpha=0.3)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()
        
#for i in range(5):
#    number = np.random.randint(10)
#    x, y = generate_shape(10)  
#    x = np.append(x, x[0])
#    y = np.append(y, y[0])
#    plt.figure(figsize=(5, 5))
#    plt.plot(x, y, 'o-', linewidth=2)
#    plt.fill(x, y, alpha=0.3)
#    plt.gca().set_aspect('equal', adjustable='box')
#    plt.show()


plt.figure(figsize=(5, 5))

for i in range(5):
    x, y = generate_shape(10)  
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    
    # clear the current figure
    plt.clf()

    # plot the new shape
    plt.plot(x, y, 'o-', linewidth=2)
    plt.fill(x, y, alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # display the shape
    plt.pause(1)  # then pause to display each shape for 1 second

plt.show()
