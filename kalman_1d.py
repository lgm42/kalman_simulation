import numpy as np
import matplotlib.pyplot as plt
import random

position = np.array([], float)
speed = np.array([], float)
acceleration = np.array([], float)
dt = 1.0

p0 = 0.0
s0 = 3.0
a0 = 1.0

time = np.arange(0, 100, dt)

for t in time:
    #compute new values
    newPosition = p0 + s0 * dt + .5 * a0 * dt * dt
    newSpeed = s0 + a0 * dt
    #change acceleration randomly
    newAcceleration = a0 + random.random() - .5
    #aggregate
    position = np.append(position, newPosition)
    speed = np.append(speed, newSpeed)
    acceleration = np.append(acceleration, newAcceleration)

    #update last value for next loop
    p0 = newPosition
    s0 = newSpeed
    a0 = newAcceleration

#charting
plt.subplot(3, 1, 1)
plt.plot(time, acceleration)
plt.xlabel('time (s)')
plt.ylabel('acceleration')
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(time, speed)
plt.xlabel('time (s)')
plt.ylabel('speed')
plt.subplot(3, 1, 3)
plt.plot(time, position)
plt.xlabel('time (s)')
plt.ylabel('position')

plt.tight_layout()
plt.show()