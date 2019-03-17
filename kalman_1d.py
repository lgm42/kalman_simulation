import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import inv
import random

position = np.array([], float)
speed = np.array([], float)
acceleration = np.array([], float)
accelerometer = np.array([], float)
dt = 1.0
noiseLevel = 5.0

p0 = 0.0
s0 = 0.0
a0 = 1.0

time = np.arange(0, 100, dt)

#simulation

for t in time:
    #compute new values
    newPosition = p0 + s0 * dt + .5 * a0 * dt * dt
    newSpeed = s0 + a0 * dt

    #save
    position = np.append(position, newPosition)
    speed = np.append(speed, newSpeed)
    acceleration = np.append(acceleration, a0)

    #accelerometer values
    accelerometer = np.append(accelerometer, a0 + (random.random() - .5) * noiseLevel)

    #update last value for next loop
    p0 = newPosition
    s0 = newSpeed

    #change acceleration randomly
    a0 = a0 + (random.random() - .5) * 3.0

#kalman computation
#on ne peut utiliser que le tableau accelerometer

estimatedAcceleration = np.array([], float)
estimatedSpeed = np.array([], float)
estimatedPosition = np.array([], float)

p0 = 0.0
s0 = 0.0
a0 = accelerometer[0]

#F = [ 1   dt  .5*dt²]
#    [ 0    1     dt ]
#    [ 0    0     1  ]
F = np.matrix([[1, dt, .5 * dt * dt], [0, 1, dt], [0, 0, 1]])

#Matrice de covariance des erreurs
# a confirmer pour l'intiialisation de la matrice de covariance
Pk0 = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

Yk0 = np.matrix([[0], [0], [a0]])

#Xkp = [ X  ]
#      [ .X ]
#      [ ..X]
Xkp0 = np.matrix([[p0], [s0], [a0]])

Q = 1 * np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
H = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

#Error matrix
positionError = 1.0
speedError = 1.0
accelerationError = 1.0
R = 3*np.matrix([[positionError, 0, 0], [0, speedError, 0], [0, 0, accelerationError]])
index = 0

for t in time:
    a0 = accelerometer[index]
    Zk = np.matrix([[0], [0], [a0]])

    #X = X + .X * dt + ..X * .5 * dt²
    #.X = .X + ..X * dt
    #..X = ..X

    Xkp1 = F * Xkp0

    #Estimation de la matrice de covariance
    Pk1 = F * Pk0 * F.transpose() + Q

    #innovation
    Yk = Zk - H * Xkp1

    #covariance de l'innovation
    Sk = H * Pk1 * H.transpose() + R

    #gain de Kalman optimal
    K = Pk1 * H.transpose() * inv(Sk)

    #état mis à jour
    Xkp2 = Xkp1 + K * Yk

    #covariance mise à jour
    Pk2 = (np.identity(3, dtype = float) - (K * H)) * Pk1

    #on loggue les valeur estimée par le filtre de kalman
    estimatedAcceleration = np.append(estimatedAcceleration, Xkp2[2, 0])
    estimatedSpeed = np.append(estimatedSpeed, Xkp2[1, 0])
    estimatedPosition = np.append(estimatedPosition, Xkp2[0, 0])

    #on remplace les valeurs pour le tour de boucle suivant
    Xkp0 = Xkp2
    Pk0 = Pk2
    index += 1

#charting
fig = plt.figure(constrained_layout=True)
gs = gridspec.GridSpec(3, 2, figure=fig)
ax = fig.add_subplot(gs[0, 0])
ax.plot(time, acceleration)
ax.plot(time, accelerometer)
ax.set_xlabel('time [s]')
ax.legend(('acceleration', 'accelerometer'))
plt.title('Acceleration')

ax = fig.add_subplot(gs[1, 0])
ax.plot(time, speed)
ax.set_xlabel('time (s)')
plt.grid(True)
plt.title('Speed')

ax = fig.add_subplot(gs[2, 0])
ax.plot(time, position)
ax.set_xlabel('time (s)')
plt.title('Position')

#resultats
ax = fig.add_subplot(gs[0, 1])
ax.plot(time, estimatedAcceleration)
ax.plot(time, accelerometer)
ax.plot(time, acceleration)
ax.set_xlabel('time (s)')
ax.legend(('acceleration', 'accelerometer', 'reelle'))
plt.title('Accelération estimée')

ax = fig.add_subplot(gs[1, 1])
ax.plot(time, estimatedSpeed)
ax.plot(time, speed)
ax.set_xlabel('time (s)')
ax.legend(('vitesse estimée', 'vitesse réelle'))
plt.title('Vitesse estimée')

ax = fig.add_subplot(gs[2, 1])
ax.plot(time, estimatedPosition)
ax.plot(time, position)
ax.set_xlabel('time (s)')
ax.legend(('position estimée', 'position réelle'))
plt.title('Position estimée')

plt.tight_layout()
plt.show()