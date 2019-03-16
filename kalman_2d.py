import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.markers as markers
from numpy.linalg import inv
import random

positionX = np.array([], float)
speedX = np.array([], float)
accelerationX = np.array([], float)
accelerometerX = np.array([], float)
positionY = np.array([], float)
speedY = np.array([], float)
accelerationY = np.array([], float)
accelerometerY = np.array([], float)

dt = 1.0
noiseLevel = 5.0

pX0 = 0.0
sX0 = 0.0
aX0 = 1.0

pY0 = 0.0
sY0 = 0.0
aY0 = 1.0

time = np.arange(0, 1000, dt)

#simulation

for t in time:
    #compute new values
    newPositionX = pX0 + sX0 * dt + .5 * aX0 * dt * dt
    newSpeedX = sX0 + aX0 * dt
    newPositionY = pY0 + sY0 * dt + .5 * aY0 * dt * dt
    newSpeedY = sY0 + aY0 * dt

    #save
    positionX = np.append(positionX, newPositionX)
    speedX = np.append(speedX, newSpeedX)
    accelerationX = np.append(accelerationX, aX0)
    positionY = np.append(positionY, newPositionY)
    speedY = np.append(speedY, newSpeedY)
    accelerationY = np.append(accelerationY, aY0)

    #accelerometer values
    accelerometerX = np.append(accelerometerX, aX0 + (random.random() - .5) * noiseLevel)
    accelerometerY = np.append(accelerometerY, aY0 + (random.random() - .5) * noiseLevel)

    #update last value for next loop
    pX0 = newPositionX
    sX0 = newSpeedX
    pY0 = newPositionY
    sY0 = newSpeedY

    #change acceleration randomly
    aX0 = aX0 + (random.random() - .5) * 3.0
    aY0 = aY0 + (random.random() - .5) * 3.0

#kalman computation
#on ne peut utiliser que le tableau accelerometer

# estimatedAcceleration = np.array([], float)
# estimatedSpeed = np.array([], float)
# estimatedPosition = np.array([], float)
# 
# p0 = 0.0
# s0 = 0.0
# a0 = accelerometer[0]
# 
#F = [ 1   dt  .5*dt²]
#   [ 0    1     dt ]
#   [ 0    0     1  ]
# F = np.matrix([[1, dt, .5 * dt * dt], [0, 1, dt], [0, 0, 1]])
# 
#Matrice de covariance des erreurs
#a confirmer pour l'intiialisation de la matrice de covariance
# Pk0 = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# 
# Yk0 = np.matrix([[0], [0], [a0]])
# 
#Xkp = [ X  ]
#     [ .X ]
#     [ ..X]
# Xkp0 = np.matrix([[p0], [s0], [a0]])
# 
# Q = 1 * np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# H = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
#H = np.identity(3, dtype = float)
# 
#Error matrix
# positionError = 1.0
# speedError = 1.0
# accelerationError = 1.0
# R = 1*np.matrix([[positionError, 0, 0], [0, speedError, 0], [0, 0, accelerationError]])
# index = 0
# 
# for t in time:
    # a0 = accelerometer[index]
    # Zk = np.matrix([[0], [0], [a0]])
# 
#    X = X + .X * dt + ..X * .5 * dt²
#    .X = .X + ..X * dt
#    ..X = ..X
# 
    # Xkp1 = F * Xkp0
# 
#    Estimation de la matrice de covariance
    # Pk1 = F * Pk0 * F.transpose() + Q
# 
#    innovation
    # Yk = Zk - H * Xkp1
# 
#    covariance de l'innovation
    # Sk = H * Pk1 * H.transpose() + R
# 
 #   gain de Kalman optimal
    # K = Pk1 * H.transpose() * inv(Sk)
# 
#    état mis à jour
    # Xkp2 = Xkp1 + K * Yk
# 
#    covariance mise à jour
    # Pk2 = (np.identity(3, dtype = float) - (K * H)) * Pk1
# 
#    on loggue les valeur estimée par le filtre de kalman
    # estimatedAcceleration = np.append(estimatedAcceleration, Xkp2[2, 0])
    # estimatedSpeed = np.append(estimatedSpeed, Xkp2[1, 0])
    # estimatedPosition = np.append(estimatedPosition, Xkp2[0, 0])
# 
#    on remplace les valeurs pour le tour de boucle suivant
    # Xkp0 = Xkp2
    # Pk0 = Pk2
    # index += 1

#charting
fig = plt.figure(constrained_layout=True)
gs = gridspec.GridSpec(3, 3, figure=fig)

ax = fig.add_subplot(gs[0, 0])
ax.plot(time, accelerationX)
ax.plot(time, accelerometerX)
ax.set_xlabel('time [s]')
ax.legend(('acceleration', 'accelerometer'))
plt.title('Acceleration')

ax = fig.add_subplot(gs[1, 0])
ax.plot(time, speedX)
ax.set_xlabel('time (s)')
plt.grid(True)
plt.title('Speed')

ax = fig.add_subplot(gs[2, 0])
ax.plot(time, positionX)
ax.set_xlabel('time (s)')
plt.title('Position')

ax = fig.add_subplot(gs[0, 1])
ax.plot(time, accelerationY)
ax.plot(time, accelerometerY)
ax.set_xlabel('time [s]')
ax.legend(('acceleration', 'accelerometer'))
plt.title('Acceleration')

ax = fig.add_subplot(gs[1, 1])
ax.plot(time, speedY)
ax.set_xlabel('time (s)')
plt.grid(True)
plt.title('Speed')

ax = fig.add_subplot(gs[2, 1])
ax.plot(time, positionY)
ax.set_xlabel('time (s)')
plt.title('Position')

ax = fig.add_subplot(gs[0, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.scatter(positionX, positionY, alpha=0.5, marker=markers.MarkerStyle('.'), s=1)
plt.title('position réèlle')

#resultats
#ax = fig.add_subplot(gs[0, 1])
#ax.plot(time, estimatedAcceleration)
#ax.plot(time, accelerometer)
#ax.plot(time, acceleration)
#ax.set_xlabel('time (s)')
#ax.legend(('acceleration', 'accelerometer', 'reelle'))
#plt.title('Accelération estimée')
#
#ax = fig.add_subplot(gs[1, 1])
#ax.plot(time, estimatedSpeed)
#ax.plot(time, speed)
#ax.set_xlabel('time (s)')
#ax.legend(('vitesse estimée', 'vitesse réelle'))
#plt.title('Vitesse estimée')
#
#ax = fig.add_subplot(gs[2, 1])
#ax.plot(time, estimatedPosition)
#ax.plot(time, position)
#ax.set_xlabel('time (s)')
#ax.legend(('position estimée', 'position réelle'))
#plt.title('Position estimée')
#
plt.tight_layout()
plt.show()