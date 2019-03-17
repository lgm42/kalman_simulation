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

TotalTime = 1000
dt = 1.0
noiseLevel = 5.0
maxSpeed = 10
PositionAsservPCoeff = 1.0
MaxAcceleration = 10.0

pX0 = 0.0
sX0 = 0.0
aX0 = 1.0

pY0 = 0.0
sY0 = 0.0
aY0 = 1.0

time = np.arange(0, TotalTime, dt)

#objective = np.array([[500, 500], [1000, 0], [500, 0]])
objective = np.array([])
currentObjective = 0
objectiveDuration = 150
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

    #change acceleration following objective ?
    if (len(objective) > 0):
        accXDelta = objective[currentObjective, 0] - newPositionX 
        accYDelta = objective[currentObjective, 1] - newPositionY

        if (accXDelta > 0):
            if (sX0 < maxSpeed):
                aX0 = aX0 + random.random() * accXDelta * PositionAsservPCoeff
            else:
                aX0 = 0
        else:
            if (sX0 > -maxSpeed):
                aX0 = aX0 + random.random() * accXDelta * PositionAsservPCoeff
            else:
                aX0 = 0

        if (accYDelta > 0):
            if (sY0 < maxSpeed):
                aY0 = aY0 + random.random() * accYDelta * PositionAsservPCoeff
            else:
                aY0 = 0
        else:
            if (sY0 > -maxSpeed):
                aY0 = aY0 + random.random() * accYDelta * PositionAsservPCoeff
            else:
                aY0 = 0

        #manage bounds
        aX0 = max(-1 * MaxAcceleration, min(MaxAcceleration, aX0))
        aY0 = max(-1 * MaxAcceleration, min(MaxAcceleration, aY0))

        #change objective ?
        if (t > (currentObjective + 1) * objectiveDuration) and (currentObjective < len(objective) - 1):
            currentObjective += 1
    else:
        #no objectives
        aX0 = aX0 + (random.random() - .5) * 3.0
        aY0 = aY0 + (random.random() - .5) * 3.0

#kalman computation
#on ne peut utiliser que le tableau accelerometer

estimatedAccelerationX = np.array([], float)
estimatedSpeedX = np.array([], float)
estimatedPositionX = np.array([], float)
estimatedAccelerationY = np.array([], float)
estimatedSpeedY = np.array([], float)
estimatedPositionY = np.array([], float)

pX0 = 0.0
sX0 = 0.0
aX0 = accelerometerX[0]
pY0 = 0.0
sY0 = 0.0
aY0 = accelerometerY[0]

#Xkp = [ X  ]
#      [ Y ]
#      [ .X ]
#      [ .Y ]
#      [ ..X]
#      [ ..Y]


#F = [ 1   0     dt    0     .5*dt²   0     ]
#    [ 0   1     0     dt      0     .5*dt² ]
#    [ 0   0     1     0       dt     0     ]
#    [ 0   0     0     1       0      dt    ]
#    [ 0   0     0     0       1      0     ]
#    [ 0   0     0     0       0      1     ]


F = np.matrix([[1, 0, dt, 0, .5 * dt * dt, 0],
               [0, 1, 0, dt, 0, .5 * dt * dt],
               [0, 0, 1, 0, dt, 0], 
               [0, 0, 0, 1, 0, dt], 
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 1]])

#Matrice de covariance des erreurs

Pk0 = np.identity(6, dtype = float)
 
Yk0 = np.matrix([[0], [0], [0], [0], [aX0], [aY0]])
 

Xkp0 = np.matrix([[pX0], 
                  [pY0], 
                  [sX0], 
                  [sY0], 
                  [aX0], 
                  [aY0]])

Q = 1.0 * np.identity(6, dtype = float)
H = np.matrix([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 1]])

 
#Error matrix
R = 100.0 * np.identity(6, dtype = float)
index = 0
 
for t in time:
    aX0 = accelerometerX[index]
    aY0 = accelerometerY[index]
    Zk = np.matrix([[0], [0], [0], [0], [aX0], [aY0]])
# 
#    X = X + .X * dt + ..X * .5 * dt²
#    .X = .X + ..X * dt
#    ..X = ..X
# 
    Xkp1 = F * Xkp0
# 
#    Estimation de la matrice de covariance
    Pk1 = F * Pk0 * F.transpose() + Q
# 
#    innovation
    Yk = Zk - H * Xkp1
# 
#    covariance de l'innovation
    Sk = H * Pk1 * H.transpose() + R
# 
 #   gain de Kalman optimal
    K = Pk1 * H.transpose() * inv(Sk)
# 
#    état mis à jour
    Xkp2 = Xkp1 + K * Yk
# 
#    covariance mise à jour
    Pk2 = (np.identity(6, dtype = float) - (K * H)) * Pk1
# 
#    on loggue les valeur estimée par le filtre de kalman
    estimatedAccelerationX = np.append(estimatedAccelerationX, Xkp2[4, 0])
    estimatedSpeedX = np.append(estimatedSpeedX, Xkp2[2, 0])
    estimatedPositionX = np.append(estimatedPositionX, Xkp2[0, 0])
    estimatedAccelerationY = np.append(estimatedAccelerationY, Xkp2[5, 0])
    estimatedSpeedY = np.append(estimatedSpeedY, Xkp2[3, 0])
    estimatedPositionY = np.append(estimatedPositionY, Xkp2[1, 0])
# 
#    on remplace les valeurs pour le tour de boucle suivant
    Xkp0 = Xkp2
    Pk0 = Pk2
    index += 1

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
ax.legend(('réelle', 'estimée'))
plt.plot(positionX, positionY)
plt.plot(estimatedPositionX, estimatedPositionY)
#ax.scatter(positionX, positionY, alpha=0.5, marker=markers.MarkerStyle('.'), s=1)
#ax.scatter(estimatedPositionX, estimatedPositionY, marker=markers.MarkerStyle('.'), s=1)
plt.title('position réèlle')

ax = fig.add_subplot(gs[1, 2])
ax.plot(time, estimatedPositionX)
ax.plot(time, positionX)
ax.set_xlabel('time (s)')
ax.legend(('position estimée X', 'position réelle'))
plt.title('Position estimée X')

ax = fig.add_subplot(gs[2, 2])
ax.plot(time, estimatedPositionY)
ax.plot(time, positionY)
ax.set_xlabel('time (s)')
ax.legend(('position estimée Y', 'position réelle'))
plt.title('Position estimée Y')

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