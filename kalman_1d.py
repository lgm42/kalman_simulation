import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

position = np.array([], float)
speed = np.array([], float)
acceleration = np.array([], float)
accelerometer = np.array([], float)
dt = 1.0
noiseLevel = 1.0

p0 = 0.0
s0 = 3.0
a0 = 1.0

time = np.arange(0, 100, dt)

#simulation

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

    #accelerometer values
    accelerometer = np.append(accelerometer, newAcceleration + (random.random() - .5) * noiseLevel)

    #update last value for next loop
    p0 = newPosition
    s0 = newSpeed
    a0 = newAcceleration


#kalman computation
#on ne peut utiliser que le tableau accelerometer

p0 = 0.0
s0 = 0.0
a0 = accelerometer[0]

#Matrice de covariance des erreurs
# a confirmer pour l'intiialisation de la matrice de covariance
Pk0 = np.matrix([[1, 0], [0, 1]])

for t in time:

    #Prediction d'etat
    #Xkp1 = A * Xkp0 + B * Uk + Wk

    #Xkp = [ X  ]
    #      [ .X ]

    #A = [ 1   dt]
    #    [ 0    1]

    #B = [ .5*dt²]
    #    [  dt ]

    #U = [ax0]  (acceleration)
    #Wk = 0

    Xkp0 = np.matrix([[p0], [s0]])
    Uk = np.matrix([a0])
    A = np.matrix([[1, dt], [0, 1]])
    B = np.matrix([[.5 * dt * dt], [dt]])
    Wk = 0
    Xkp1 = A * Xkp0 + B * Uk + Wk

    #Estimation de la matrice de covariance
    Pk1 = A * Pk0 * A.transpose()

    #Gain de Kalman
    #     Pkp * Ht
    #K = --------------
    #     H*Pkp*Ht + R

    #H = [1  0]
    #    [0  1]

    H = np.identity(2, dtype = float)
    #Error matrix
    positionError = 0.0
    speedError = 0.0
    R = np.matrix([[positionError, 0], [0, speedError]])
    K = (Pk1 * H.transpose()) / (H * Pk1 * H.transpose() + R)

    #Estimation de l'observation
    #Yk1 = C * Yk0 + Zk
    C = np.identity(2, dtype = float)




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

plt.tight_layout()
plt.show()