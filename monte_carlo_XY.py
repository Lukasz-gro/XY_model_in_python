import numpy as np
from numpy.random import rand


def model(n):
    return np.random.uniform(0, 2 * np.pi, (n, n))


def monte_carlo(config, temperature, n, steps):
    beta = 1 / temperature
    for i in range(steps):
        a = np.random.randint(0, n)
        b = np.random.randint(0, n)
        s = config[a, b]
        h1 = (-1) * (np.cos(s - config[(a + 1) % n, b]) + np.cos(s - config[(a - 1) % n, b]) + np.cos(
            s - config[a, (b + 1) % n]) + np.cos(s - config[a, (b - 1) % n]))
        delta = s + np.random.uniform(0, 2 * np.pi)
        h2 = (-1) * (np.cos(delta - config[(a + 1) % n, b]) + np.cos(delta - config[(a - 1) % n, b]) + np.cos(
            delta - config[a, (b + 1) % n]) + np.cos(delta - config[a, (b - 1) % n]))
        cost = h2 - h1
        if cost < 0:
            s = delta
        elif rand() < np.exp(-cost * beta):
            s = delta
        config[a, b] = s

        if config[a, b] > 2 * np.pi:
            config[a, b] -= 2 * np.pi
    return config


def calc_magnetism(config):

    n = config.shape[0]
    ox = np.sum(np.cos(config))
    oy = np.sum(np.sin(config))
    return(np.sqrt(ox**2 + oy**2))/(n**2)


def calc_energy(config):

    n = config.shape[0]
    energy = 0
    for k in range(len(config)):

        for m in range(len(config)):
            s = config[k, m]
            energy += np.cos(s-config[(k+1) % n, m]) + np.cos(s-config[(k-1) % n, m]) + np.cos(s-config[k, (m+1) % n]) \
                      + np.cos(s - config[k, (m-1) % n])
    return energy/2.


def susceptibility(m, temperature):
    return(np.mean(m*m) - (np.mean(m)) ** 2)/temperature


def heat(e, temperature):
    return(np.mean(e*e) - (np.mean(e)) ** 2) / (temperature ** 2)


def stiffness(config, temperature):
    length = config.shape[0]
    n = length ** 2
    beta = 1 / temperature
    part1 = []
    part2 = []
    u = [0.8, 0.6]
    for o in range(length):
        for p in range(length):
            a = config[o, p]
            b = config[(o + 1) % length, p]
            c = config[o, (p - 1) % length]
            d = config[(o - 1) % length, p]
            e = config[o, (p + 1) % length]

            a1 = np.cos(a - b) * (np.dot([1, 0], u) ** 2)
            a2 = np.cos(a - c) * (np.dot([0, -1], u) ** 2)
            a3 = np.cos(a - d) * (np.dot([-1, 0], u) ** 2)
            a4 = np.cos(a - e) * (np.dot([0, 1], u) ** 2)
            part1.append(np.mean(a1 + a2 + a3 + a4))

            b1 = (np.sin(b - a) * np.dot([1, 0], u)) ** 2
            b2 = (np.sin(c - a) * np.dot([0, -1], u)) ** 2
            b3 = (np.sin(d - a) * np.dot([-1, 0], u)) ** 2
            b4 = (np.sin(e - a) * np.dot([0, 1], u)) ** 2
            part2.append(np.mean(b1 + b2 + b3 + b4))
    return (1 / n) * np.sum(part1) - (1 / n) * beta * np.sum(part2)


# parameters of your model
model_size = 16
number_of_configs = 20000
dest = 'tmp'
new_XY = model(model_size)


Susceptibility = []
SpecificHeat = []
Magnetism = []
Energy = []
Stiffness = []

#range of temperature
Temps = np.arange(20, 201)*0.01
Temperature = Temps[::-1]


for t in Temperature:
    print(t)
    name = str(t)[:4]
    Mag = []
    Eng = []
    Stiff = []
    configs_to_file = []
    new_XY = monte_carlo(new_XY, t, new_XY.shape[0], 15000)
    print("Thermalisation done")

    for j in range(number_of_configs):
        if j%100 == 0:
            print("Configuration nr {}".format(j))
        new_XY = monte_carlo(new_XY, t, new_XY.shape[0], model_size**2)
        configs_to_file.append(new_XY)
        Mag.append(calc_magnetism(new_XY))
        Eng.append(calc_energy(new_XY))
        Stiff.append(stiffness(new_XY, t))

    configs = np.asarray(configs_to_file).reshape(number_of_configs, model_size**2)
    Stiff = np.asarray(Stiff).reshape(number_of_configs, 1)
    Mag = np.asarray(Mag).reshape(number_of_configs, 1)
    Eng = np.asarray(Eng).reshape(number_of_configs, 1)
    np.savetxt(dest + '/T=%s.txt' % (name), configs, fmt='%f', delimiter=' ')
    np.savetxt(dest + '/Stiff=%s.txt' % (name), Stiff, fmt='%f', delimiter=' ')
    np.savetxt(dest + '/Mag=%s.txt' % (name), Mag, fmt='%f', delimiter=' ')
    np.savetxt(dest + '/Eng=%s.txt' % (name), Eng, fmt='%f', delimiter=' ')

    Stiffness.append(np.mean(Stiff))

Stiffness = np.asarray(Stiffness).reshape(len(Temperature), 1)
np.savetxt(dest + '/AllStiff.txt', Stiffness, fmt='%f', delimiter=' ')
