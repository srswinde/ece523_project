import pickle
import numpy as np
from matplotlib import pyplot as plt

with open('bestTrained.pkl', 'rb') as f:
      lShipData = pickle.load(f)
   

generation = np.zeros(len(lShipData))
fitness = np.zeros(len(lShipData))
for i in range(len(lShipData)):
    generation[i] = lShipData[i]['Generation']
    fitness[i] = lShipData[i]['score']

plt.plot(generation,fitness)
plt.title("Ship Fitness Evolution")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.show()