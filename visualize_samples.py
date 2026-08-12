import numpy as np
import matplotlib.pyplot as plt

afib = np.load("samples/afib_demo.npy")
normal = np.load("samples/normal_demo.npy")

fig, ax = plt.subplots(2, 1, figsize=(14, 6))

ax[0].plot(normal)
ax[0].set_title("Normal ECG")

ax[1].plot(afib)
ax[1].set_title("AFib ECG")

plt.tight_layout()
plt.show()