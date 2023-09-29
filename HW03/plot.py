import numpy as np
import matplotlib.pyplot as plt

# Read data
with open("results512.txt", "r") as file:
    lines512 = file.readlines()

with open("results16.txt", "r") as file:
    lines16 = file.readlines()


# Extract data
n_values = [2**exp for exp in range(10, 30)]
times512 = [float(lines512[i]) for i in range(0, len(lines512), 3)]
times16 = [float(lines16[i]) for i in range(0, len(lines16), 3)]

# Plot
plt.figure(figsize=(10,6))

# plt.subplot(2, 1, 1)
plt.plot(n_values, times512, 'o-', label='Time (ms) (512 threads/block)')
plt.plot(n_values, times16, 'o-', label='Time (ms) (16 threads/block)')
plt.xlabel("Array size")
plt.ylabel("Time (ms)")
plt.title("Performance of vscale kernel")
plt.grid(True)
plt.legend()s


plt.tight_layout()
plt.savefig("task3.pdf")
plt.show()
