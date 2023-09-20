import subprocess
import matplotlib.pyplot as plt
import numpy as np

# Define the range for n values
n_values = [2**i for i in range(10, 31)]

# Define a list to store the times for each n
times = []

# For each n value, run the task1 program and capture its output
for n in n_values:
    result = subprocess.run(["./task1", str(n)], capture_output=True, text=True)
    # The first line of the output should be the time taken
    time_taken = float(result.stdout.split("\n")[0])
    times.append(time_taken)
    print(f"n={n}, time={time_taken}ms")

# Now, plot the times
plt.figure(figsize=(10, 6))
plt.plot(n_values, times, marker='o', linestyle='-')
plt.xscale('log', base=2)  # Set x-axis to be logarithmic with base 2
plt.xlabel("n")
plt.ylabel("Time (ms)")
plt.title("Time taken by task1 algorithm as a function of n")
plt.grid(True, which="both", ls="--", c='0.65')
plt.tight_layout()
plt.savefig("task1.pdf")
plt.show()
