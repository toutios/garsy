
import matplotlib.pyplot as plt

with open("../reference/error_plot.txt", 'r') as file:
    ints = [int(a) for a in file.readlines()]
    plt.plot(ints)
    plt.show()
