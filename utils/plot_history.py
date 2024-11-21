import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


history_file = "train_log_130000.txt"

f = open(history_file, 'r')

history = []

steps = []
gen_losses = []
disc_losses = []

for line in f.readlines():
    line = line.strip()
    line = line.replace(':', '')
    line = line.replace(',', '')
    res = line.split(" ")
    step = int(res[3])
    gen_loss = float(res[7])
    disc_loss = float(res[10])
    
    gen_losses.append(gen_loss)
    disc_losses.append(disc_loss)

gen_losses = gen_losses[1:601]
disc_losses = disc_losses[1:601]

# Example data
x = range(0, 120000, 200)  # Assuming both lists have the same length

# Set seaborn style
# Create the figure and the first axis
fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot List A on the primary y-axis
color_a = sns.color_palette("tab10")[0]
ax1.plot(x, gen_losses, linestyle='-', color=color_a, label='Generator')
ax1.set_ylabel('Generator loss', color=color_a, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color_a)
ax1.spines['left'].set_color(color_a)
ax1.spines['left'].set_linewidth(1.2)

# Create the secondary y-axis
ax2 = ax1.twinx()

# Plot List B on the secondary y-axis
color_b = sns.color_palette("tab10")[3]
ax2.plot(x, disc_losses, linestyle='-', color=color_b, label='Discriminator')
ax2.set_ylabel('Discriminator loss', color=color_b, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color_b)
ax2.spines['right'].set_color(color_b)
ax2.spines['right'].set_linewidth(1.2)

# Titles and Legends
ax1.set_xlabel('Steps', fontsize=12)
plt.title('CycleGAN loss', fontsize=14, fontweight='bold')
fig.tight_layout()

# Show the plot
plt.grid(True)
plt.show()
