import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

path = r'C:\Codes\DP\fft_dat'

num = 10

sensor = 'GYR'
plt.figure(1)
plt.subplot(321)
plt.title('Concrete')
df = pd.read_csv(f'{path}\\{sensor}\\concrete\\{num}.csv')
mag = df.iloc[:,1].to_numpy()
plt.plot(mag)
if sensor == 'MAG':
    plt.ylim([0,0.2])

elif sensor == 'ACC':
    plt.ylim([0,50])

plt.subplot(322)
plt.title('Grass')
df = pd.read_csv(f'{path}\\{sensor}\\grass\\{num}.csv')
mag = df.iloc[:,1].to_numpy()
plt.plot(mag)
if sensor == 'MAG':
    plt.ylim([0,0.2])

elif sensor == 'ACC':
    plt.ylim([0,50])



plt.subplot(323)
plt.title('Paving stone')
df = pd.read_csv(f'{path}\\{sensor}\\paving stone\\{num}.csv')
mag = df.iloc[:,1].to_numpy()
plt.plot(mag)
if sensor == 'MAG':
    plt.ylim([0,0.2])
elif sensor == 'ACC':
    plt.ylim([0,50])

plt.subplot(324)
plt.title('Pebbles')
df = pd.read_csv(f'{path}\\{sensor}\\pebbles\\{num}.csv')
mag = df.iloc[:,1].to_numpy()
plt.plot(mag)
if sensor == 'MAG':
    plt.ylim([0,0.2])
elif sensor == 'ACC':
    plt.ylim([0,50])

plt.subplot(325)
plt.title('Sand')
df = pd.read_csv(f'{path}\\{sensor}\\sand\\{num}.csv')
mag = df.iloc[:,1].to_numpy()
plt.plot(mag)
if sensor == 'MAG':
    plt.ylim([0,0.2])
elif sensor == 'ACC':
    plt.ylim([0,50])

plt.subplot(326)
plt.title('Synthetic running track')
df = pd.read_csv(f'{path}\\{sensor}\\synthetic running track\\{num}.csv')
mag = df.iloc[:,1].to_numpy()
plt.plot(mag)
if sensor == 'MAG':
    plt.ylim([0,0.2])
elif sensor == 'ACC':
    plt.ylim([0,50])
plt.show()