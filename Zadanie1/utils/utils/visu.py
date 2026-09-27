import matplotlib.pyplot as plt 
import pandas as pd 

filename2 = "C:\\Codes\\DP\\dataset\\terrains\\wood_100_6.csv"
filename = "C:\\Codes\\DP\\dataset\\terrains\\concrete_100_6.csv"
df = pd.read_csv(filename,skiprows=1)

t = df.iloc[:,0]
ax = df.iloc[:,1]
ay = df.iloc[:,2]
az = df.iloc[:,3]

gx = df.iloc[:,4]
gy = df.iloc[:,5]
gz = df.iloc[:,6]

mx = df.iloc[:,7]
my = df.iloc[:,8]
mz = df.iloc[:,9]

t = t - 1

fig, axs = plt.subplots(3,3)
axs[0,0].plot(t,ax,label="bet.")
axs[0,0].set_title("Accel")
axs[0,0].set_ylabel("X [m/s^2]")
axs[1,0].plot(t,ay)
axs[1,0].set_ylabel("Y [m/s^2]")
axs[2,0].plot(t,az)
axs[2,0].set_ylabel("Z [m/s^2]")
axs[2,0].set_xlabel("t [s]")

axs[0,1].plot(t,gx)
axs[0,1].set_ylabel("X [°/s]")
axs[0,1].set_title("Gyro")
axs[1,1].plot(t,gy)
axs[1,1].set_ylabel("Y [°/s]")
axs[2,1].plot(t,gz)
axs[2,1].set_xlabel("t [s]")
axs[2,1].set_ylabel("Z [°/s]")

axs[0,2].plot(t,mx)
axs[0,2].set_ylabel("X [µT]")
axs[0,2].set_title("Mag")
axs[1,2].plot(t,my)
axs[1,2].set_ylabel("Y [µT]")
axs[2,2].plot(t,mz)
axs[2,2].set_ylabel("Z [µT]")
axs[2,2].set_xlabel("t [s]")

df = pd.read_csv(filename2,skiprows=1)

t = df.iloc[:,0]
ax = df.iloc[:,1]
ay = df.iloc[:,2]
az = df.iloc[:,3]

gx = df.iloc[:,4]
gy = df.iloc[:,5]
gz = df.iloc[:,6]

mx = df.iloc[:,7]
my = df.iloc[:,8]
mz = df.iloc[:,9]

t = t - 1

axs[0,0].plot(t,ax,label = "park.")
axs[1,0].plot(t,ay)
axs[2,0].plot(t,az)

axs[0,1].plot(t,gx)
axs[1,1].plot(t,gy)
axs[2,1].plot(t,gz)

axs[0,2].plot(t,mx)
axs[1,2].plot(t,my)
axs[2,2].plot(t,mz)

handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2)

for i in range(3):
    for j in range(3):
        axs[i,j].set_xlim([0,2])

plt.subplots_adjust(top=0.88)  # Adjust to make space for the global legend
plt.show()