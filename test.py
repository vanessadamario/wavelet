import numpy as np
from wavelet.wavelet import wavelet
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1000)
f1 = 1;
f2 = 3;
signal =  np.sin(0.1 * np.pi * f1 * t)
signal += np.cos(0.5 * np.pi * f2 * t) * np.exp(-(t - 5)**2)
signal += np.cos(np.pi * f2 * t) * 12 * np.exp(-(t - 5)**2) +5*t

meyer_wave = wavelet(dyadic_exp=4.)
[coef, scale] = meyer_wave.cwt(signal, 10)
print('scales', scale)

plt.imshow(np.abs(coef), aspect='auto')
plt.colorbar()
plt.show()
plt.close()

recons_sig = meyer_wave.icwt(coef)
print(recons_sig)
plt.plot(t, signal, color='r')
plt.plot(t, np.real(recons_sig), '.')
plt.show()
plt.close()
