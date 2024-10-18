import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import constants as cons

from google.colab import files
uploaded = files.upload()

data = open('Data.txt', 'r')

wl = []
intensity = []

for line in data:
    rows = [i for i in line.split()]
    wl.append(float(rows[0]))
    intensity.append(float(rows[1]))

h = 6.626e-34  # Planck's constant (J·s)
c = 3e8       # Speed of light (m/s)
k = 1.381e-23  # Boltzmann constant (J/K)
temperature = 5800  # Temperature (K)
wien_constant = 2.897e-3  # Wien's displacement constant (m/K)

desired_peak = 2

# Wavelength range (in meters)
wavelengths = np.linspace(280, 1100, 941) * 1e-9

# Planck's law for spectral radiance
def planck_law(wavelength, temperature):
    return (8 * np.pi * h * c) / (wavelength**5) / (np.exp((h * c) / (wavelength * k * temperature)) - 1)

# Calculate spectral radiance for the Sun
spectral_irradiance = planck_law(wavelengths, temperature)

# Normalize to have the peak value equal to the desired value
scaling_factor = desired_peak / np.max(spectral_irradiance)
spectral_irradiance *= scaling_factor

peak_wavelength = 1e9*wien_constant / temperature
peak_wavelength

DWTcoeffs = pywt.wavedec(intensity, 'db4')
DWTcoeffs[-1] = np.zeros_like(DWTcoeffs[-1])
DWTcoeffs[-2] = np.zeros_like(DWTcoeffs[-2])
DWTcoeffs[-3] = np.zeros_like(DWTcoeffs[-3])
DWTcoeffs[-4] = np.zeros_like(DWTcoeffs[-4])
#DWTcoeffs[-5] = np.zeros_like(DWTcoeffs[-5])
#DWTcoeffs[-6] = np.zeros_like(DWTcoeffs[-6])
#DWTcoeffs[-7] = np.zeros_like(DWTcoeffs[-7])
#DWTcoeffs[-8] = np.zeros_like(DWTcoeffs[-8])
#DWTcoeffs[-9] = np.zeros_like(DWTcoeffs[-9])

filtered_data=pywt.waverec(DWTcoeffs,'db4')
filtered_data = filtered_data[:len(wl)]

plt.figure(figsize=(10, 6))
plt.plot(wavelengths * 1e9, spectral_irradiance, color='green', label='Theoretical Irradiance')
plt.plot(wl, intensity, color='red', label='Noisy Signal')
plt.plot(wl, filtered_data,  markerfacecolor='none',color='black', label='Denoised Signal')

plt.annotate(
    str(temperature)+' K BlackBody Spectrum',
    ha = 'center', va = 'bottom',
    xytext = (900,1.75),
    xy = (700,1.60),
    arrowprops = { 'facecolor' : 'black', 'shrink' : 0.05 }
)

plt.legend()
plt.title('Wavelet Denoising of Solar Linear Spectra (280nm - 1100nm)')
plt.xlabel(r'Wavelength (nm)')
plt.ylabel(r'Spectral Irradiance $(\frac{W}{m^2nm})$')
plt.axis([280, 1100, 0, 2.5])
plt.show()

plt.figure(figsize=(10, 12))  # Set the figure size for both subplots

# Subplot 1
plt.subplot(3, 1, 1)  # Three rows, one column, index 1
plt.plot(wavelengths * 1e9, spectral_irradiance, color='green', label='Theoretical Irradiance')
plt.plot(wl, intensity, color='red', label='Noisy Signal')
plt.legend()
plt.title('Wavelet Denoising of Solar Linear UVB and UVA Spectra (280nm - 400nm)')
plt.xlabel(r'Wavelength (nm)')
plt.ylabel(r'Spectral Irradiance $(\frac{W}{m^2nm})$')
plt.axis([280, 400, 0, 2])

# Subplot 2
plt.subplot(3, 1, 2)  # Three rows, one column, index 2
plt.plot(wl, intensity, color='red', label='Noisy Signal')
plt.plot(wl, filtered_data, markerfacecolor='none', color='black', label='Denoised Signal')
plt.legend()
plt.xlabel(r'Wavelength (nm)')
plt.ylabel(r'Spectral Irradiance $(\frac{W}{m^2nm})$')
plt.axis([280, 400, 0, 2])

# Subplot 2
plt.subplot(3, 1, 3)  # Three rows, one column, index 3
plt.plot(wavelengths * 1e9, spectral_irradiance, color='green', label='Real Data')
plt.plot(wl, filtered_data, markerfacecolor='none', color='black', label='Denoised Signal')
plt.legend()
plt.xlabel(r'Wavelength (nm)')
plt.ylabel(r'Spectral Irradiance $(\frac{W}{m^2nm})$')
plt.axis([280, 400, 0, 2])

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

plt.figure(figsize=(10, 12))  # Set the figure size for both subplots

# Subplot 1
plt.subplot(3, 1, 1)  # Three rows, one column, index 1
plt.plot(wavelengths * 1e9, spectral_irradiance, color='green', label='Theoretical Irradiance')
plt.plot(wl, intensity, color='red', label='Noisy Signal')
plt.legend()
plt.title('Wavelet Denoising of Solar Linear Visible Spectra (400nm - 700nm)')
plt.xlabel(r'Wavelength (nm)')
plt.ylabel(r'Spectral Irradiance $(\frac{W}{m^2nm})$')
plt.axis([400, 700, 1, 2.5])

# Subplot 2
plt.subplot(3, 1, 2)  # Three rows, one column, index 2
plt.plot(wl, intensity, color='red', label='Noisy Signal')
plt.plot(wl, filtered_data, markerfacecolor='none', color='black', label='Denoised Signal')
plt.legend()
plt.xlabel(r'Wavelength (nm)')
plt.ylabel(r'Spectral Irradiance $(\frac{W}{m^2nm})$')
plt.axis([400, 700, 1, 2.5])

# Subplot 2
plt.subplot(3, 1, 3)  # Three rows, one column, index 3
plt.plot(wavelengths * 1e9, spectral_irradiance, color='green', label='Theoretical Irradiance')
plt.plot(wl, filtered_data, markerfacecolor='none', color='black', label='Denoised Signal')
plt.legend()
plt.xlabel(r'Wavelength (nm)')
plt.ylabel(r'Spectral Intensity $(\frac{W}{m^2nm})$')
plt.axis([400, 700, 1, 2.5])

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

plt.figure(figsize=(10, 12))  # Set the figure size for both subplots

# Subplot 1
plt.subplot(3, 1, 1)  # Three rows, one column, index 1
plt.plot(wavelengths * 1e9, spectral_irradiance, color='green', label='Theoretical Irradiance')
plt.plot(wl, intensity, color='red', label='Noisy Signal')
plt.legend()
plt.title('Wavelet Denoising of Solar Linear Visible Spectra (400nm - 700nm)')
plt.xlabel(r'Wavelength (nm)')
plt.ylabel(r'Spectral Irradiance $(\frac{W}{m^2nm})$')
plt.axis([700, 1100, 0, 2.5])

# Subplot 2
plt.subplot(3, 1, 2)  # Three rows, one column, index 2
plt.plot(wl, intensity, color='red', label='Noisy Signal')
plt.plot(wl, filtered_data, markerfacecolor='none', color='black', label='Denoised Signal')
plt.legend()
plt.xlabel(r'Wavelength (nm)')
plt.ylabel(r'Spectral Irradiance $(\frac{W}{m^2nm})$')
plt.axis([700, 1100, 0, 2.5])

# Subplot 2
plt.subplot(3, 1, 3)  # Three rows, one column, index 3
plt.plot(wavelengths * 1e9, spectral_irradiance, color='green', label='Theoretical Irradiance')
plt.plot(wl, filtered_data, markerfacecolor='none', color='black', label='Denoised Signal')
plt.legend()
plt.xlabel(r'Wavelength (nm)')
plt.ylabel(r'Spectral Irradiance $(\frac{W}{m^2nm})$')
plt.axis([700, 1100, 0, 2.5])

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

def calculate_mse(original, denoised):
    mse = np.mean(np.square(np.subtract(original, denoised)))
    return mse

def calculate_rmse(original, denoised):
    mse = calculate_mse(original, denoised)
    rmse = np.sqrt(mse)
    return rmse

def calculate_psnr(original, denoised):
    max_pixel_value = 255.0  # Assuming pixel values are in the range [0, 255]
    mse_value = calculate_mse(original, denoised)
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse_value)
    return psnr

# Assuming 'original_signal' and 'denoised_signal' are your original and denoised signals as lists or arrays
mse_value = calculate_mse(spectral_irradiance, filtered_data)
rmse_value = calculate_rmse(spectral_irradiance, filtered_data)
psnr_value = calculate_psnr(spectral_irradiance, filtered_data)

print(f"MSE: {mse_value}")
print(f"RMSE: {rmse_value}")
print(f"PSNR: {psnr_value} dB")

