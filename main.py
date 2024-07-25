import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# generating and normalizing data
def generate_data(num_samples):
    G = 6.67430e-11
    M = np.random.uniform(1e24, 1e30, num_samples)
    r = np.random.uniform(1e6, 1e9, (num_samples, 100))
    g = G * M[:, np.newaxis] / (r**2)
    return M, r, g

M_train, r_train, g_train = generate_data(1000)
M_mean, M_std = np.mean(M_train), np.std(M_train)
r_mean, r_std = np.mean(r_train), np.std(r_train)
g_mean, g_std = np.mean(g_train), np.std(g_train)

M_train_norm = (M_train - M_mean) / M_std
r_train_norm = (r_train - r_mean) / r_std
g_train_norm = (g_train - g_mean) / g_std

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

X_train = np.column_stack((np.repeat(M_train_norm, 100), r_train_norm.flatten()))
y_train = g_train_norm.flatten()

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

def create_circular_field(mass, radius, resolution=200):
    x = np.linspace(-radius, radius, resolution)
    y = np.linspace(-radius, radius, resolution)
    xx, yy = np.meshgrid(x, y)
    
    r = np.sqrt(xx**2 + yy**2)
    r_norm = (r - r_mean) / r_std
    mass_norm = (mass - M_mean) / M_std
    
    X_test = np.column_stack((np.repeat(mass_norm, resolution**2), r_norm.flatten()))
    g_pred_norm = model.predict(X_test)
    g_pred = g_pred_norm * g_std + g_mean
    
    return xx, yy, g_pred.reshape(resolution, resolution)

M_test = np.random.uniform(1e24, 1e30, 1)[0]
radius = 1e9
xx, yy, g_field = create_circular_field(M_test, radius)

plt.figure(figsize=(12, 10))
# using log for better viz
log_g_field = np.log10(g_field + 1)  # Adding 1 to avoid log(0)


contour = plt.contourf(xx, yy, log_g_field, levels=50, cmap='viridis')
plt.contour(xx, yy, log_g_field, levels=15, colors='white', alpha=0.3, linewidths=0.5)

plt.colorbar(contour, label='Log10(Gravitational field strength (m/s^2))')
plt.title(f'Gravitational Field for Mass {M_test:.2e} kg')
plt.xlabel('Distance (m)')
plt.ylabel('Distance (m)')
plt.axis('equal')

#not to scale
body_radius = radius / 20  # Arbitrary size for visualization
circle = plt.Circle((0, 0), body_radius, color='white', fill=False)
plt.gca().add_artist(circle)

plt.show()
