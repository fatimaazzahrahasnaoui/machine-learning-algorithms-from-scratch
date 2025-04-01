import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Définition de la fonction de coût (exemple : fonction quadratique simple)
def cost_function(x, y):
    return x**2 + y**2

# Gradient de la fonction (dérivées partielles)
def cost_gradient(x, y):
    dx = 2 * x  # Dérivée partielle par rapport à x
    dy = 2 * y  # Dérivée partielle par rapport à y
    return dx, dy

# Fonction de descente de gradient avec tracé des étapes
def gradient_descent_3d(start_x, start_y, learning_rate, epochs):
    x, y = start_x, start_y
    steps_x, steps_y, steps_cost = [x], [y], [cost_function(x, y)]
    
    for _ in range(epochs):
        dx, dy = cost_gradient(x, y)
        x -= learning_rate * dx
        y -= learning_rate * dy

        # Stocker les étapes pour le tracé
        steps_x.append(x)
        steps_y.append(y)
        steps_cost.append(cost_function(x, y))

    return steps_x, steps_y, steps_cost

# Paramètres de descente de gradient
start_x, start_y = 4.0, 3.0  # Point de départ
learning_rate = 0.1
epochs = 50

# Obtenir les étapes de la descente de gradient
steps_x, steps_y, steps_cost = gradient_descent_3d(start_x, start_y, learning_rate, epochs)

# Création d'une grille pour la surface de la fonction de coût
x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = cost_function(X, Y)

# Tracé en 3D de la surface de la fonction de coût et des étapes de la descente de gradient
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Ajouter les points des étapes de la descente de gradient
ax.plot(steps_x, steps_y, steps_cost, color='red', marker='o', markersize=5, label='Descente de Gradient')

# Configurer les étiquettes et le titre
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Coût (f(x, y))')
ax.set_title('Descente de Gradient sur la Surface de la Fonction de Coût')
plt.legend()
plt.show()
