import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, griddata
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# 1ère Partie: Interpolation des points appartenant à la source linéaire
data = pd.read_csv(r'C:\Users\HP\Desktop\Stage PFE\Excel\Coordonnées_164 points_autoroute.csv', sep=";")
data.columns = ['X', 'Y']
x_points, y_points = data['X'].values, data['Y'].values
n_points = 10
x_interp = np.linspace(x_points.min(), x_points.max(), n_points)

try:
    interpolation_function = interp1d(x_points, y_points, kind='linear')
    y_interp = interpolation_function(x_interp)
except ValueError as e:
    print(f"Erreur lors de l'interpolation : {e}")
    exit()

new_data = pd.DataFrame({'X': x_interp.astype(int), 'Y': y_interp.astype(int)})
new_data.to_csv(r'C:\Users\HP\Desktop\Stage PFE\Excel\Points_autoroute_interpolés.csv', index=False, sep=";")

# Tracer les résultats
plt.figure(figsize=(16, 8))
plt.plot(x_points, y_points, 'o', label='Données originales', markersize=5)
plt.plot(x_interp, y_interp, '-', label='Interpolation linéaire', linewidth=1)
plt.xlabel("Axe X", fontsize=12)
plt.ylabel("Axe Y", fontsize=12)
plt.title("INFRASTRUCTURE ROUTIERE", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.axis('equal')
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.show()

# 2ème Partie: Transformation et affichage des points
df1 = new_data
x0, y0 = df1['X'][0], df1['Y'][0]
x_trans, y_trans = df1['X'] - x0, df1['Y'] - y0

def generer_points_matrice(x_trans, y_trans):
    pas = 100
    xmin, xmax = np.floor(x_trans.min() / pas) * pas, np.ceil(x_trans.max() / pas) * pas
    ymin, ymax = np.floor(y_trans.min() / pas) * pas, np.ceil(y_trans.max() / pas) * pas
    x = np.arange(xmin, xmax + pas, pas)
    y = np.arange(ymin, ymax + pas, pas)
    X, Y = np.meshgrid(x, y)
    points_matrice = np.column_stack((X.flatten(), Y.flatten()))
    return points_matrice

points_matrice = generer_points_matrice(x_trans, y_trans)
x_matrice, y_matrice = zip(*points_matrice)
x_matrice, y_matrice = np.array(x_matrice).astype(int), np.array(y_matrice).astype(int)

plt.figure(figsize=(16, 8))
plt.scatter(x_trans, y_trans, c='blue', label="INFRASTRUCTURE ROUTIERE", s=5)
plt.scatter(x_matrice, y_matrice, c='red', label="POINTS DE L'ESPACE", s=0.5)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)

# Ajustement des limites de l'axe des X pour éviter l'espace inutile
plt.xlim(0, x_trans.max())

plt.xlabel('Axe X', fontsize=10)
plt.ylabel('Axe Y', fontsize=10)
plt.title("Visualisation des points de l'Autoroute A1 et de l'espace", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.axis('equal')
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.show()

# 3ème Partie: Calculs de la concentration
def rotate_point(x, y, angle):
    cos_theta, sin_theta = np.cos(angle), np.sin(angle)
    x_rot = cos_theta * x + sin_theta * y
    y_rot = -sin_theta * x + cos_theta * y
    return x_rot, y_rot

def calculer_sigma(classe, xd, zone):
    xd = np.maximum(np.abs(xd), 1e-6)
    valeurs_sigma = {
        'rural': {
            1: (0.22 * xd * (1 + 0.0001 * xd)**-0.5, 0.20 * xd),
            2: (0.16 * xd * (1 + 0.0001 * xd)**-0.5, 0.12 * xd),
            3: (0.11 * xd * (1 + 0.0001 * xd)**-0.5, 0.08 * xd * (1 + 0.0002 * xd)**-0.5),
            4: (0.08 * xd * (1 + 0.0001 * xd)**-0.5, 0.06 * xd * (1 + 0.0015 * xd)**-0.5),
            5: (0.06 * xd * (1 + 0.0001 * xd)**-0.5, 0.03 * xd * (1 + 0.0003 * xd)**-1),
            6: (0.04 * xd * (1 + 0.0001 * xd)**-0.5, 0.016 * xd * (1 + 0.0001 * xd)**-1),
        },
        'urbain': {
            1: (0.32 * xd * (1 + 0.0004 * xd)**-0.5, 0.24 * xd * (1 + 0.001 * xd)**0.5),
            2: (0.32 * xd * (1 + 0.0004 * xd)**-0.5, 0.24 * xd * (1 + 0.001 * xd)**0.5),
            3: (0.22 * xd * (1 + 0.0004 * xd)**-0.5, 0.2 * xd),
            4: (0.16 * xd * (1 + 0.0004 * xd)**-0.5, 0.14 * xd * (1 + 0.0003 * xd)**-0.5),
            5: (0.11 * xd * (1 + 0.0004 * xd)**-0.5, 0.08 * xd * (1 + 0.0015 * xd)**-0.5),
            6: (0.11 * xd * (1 + 0.0004 * xd)**-0.5, 0.08 * xd * (1 + 0.0015 * xd)**-0.5),
        }
    }
    if zone not in valeurs_sigma:
        raise ValueError(f"Zone invalide : '{zone}'. Choisir 'rural' ou 'urbain'.")
    if classe not in valeurs_sigma[zone]:
        raise ValueError(f"Classe de stabilité invalide : '{classe}'. Doit être entre 1 et 6.")
    return valeurs_sigma[zone][classe]

def concentration_gauss(xd, yd, Q, u, h, sig_y, sig_z):
    sig_y = np.maximum(sig_y, 1e-6)
    sig_z = np.maximum(sig_z, 1e-6)
    exp_term = np.exp(-0.5 * ((yd / sig_y) ** 2 + (h / sig_z) ** 2))
    concentration = (Q / (np.pi * u * sig_y * sig_z)) * exp_term
    return concentration

def calculate_concentration(x_matrice, y_matrice, x_trans, y_trans, wind_direction_deg, Q, u, h, classe, zone):
    wind_angle = np.radians(wind_direction_deg)
    road_rotated = np.array([rotate_point(rx, ry, wind_angle) for rx, ry in zip(x_trans, y_trans)])
    buffer_distance = 100
    concentrations = []
    for px, py in zip(x_matrice, y_matrice):
        px_rot, py_rot = rotate_point(px, py, wind_angle)
        total_concentration = 0
        distances = np.sqrt((road_rotated[:, 0] - px_rot)**2 + (road_rotated[:, 1] - py_rot)**2)
        if np.min(distances) <= buffer_distance:
            total_concentration = 0
        else:
            for rx_rot, ry_rot in road_rotated:
                if rx_rot <= px_rot:
                    xd = px_rot - rx_rot
                    yd = py_rot - ry_rot
                    sigma_y, sigma_z = calculer_sigma(classe, xd, zone)
                    contribution = concentration_gauss(xd, yd, Q, u, h, sigma_y, sigma_z)
                    total_concentration += np.sum(contribution)
        concentrations.append((px, py, total_concentration))
    return pd.DataFrame(concentrations, columns=['X', 'Y', 'Concentration'])

i = 10**6
wind_direction_deg = 180
Q, u, h, classe, zone = (500 * 10**-3) * i, 2, 0, 3, 'rural'
result = calculate_concentration(x_matrice, y_matrice, x_trans, y_trans, wind_direction_deg, Q, u, h, classe, zone)
print(result.head())
result.to_csv(r'C:\Users\HP\Desktop\Stage PFE\Excel\Concentration.csv', index=False, sep=";")

# 4ème Partie : Affichage de la carte des concentrations
step = 200
x_min, x_max = x_matrice.min(), x_matrice.max()
y_min, y_max = y_matrice.min(), y_matrice.max()
xi = np.arange(x_min, x_max + step, step)
yi = np.arange(y_min, y_max + step, step)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x_matrice, y_matrice), result["Concentration"], (xi, yi), method='linear')

buffer_distance = 100
dx, dy = np.gradient(x_trans), np.gradient(y_trans)
norm = np.sqrt(dx**2 + dy**2)
dx, dy = dx / norm, dy / norm
x_upper, y_upper = x_trans + buffer_distance * dy, y_trans - buffer_distance * dx
x_lower, y_lower = x_trans - buffer_distance * dy, y_trans + buffer_distance * dx
coords = np.vstack([np.column_stack([x_upper, y_upper]), np.column_stack([x_lower[::-1], y_lower[::-1]])])
polygon = Polygon(coords, edgecolor='yellow', facecolor='none', linewidth=1, label="Zone Tampon")

plt.figure(figsize=(16, 8))
plt.contourf(xi, yi, zi, levels=100, cmap='jet', alpha=1)
plt.colorbar(label="Concentration en μg/m³")
plt.scatter(x_trans, y_trans, color='purple', marker='o', s=5, label="INFRASTRUCTURE ROUTIERE")
plt.gca().add_patch(polygon)

angle_rad = np.radians(wind_direction_deg)
flx, fly = x_trans.mean(), y_trans.mean()
lf = (x_trans.max() - x_trans.min()) * 0.08
flx_a, fly_a = flx + lf * np.cos(angle_rad), fly + lf * np.sin(angle_rad)

plt.annotate("Vent", xy=(flx_a, fly_a), xytext=(flx, fly),
             arrowprops=dict(facecolor='orange', edgecolor='orange', arrowstyle='-|>', lw=3),
             fontsize=12, color='orange')

plt.xlabel("Axe X", fontsize=12)
plt.ylabel("Axe Y", fontsize=12)
plt.title("CARTE DES CONCENTRATIONS DES POLLUANTS", fontsize=14)
plt.legend(fontsize=10)
plt.axis('equal')
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.show()
