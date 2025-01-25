import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Chargement du fichier CSV
df = pd.read_csv(r'C:\Users\HP\Desktop\Stage PFE\Excel\points_interpolés.csv', sep=";")

# Transformation des coordonnées
x_origine, y_origine = df['X'][0], df['Y'][0]
x_trans, y_trans = df['X'] - x_origine, df['Y'] - y_origine

# Fonction pour effectuer la rotation
def rotation(x, y, angle_degres):
    angle_radians = np.radians(angle_degres)
    x_rot = x * np.cos(angle_radians) - y * np.sin(angle_radians)
    y_rot = x * np.sin(angle_radians) + y * np.cos(angle_radians)
    return x_rot, y_rot

# Exemple : Direction du vent à 90 degrés
direction_vent = 90
x_final, y_final = rotation(x_trans, y_trans, direction_vent)

# Mise à jour du DataFrame avec les nouvelles coordonnées
df['X'], df['Y'] = x_final, y_final

# Sauvegarder le nouveau fichier CSV
df.to_csv(r'C:\Users\HP\Desktop\Stage PFE\Excel\coordonnees_transformees.csv', index=False,sep=";")

# Fonction pour générer des points dans une matrice
def generer_points_matrice():

# Obtenir les limites pour la grille
    xmin, xmax = np.min(x_final), np.max(x_final)
    ymin, ymax = np.min(y_final), np.max(y_final)

# Définir les valeurs de x et y avec un pas
    pas=400
    x = np.arange(xmin, xmax + 1, pas)
    y = np.arange(ymin, ymax + 1, pas)
    X, Y = np.meshgrid(x, y)

# Retourner les points comme une liste
    points_matrice = list(zip(X.flatten(), Y.flatten()))
    return points_matrice

# Obtenir les points de la matrice
points_matrice = generer_points_matrice ()
x_matrice, y_matrice = zip(*points_matrice)


# Fonction pour visualiser les points
def afficher_points(x_route, y_route, x_matrice, y_matrice):
    plt.figure(figsize=(15, 10))

# Afficher les points de la route
    plt.scatter(x_route, y_route, c='blue', label="Points de l'Autoroute A1", s=4)

# Afficher les points de la matrice
    plt.scatter(x_matrice, y_matrice, c='red', label="Points de l'espace", s=2)

    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Visualisation des points de l'Autoroute A1 et de l'espace")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Afficher les points
afficher_points(x_final, y_final, x_matrice, y_matrice)

# Fonction pour identifier les points contributeurs
def contributeurs(x_route, y_route, x_espace, y_espace):
    contributeurs = {}
    for x_espace, y_espace in zip(x_espace, y_espace):
        contributeurs[(x_espace, y_espace)] = [
            (x_route, y_route)
            for x_route, y_route in zip(x_route, y_route)
            if x_espace > x_route
        ]
    return contributeurs

# Identifier les contributeurs
contributeurs = contributeurs(x_final, y_final, x_matrice, y_matrice)

# Paramètres du modèle de Gauss
Q , u, h = 10, 10, 0 # Taux d'émission, Vitesse du vent, Hauteur effective de la source
zone, classe = 'urbain', 'A'

def calculer_sigma(x_diff, classe, zone):
    if zone == 'urbain':

        if classe == 'A' or 'B':
            sigma_y = 0.32 *x_diff* (1 + 0.0004 * x_diff)**(-1/2)  # en mètres
            sigma_z = 0.24 *x_diff* (1 + 0.001 * x_diff) **(-1/2)  # en mètres
        elif classe == 'C':
            sigma_y = 0.22 *x_diff* (1 + 0.0004 * x_diff)**(-1/2)
            sigma_z = 0.20 *x_diff
        elif classe == 'D':
            sigma_y = 0.16 *x_diff* (1 + 0.0004 * x_diff)**(-1/2)
            sigma_z = 0.14 *x_diff* (1 + 0.0003 * x_diff)**(-1/2)
        else: #classe == 'E' or 'F':
            sigma_y = 0.11 *x_diff* (1 + 0.0004 * x_diff)**(-1/2)
            sigma_z = 0.08 *x_diff* (1 + 0.0015 * x_diff)**(-1/2)

        return sigma_y, sigma_z
    else:  # rural
        if classe == 'A':
            sigma_y = 0.22 *x_diff* (1 + 0.0001 * x_diff)**(-1/2)
            sigma_z = 0.20 *x_diff
        elif classe == 'B':
            sigma_y = 0.16 *x_diff* (1 + 0.0001 * x_diff)**(-1/2)
            sigma_z = 0.12 *x_diff
        elif classe == 'C':
            sigma_y = 0.11 *x_diff* (1 + 0.0001 * x_diff)**(-1/2)
            sigma_z = 0.08 *x_diff* (1 + 0.0002 * x_diff)**(-1/2)
        elif classe == 'D':
            sigma_y = 0.08 *x_diff* (1 + 0.0001 * x_diff)**(-1/2)
            sigma_z = 0.06 *x_diff* (1 + 0.0015 * x_diff)**(-1/2)
        elif classe == 'E':
            sigma_y = 0.06 *x_diff* (1 + 0.0001 * x_diff)**(-1/2)
            sigma_z = 0.03 *x_diff* (1 + 0.0003 * x_diff)**(-1)
        else : #classe == 'F':
            sigma_y = 0.04 *x_diff* (1 + 0.0001 * x_diff)**(-1/2)
            sigma_z = 0.016*x_diff* (1 + 0.0001 * x_diff)**(-1)
        return sigma_y, sigma_z

# Vérifie que sigma_y et sigma_z ne sont pas nuls
    if sigma_y == 0 or sigma_z == 0:
        print(f"Erreur : valeurs de sigma invalides pour x_diff={x_diff}, classe={classe}, zone={zone}")
        sigma_y, sigma_z = 1e-10, 1e-10  # Valeur par défaut pour éviter la division par zéro
    return sigma_y, sigma_z

# Fonction pour calculer la concentration selon le modèle de Gauss
def concentration_gauss(x_diff, y_diff, Q, u, h, sigma_y, sigma_z):
    # Vérifie que sigma_y et sigma_z sont valides
    if sigma_y == 0 or sigma_z == 0:
        print(f"Erreur: sigma_y ou sigma_z est invalide: sigma_y={sigma_y}, sigma_z={sigma_z}")
        return 0  # Ou une valeur par défaut

    exp_term = np.exp(-0.5 * ((y_diff / sigma_y) ** 2 + (h / sigma_z) **2))

    # Vérifie si l'expression exponentielle est valide
    if np.isnan(exp_term):
        print(f"Erreur: valeur exponentielle non valide pour y_diff={y_diff}, sigma_y={sigma_y}, h={h}, sigma_z={sigma_z}")
        return 0  # Ou une valeur par défaut

    concentration = (Q / (2 * np.pi * u * sigma_y * sigma_z)) * exp_term

    # Vérifie si la concentration est valide
    if np.isnan(concentration):
        print(f"Erreur: concentration non valide pour x_diff={x_diff}, y_diff={y_diff}, Q={Q}, u={u}, sigma_y={sigma_y}, sigma_z={sigma_z}")

    return concentration

# Calculer la concentration pour chaque point de l'espace
def calculer_concentration(contributeurs, classe, zone):
    concentrations = {}
    for point_espace, points_contrib in contributeurs.items():
        x_espace, y_espace = point_espace
        concentration_total = 0
        for x_route, y_route in points_contrib:
            x_diff, y_diff = x_espace - x_route, y_espace - y_route
            
            # Calculer sigma_y et sigma_z
            sigma_y, sigma_z = calculer_sigma(x_diff, classe, zone)
            
            # Calculer la concentration
            concentration_total += concentration_gauss(x_diff, y_diff, Q, u, h, sigma_y, sigma_z)
        concentrations[point_espace] = (concentration_total)
    return concentrations

# Calculer les concentrations
concentrations = calculer_concentration(contributeurs, classe, zone)

# Afficher les résultats
for point_espace, concentration in concentrations.items():
    x, y = point_espace
    print(f"Concentration espace ({int(x)}, {int(y)}) : {float(concentration):.10f}")
