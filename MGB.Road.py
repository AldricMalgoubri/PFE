import pandas as pd  # Importation de la bibliothèque pandas pour la manipulation des données
import numpy as np  # Importation de numpy pour les calculs numériques
from scipy.interpolate import interp1d  # Importation de la fonction d'interpolation
import matplotlib.pyplot as plt # Importation de matplotlib pour la visualisation des données
from matplotlib.patches import Polygon # Importation de Polygon pour tracer des polygones
from scipy.interpolate import griddata # Importation de griddata pour l'interpolation sur une grille



# 1ère Partie: Interpollation des points appartenant à la source linéaire
data = pd.read_csv(r'C:\Users\HP\Desktop\Stage PFE\Excel\Coordonnées_IR.csv', sep=";") # Charger les données depuis le fichier CSV
data.columns=['X','Y'] # Renommer les colonnes en 'X' et 'Y'
x_points, y_points = data['X'].values, data['Y'].values # x_points, y_points sont défini par la colonne nommée 'X' et 'Y'
n_points = 1000 # Nombre de points à générer
x_interp = np.linspace(x_points.min(), x_points.max(), n_points) # Génération de nouveaux X interpolés
# Créer une fonction d'interpolation
try:
    interpolation_function = interp1d(x_points, y_points, kind='linear') # Interpolation linéaire
    y_interp = interpolation_function(x_interp) # Génération de nouveaux Y interpolés
except ValueError as e:
    print(f"Erreur lors de l'interpolation : {e}") # Afficher erreur en cas de problème lors de l'interpolation
    exit()
new_data = pd.DataFrame({'X': x_interp, 'Y': y_interp}) # Créer un DataFrame avec les nouveaux points interpolés
x_interp = x_interp. astype(int) # Convertir les X interpolés en entiers
y_interp = y_interp.astype(int) # Convertir les Y interpolés en entiers
data_int = pd.DataFrame({'X': x_interp, 'Y': y_interp}) # Créer un DataFrame avec les nouveaux points
data_int.to_csv (r'C:\Users\HP\Desktop\Stage PFE\Excel\Points_interpolés.csv',index=False,sep=";") # Exporter vers un fichier CSV
# Tracer les résultats
plt.figure(figsize=(16, 8)) # Définition de la taille de la figure
plt.plot(x_points, y_points, 'o', label="POINTS D'ORIGINE", markersize=5) # Tracé des points originaux
plt.plot(x_interp, y_interp, '-', label='POINTS INTERPOLE', linewidth=1) # Tracé de l'interpolation
plt.xlabel("Axe X", fontsize=12)  # Nom de l'axe X
plt.ylabel("Axe Y", fontsize=12)  # Nom de l'axe Y
plt.title("INFRASTRUSTURE ROUTIERE", fontsize=14)  # Ajuste la taille selon ton besoin
plt.legend(fontsize=10) # Legende
plt.grid(True) # Ajout d'une grille
plt.xlim(x_interp.min(), x_interp.max()) # Ajustement des limites pour bien entourer la route
plt.ylim(y_interp.min(), y_interp.max()) # Ajustement des limites pour bien entourer la route
plt.tick_params(axis='both', which='both', direction='in')  # Garde les ticks des deux axes
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)  # Ajuste les marges
plt.show() #Afficher l'afficher


# 2ème Partie: Transformation (translation & rotation) des points généré par interpolation et Affichage d'une grille de point de l'espace
df1 = data_int # Definition de df pour recupérer le resultat de la 1ère partie
x0, y0 = df1['X'][0], df1['Y'][0] # Définition de l'origine de notre système d'axe (Rabat)
x_trans, y_trans = df1['X'] - x0, df1['Y'] - y0 # Transformation des coordonnées par translation
# Fonction pour générer des points appartenant à l'espace dans une matrice
def generer_points_matrice(x_trans, y_trans):
    pas = 100 # Pas fixer pour la grille de points de la matrice (point a chaque 100 m)
    # Déterminer les limites de la matrice
    xmin, xmax = np.floor(x_trans.min() / pas) * pas, np.ceil(x_trans.max() / pas) * pas
    ymin, ymax = np.floor(y_trans.min() / pas) * pas, np.ceil(y_trans.max() / pas) * pas
    # Création de la grille de points
    x = np.arange(xmin, xmax + pas, pas)
    y = np.arange(ymin, ymax + pas, pas)
    X, Y = np.meshgrid(x, y)
    points_matrice = np.column_stack((X.flatten(), Y.flatten()))
    return points_matrice
# Génération des points de la matrice excluant la route
points_matrice = generer_points_matrice(x_trans, y_trans)
x_matrice, y_matrice = zip(*points_matrice)
x_matrice, y_matrice = np.array(x_matrice).astype(int), np.array(y_matrice).astype(int)
# Tracer les résultats
plt.figure(figsize=(16, 8)) # Définition de la taille de la figure
plt.scatter(x_trans, y_trans, c='blue', label="INFRASTRUCTURE ROUTIERE", s=5) # Afficher les points de l'autoroute
plt.scatter(x_matrice, y_matrice, c='red', label="POINTS DE L'ESPACE", s=0.5) # Afficher les points de l'espace
plt.axhline(0, color='black', linewidth=0.8) # Afficher l'axe horizontale (Axe X)
plt.axvline(0, color='black', linewidth=0.8) # Afficher l'axe verticale (Axe Y)
plt.xlabel('Axe X', fontsize=12) # Ajout de la mention Axe Y
plt.ylabel('Axe Y', fontsize=12) # Ajout de la mention Axe Y
plt.title("INFRASTRUCTURE ROUTIERE ET POINTS DE L'ESPACE", fontsize=14) # Titre de la carte
plt.legend(fontsize=10) # Légende
plt.grid(True) # Grille pour faciliter la lecture
# Ajustement des limites des axes pour éviter le vide
plt.xlim(x_trans.min(), x_trans.max()) # Ajustement des limites pour bien entourer la route
plt.ylim(y_trans.min(), y_trans.max()) # Ajustement des limites pour bien entourer la route
plt.tick_params(axis='both', which='both', direction='in')  # Garde les ticks des deux axes
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)  # Ajuste les marges
plt.show() #Afficher l'afficher


# 3ème Partie: Calculs de la concentration
def rotate_point(x, y, angle):
    cos_theta, sin_theta = np.cos(angle), np.sin(angle)
    x_rot = cos_theta * x + sin_theta * y
    y_rot = -sin_theta * x + cos_theta * y
    return x_rot, y_rot
# Fonction pour Calcul de Sigma avec la méthode de Briggs
def calculer_sigma(classe, xd, zone):
    xd = np.maximum(np.abs(xd), 1e-6)  # Éviter des valeurs nulles ou négatives
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
    # Vérification des entrées
    if zone not in valeurs_sigma:
        raise ValueError(f"Zone invalide : '{zone}'. Choisir 'rural' ou 'urbain'.")
    if classe not in valeurs_sigma[zone]:
        raise ValueError(f"Classe de stabilité invalide : '{classe}'. Doit être entre 1 et 6.")
    return valeurs_sigma[zone][classe]
# Fonction pour le calcul de la concentration avec le modèle de GAUSS
def concentration_gauss(xd, yd, Q, u, h, sig_y, sig_z):
    sig_y = np.maximum(sig_y, 1e-6) # Éviter division par zéro
    sig_z = np.maximum(sig_z, 1e-6) # Éviter division par zéro
    exp_term = np.exp(-0.5 * ((yd / sig_y) ** 2 + (h / sig_z) ** 2)) # Expression de la partie exponentiel dans le modèle de Gauss
    concentration = (Q / (np.pi * u * sig_y * sig_z)) * exp_term # Modèle de GAUSS
    return concentration 
# Fonction pour le Calcule la concentration des polluants en chaque point de l’espace en fonction du modèle de dispersion gaussien.
def calculate_concentration(x_matrice, y_matrice, x_trans, y_trans, wind_direction, Q, u, h, classe, zone):
    wind_angle = np.radians(wind_direction) # Conversion de l'angle du vent en radians
    road_rotated = np.array([rotate_point(rx, ry, wind_angle) for rx, ry in zip(x_trans, y_trans)]) # Pré-calcul de la rotation des points de la route 
    buffer_distance = 100 # Fixer la zone tampon de 100m selon la méthode de Briggs
    concentrations = []
    for px, py in zip(x_matrice, y_matrice):
        px_rot, py_rot = rotate_point(px, py, wind_angle) # Rotation du point d’espace pour l’aligner avec le vent
        total_concentration = 0
        distances = np.sqrt((road_rotated[:, 0] - px_rot)**2 + (road_rotated[:, 1] - py_rot)**2) # Vérifier si le point est sur la route ou dans la zone tampon (100m de chaque côté)
        if np.min(distances) <= buffer_distance:
            total_concentration = 0  # Le point est sur la route ou dans la zone tampon, concentration = 0
        else:
            for rx_rot, ry_rot in road_rotated:
                if rx_rot <= px_rot:  # Vérification des contributeurs
                    xd = px_rot - rx_rot # xd est la coordonée x utilisé dans le calcul
                    yd = py_rot - ry_rot #yd est la coordonnée y utilisé dans le calcul
                    sigma_y, sigma_z = calculer_sigma(classe, xd, zone) # Calcul des écart types
                    contribution = concentration_gauss(xd, yd, Q, u, h, sigma_y, sigma_z) # Calcul de la contribution de chaque point contributeur
                    total_concentration += np.sum(contribution) # Somme des contributions
        concentrations.append((px, py, total_concentration)) # Concentration égale à le somme des contributions
    return pd.DataFrame(concentrations, columns=['X', 'Y', 'Concentration'])
# ---- Données d'entrée ----
i=10**6
wind_direction = 180
Q, u, h, classe, zone = (500 * 10**-3)*i , 2, 0, 3, 'rural' # Q (g/s), u (m/s), h (m)
result = calculate_concentration(x_matrice, y_matrice, x_trans, y_trans, wind_direction, Q, u, h, classe, zone)
print(result.head())
result.to_csv (r'C:\Users\HP\Desktop\Stage PFE\Excel\Concentration_IR.csv',index=False,sep=";")


# 4ème Partie : Affichage de la carte des concentrations
pas = 200  # Pas fixer pour la grille de concentration (point a chaque 200 m)
# Définition des bornes de la grille
x_min, x_max = x_matrice.min(), x_matrice.max()
y_min, y_max = y_matrice.min(), y_matrice.max()
# Génération des points avec un pas constant
xi = np.arange(x_min, x_max + pas, pas) # +pas pour inclure la borne max suibant x
yi = np.arange(y_min, y_max + pas, pas) # +pas pour inclure la borne max suibant y
xi, yi = np.meshgrid(xi, yi) # Création de la grille
zi = griddata((x_matrice, y_matrice), result["Concentration"], (xi, yi), method='linear') # Interpolation linéaire des concentrations sur la grille
# Définition de la zone tampon autour de la route
buffer_distance = 100 # Fixer Distance de la zone tampon en mètres
# Calcul des vecteurs perpendiculaires à l'Infrastructure Routière
dx, dy = np.gradient(x_trans), np.gradient(y_trans)
norm = np.sqrt(dx**2 + dy**2)
dx, dy = dx / norm, dy / norm
x_upper, y_upper = x_trans + buffer_distance * dy, y_trans - buffer_distance * dx
x_lower, y_lower = x_trans - buffer_distance * dy, y_trans + buffer_distance * dx
# Création du polygone représentant la zone tampon
coords = np.vstack([np.column_stack([x_upper, y_upper]), np.column_stack([x_lower[::-1], y_lower[::-1]])])
polygon = Polygon(coords, edgecolor='yellow', facecolor='none', linewidth=1, label="Zone Tampon")
# Création du tracé
plt.figure(figsize=(16, 8))  # Taille de la figure
plt.contourf(xi, yi, zi, levels=100, cmap='jet', alpha=1) # Affichage des concentrations
plt.colorbar(label="Concentration en μg/m³") # Affichage de la barre de concentration
plt.scatter(x_trans, y_trans, color='purple', marker='o', s=5, label="INFRASTRUCTURE ROUTIERE") # Ajouter les points de la route
plt.gca().add_patch(polygon) # Ajouter le polygone de la zone tampon
# Ajouter une flèche pour la direction du vent
angle_rad = np.radians(wind_direction) # Direction du vent
flx, fly = x_trans.mean(), y_trans.mean() # Position de départ de la flèche
lf = (x_trans.max() - x_trans.min()) * 0.08 # Longueur de la flèche
flx_a, fly_a = flx + lf * np.cos(angle_rad), fly + lf * np.sin(angle_rad) # Flèche dynamique
# Annotation 'Vent'
plt.annotate("Vent",
             xy=(flx_a, fly_a),
             xytext=(flx, fly),
             arrowprops=dict(facecolor='orange', edgecolor='orange', arrowstyle='-|>', lw=3),
             fontsize=12, color='orange')
plt.xlabel("Axe X", fontsize=12)  # Nom de l'axe X
plt.ylabel("Axe Y", fontsize=12)  # Nom de l'axe Y
plt.title("CARTE DES CONCENTRATIONS DES POLLUANTS", fontsize=14)  # Ajuste la taille selon ton besoin
plt.legend(fontsize=10) #Affichage de la legende
plt.xlim(x_trans.min(), x_trans.max()) # Ajustement des limites pour bien entourer la route
plt.ylim(y_trans.min(), y_trans.max()) # Ajustement des limites pour bien entourer la route
plt.tick_params(axis='both', which='both', direction='in')  # Garde les ticks des deux axes
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)  # Ajuste les marges
plt.show() #Afficher l'afficher