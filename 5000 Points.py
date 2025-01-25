import pandas as pd
import numpy as np

# Charger les données depuis le fichier CSV
data = pd.read_csv(r'C:\Users\HP\Desktop\Stage PFE\Excel\Coordonnées des 164 points de la route.csv', sep=";")

# Afficher les colonnes disponibles pour vérifier les noms
print(data.head())
print(data.columns)
data.columns=['X','Y']

# Vérifier la longueur des colonnes
print(len(data['X']), len(data['Y']))

# les colonnes sont nommées 'X' et 'Y'
x_points = data['X'].values
y_points = data['Y'].values

# Nombre de points à générer
n_points = 5000

# Créer un tableau de nouveaux X entre le minimum et le maximum de x_points
x_new = np.linspace(x_points.min(), x_points.max(), n_points)

# Interpolation linéaire des Y nouveaux
y_new = np.linspace(y_points.min(), y_points.max(), n_points)

# Convertir les résultats en entiers
x_new = x_new. astype(int)
y_new = y_new.astype(int)

# Créer un DataFrame avec les nouveaux points
new_data = pd.DataFrame({'X': x_new, 'Y': y_new})

# Exporter vers un fichier CSV
new_data.to_csv (r'C:\Users\HP\Desktop\Stage PFE\Excel\points_interpolés.csv',index=False,sep=";")