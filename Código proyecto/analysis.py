import pandas as pd
import numpy as np

#Cargamos primeramente el dataset original llamado: Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv del Instituto Canadiense de Ciberseguridad.
print("=== Cargando dataset original ===")
df = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
print(f"Tamaño original del dataset: {len(df)} muestras")
print()
print(df.info())

#Reducimos a 50000 muestras debido a que será mucho más fácil de procesar los datos con los algoritmos.
df = df.head(50000)
print(f"\nDataset reducido a {len(df)} muestras")

#Limpiamos los nombres de las columnas (eliminar espacios al inicio y final).
df.columns = df.columns.str.strip()

#Mostramos la distribución de clases.
print("\nDistribución de clases en el dataset reducido:")
print(df['Label'].value_counts())

#Realizamos una información básica del dataset.
print("\n=== Información del Dataset ===")
print(df.info())

#Verificamos los valores nulos e infinitos antes de la limpieza.
print("\n=== Valores problemáticos antes de la limpieza ===")
print("Valores nulos:", df.isnull().sum().sum())
print("Valores infinitos:", df.isin([np.inf, -np.inf]).sum().sum())

#Vamos a comenzar a realizar la limpieza de datos:
#Primero reemplazamos infinitos con NaN
df_limpio = df.replace([np.inf, -np.inf], np.nan)

#Calculamos las estadísticas básicas solo para columnas numéricas
columnas_numericas = df_limpio.select_dtypes(include=[np.number]).columns
stats = df_limpio[columnas_numericas].describe()
print("\n=== Estadísticas antes de la limpieza (solo columnas numéricas) ===")
print(stats)

#Eliminamos los valores nulos.
df_limpio = df_limpio.dropna()

#Verificamos además de los valores nulos, los valores duplicados.
duplicados = df_limpio.duplicated().sum()
print(f"\nFilas duplicadas encontradas: {duplicados}")
df_limpio = df_limpio.drop_duplicates()

#Mostramos el tamaño final del dataset.
print("\n=== Tamaño Original vs Tamaño Limpio ===")
print(f"Tamaño original: {df.shape}")
print(f"Tamaño después de limpieza: {df_limpio.shape}")

#Realizamos un estudio de las features seleccionadas.
features_seleccionadas = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Label'
]

#Verificamos si las columnas existen o no.
print("\n=== Verificación de columnas ===")
for feature in features_seleccionadas:
    if feature in df_limpio.columns:
        print(f"Columna '{feature}' encontrada")
    else:
        print(f"Columna '{feature}' NO encontrada")

df_features = df_limpio[features_seleccionadas]

#Creamos una copia explícita para evitar SettingWithCopyWarning (error que me daba cuando ejecutaba el código las primeras veces).
df_features = df_limpio[features_seleccionadas].copy()

#Calculamos la asimetría de tráfico (ya que será una nueva feature que añadamos a nuestras features).
print("\n=== Calculando asimetría de tráfico ===")
df_features['asimetria_de_trafico'] = (
    np.abs(df_features['Total Fwd Packets'] - df_features['Total Backward Packets']) /
    np.abs(df_features['Total Fwd Packets'] + df_features['Total Backward Packets'])
)

#Mostramos las estadísticas de la nueva feature.
print("\n=== Estadísticas de asimetría de tráfico ===")
print(df_features['asimetria_de_trafico'].describe())

#Mostramos la distribución de asimetría por tipo de tráfico.
print("\n=== Asimetría promedio por tipo de tráfico ===")
print(df_features.groupby('Label')['asimetria_de_trafico'].mean())

#Realizamos las estadísticas de las features numéricas seleccionadas.
features_numericas = [f for f in df_features.columns if f != 'Label']
print("\n=== Estadísticas de Features Numéricas Seleccionadas ===")
print(df_features[features_numericas].describe())

#Realizamos la distribución de la variable objetivo.
print("\n=== Distribución de Labels ===")
print(df_features['Label'].value_counts())
print("\nPorcentaje de cada clase:")
print(df_features['Label'].value_counts(normalize=True) * 100)

#Guardamos por último el dataset limpio.
df_limpio.to_csv('dataset_limpio.csv', index=False)

#Guardamos el dataset con las features seleccionadas y la nueva feature de asimetría.
df_features.to_csv('features_seleccionadas.csv', index=False)