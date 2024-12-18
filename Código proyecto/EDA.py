import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def matriz_de_correlacion(df):
    
    #Comenzamos analizando y visualizando las correlaciones entre características.
    
    #Obtenemos las columnas de características.
    columnas_features = [col for col in df.columns if col != 'Label']
    correlacion = df[columnas_features].corr()
    
    #Visualizamos la matriz de correlación.
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlacion, annot=True, fmt=".2f", cmap="coolwarm", 
                cbar=True, square=True)
    plt.title("Matriz de Correlación entre Features")
    plt.tight_layout()
    plt.show()
    
    #Identificamos y reportamos las correlaciones altas.
    correlaciones_altas = []
    for i in range(len(correlacion.columns)):
        for j in range(i+1, len(correlacion.columns)):
            if abs(correlacion.iloc[i,j]) > 0.8:
                correlaciones_altas.append((
                    correlacion.columns[i], 
                    correlacion.columns[j], 
                    correlacion.iloc[i,j]
                ))
    
    if correlaciones_altas:
        print("\nCaracterísticas altamente correlacionadas (>0.8):")
        for feat1, feat2, corr in correlaciones_altas:
            print(f"{feat1} - {feat2}: {corr:.3f}")
    else:
        print("\nNo se encontraron características altamente correlacionadas (>0.8)")
    
    #Realizamos la correlación con la variable objetivo.
    correlacion_objetivo = []
    for col in columnas_features:
        corr = abs(pd.get_dummies(df['Label'])[df['Label'].unique()[0]].corr(df[col]))
        if corr > 0.5:  # Umbral más bajo para la variable objetivo.
            correlacion_objetivo.append((col, corr))
    
    if correlacion_objetivo:
        print("\nCaracterísticas más correlacionadas con la variable objetivo (>0.5):")
        for feat, corr in correlacion_objetivo:
            print(f"{feat}: {corr:.3f}")
    
    #Añadimos análisis de features más discriminativas.
    print("\nFeatures más discriminativas para detectar DDoS:")
    for feat, corr in sorted(correlacion_objetivo, key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"{feat}: {corr:.3f}")

def analisis_patrones_trafico(df):
    
    #Comenzamos a continuación a analizar los patrones específicos del tráfico DDoS vs Normal.
    
    print("\n=== Análisis de Patrones de Tráfico ===")
    
    #Realizamos la distribución de clases.
    print("\nDistribución del tráfico:")
    clase_dist = df['Label'].value_counts(normalize=True)
    print(clase_dist)
    
    #Visualizamos la distribución.
    plt.figure(figsize=(8, 6))
    clase_dist.plot(kind='bar')
    plt.title('Distribución de Tráfico Normal vs DDoS')
    plt.xlabel('Tipo de Tráfico')
    plt.ylabel('Proporción')
    plt.tight_layout()
    plt.show()

def analisis_estadistico_features(df):
    
    #Realizamos el análisis estadístico detallado de features por tipo de tráfico.
    
    print("\n=== Análisis Estadístico por Tipo de Tráfico ===")
    
    for clase in df['Label'].unique():
        datos_clase = df[df['Label'] == clase]
        print(f"\nEstadísticas para {clase}:")
        print(datos_clase.describe().round(3))
        
        #Identificamos features distintivas.
        otras_clases = df[df['Label'] != clase]
        
        print(f"\nFeatures más distintivas para {clase}:")
        for col in df.drop('Label', axis=1).columns:
            diff = abs(datos_clase[col].mean() - otras_clases[col].mean())
            std_pooled = (datos_clase[col].std() + otras_clases[col].std()) / 2
            if std_pooled != 0:
                effect_size = diff / std_pooled
                if effect_size > 0.5:
                    print(f"{col}: Effect size = {effect_size:.3f}")


def analisis_univariante_de_features(df):
    
    #Realizamos un análisis univariante detallado de cada feature (a excepción del target).
    
    print("\n=== Análisis Univariante de Features ===")
    
    #Excluimos nuestro target 'Label' del análisis univariante.
    features = [col for col in df.columns if col != 'Label']
    
    for feature in features:
        plt.figure(figsize=(12, 5))
        
        #Creamos subplot con dos gráficos.
        plt.subplot(1, 2, 1)
        
        for label in df['Label'].unique():
            sns.histplot(data=df[df['Label'] == label], x=feature, 
                        label=label, alpha=0.5, bins=30)
        plt.title(f'Distribución de {feature} por Clase')
        plt.xlabel(feature)
        plt.ylabel('Frecuencia')
        plt.legend()
        
        #Realizamos incluso gráficos Boxplot para visualizar la distribución de cada feature.
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x='Label', y=feature)
        plt.title(f'Boxplot de {feature} por Clase')
        
        plt.tight_layout()
        plt.show()
        
        #Realizamos las estadísticas descriptivas.
        print(f"\nEstadísticas para {feature}:")
        print(df.groupby('Label')[feature].describe())
        print("\nInterpretación:")
        
        #Realizamos el análisis específico para cada feature.
        if feature == 'Flow Duration':
            print("- Duración del flujo de red")
            print("- Medido en microsegundos")
            print(f"- Rango: {df[feature].min():.2f} a {df[feature].max():.2f}")
            
        elif feature == 'Total Fwd Packets':
            print("- Total de paquetes enviados en dirección forward")
            print(f"- Media para tráfico normal: {df[df['Label']=='BENIGN'][feature].mean():.2f}")
            print(f"- Media para DDoS: {df[df['Label']=='DDoS'][feature].mean():.2f}")
            
        elif feature == 'Total Backward Packets':
            print("- Total de paquetes enviados en dirección backward")
            print(f"- Media para tráfico normal: {df[df['Label']=='BENIGN'][feature].mean():.2f}")
            print(f"- Media para DDoS: {df[df['Label']=='DDoS'][feature].mean():.2f}")
            
        elif feature == 'Flow Bytes/s':
            print("- Tasa de bytes por segundo en el flujo")
            print(f"- Mediana para tráfico normal: {df[df['Label']=='BENIGN'][feature].median():.2f}")
            print(f"- Mediana para DDoS: {df[df['Label']=='DDoS'][feature].median():.2f}")
            
        elif feature == 'Flow Packets/s':
            print("- Tasa de paquetes por segundo en el flujo")
            print(f"- Mediana para tráfico normal: {df[df['Label']=='BENIGN'][feature].median():.2f}")
            print(f"- Mediana para DDoS: {df[df['Label']=='DDoS'][feature].median():.2f}")
            
        elif feature == 'asimetria_de_trafico':
            print("- Medida de la asimetría entre paquetes forward y backward")
            print(f"- Media para tráfico normal: {df[df['Label']=='BENIGN'][feature].mean():.2f}")
            print(f"- Media para DDoS: {df[df['Label']=='DDoS'][feature].mean():.2f}")
        
        print("\n" + "="*50)

def main():
    print("=== Análisis Exploratorio para Detección de DDoS ===")
    df = pd.read_csv("./features_seleccionadas.csv", sep=",")
    
    #Información básica a tener en cuenta de nuestro dataset, como el número de filas, columnas, tipos de datos, etc.
    print("\nInformación básica del dataset:")
    print(df.info())
    
    #Funciones a tener en cuenta para realizar el análisis completo.
    matriz_de_correlacion(df)
    analisis_patrones_trafico(df)
    analisis_estadistico_features(df)
    analisis_univariante_de_features(df)
    
if __name__ == "__main__":
    main()