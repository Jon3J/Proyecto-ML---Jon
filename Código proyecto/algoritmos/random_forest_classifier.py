#Importación de las librerías.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import roc_curve, auc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def plot_roc_auc(y_true, y_pred_proba):
    #En este método vamos a realizar la curva ROC y calcular el AUC

    #Vamos a asegurarnos de que estamos usando las etiquetas correctamente.
    y_true_binary = (y_true == 'DDoS').astype(int)  # Convertimos DDoS a 1, BENIGN a 0
    
    # Calculamos a continuación, la curva ROC para realizar un modelo más robusto.
    fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba) #Aplicamos la función de "roc_curve".
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC - Random Forest')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('algoritmos/resultados/roc_curve_randforest.png')
    plt.show()
    return roc_auc

def preparar_datos_antes_modelo(df):
    #Este método se encarga de preparar los datos separando features y target, y dividiendo en train y en test.
    X = df.drop('Label', axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def escalar_datos(X_train, X_test):
    #Este método escala los datos empleando StandardScaler y aplicamos para X_train fit_transform y para X_test transform.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def plot_confusion_matrix(y_true, y_pred):
    #En este método nos encargamos de visualizar la matriz de confusión que acabaremos guardándola como una imagen dentro de la carpeta resultados como ".png".
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión - Random Forest')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.savefig('algoritmos/resultados/confusion_matrix_randforest.png')
    plt.show()

def plot_feature_importance(model, feature_names):
    #Este método, trata de visualizar la importancia de las características del modelo.
    importances = model.best_estimator_.feature_importances_ #definimos una variable importances, que almacena el mejor estimador del modelo.
    indices = np.argsort(importances)[::-1]  # Ordenamos de mayor a menor
    
    plt.figure(figsize=(10, 8))
    sns.barplot( #Realizamos un gráfico de barras empleando la librería de Python "Seaborn".
        x=importances[indices], 
        y=np.array(feature_names)[indices], 
        hue=np.array(feature_names)[indices],  #Asignamos el hue.
        legend=False,  #Podemos ocultar en este caso, la leyenda ya que no es necesaria.
        palette="viridis"
    )
    plt.title("Importancia de Características - Random Forest")
    plt.xlabel("Importancia") #La importancia de dichas features en el eje x.
    plt.ylabel("Características") #Las features (características) en el eje y.
    plt.tight_layout()
    plt.savefig('algoritmos/resultados/feature_importance_randforest.png')
    plt.show() #Mostramos dicho gráfico de barras.

def analizar_caracteristicas(df):
    #Con este método, tratamos de analizar las características del dataset para detectar posibles problemas.
    X = df.drop('Label', axis=1) #Guardamos en una variable X nuestras features.
    y = df['Label'] #Guardamos en una variable y nuestro target.
    
    print("\n=== Análisis de Características ===")
    
    #Como primer paso, primeramente realizamos una distribución de clases, permitiéndonos saber, si hay
    # desequilibrio de clases (cosa que puede llegar a afectar a nuestro modelo).
    print("\nDistribución de clases:")
    print(y.value_counts(normalize=True))
    
    #Analizamos los casos de características con varianza muy baja.
    varianzas = X.var() #Para ello, calculamos la varianza de cada característica.
    caract_baja_varianza = varianzas[varianzas < 0.01].index.tolist() #Si vemos que una de las características es menor a 0.01
    #consideramos que se aporta poca información, guardando luego el nombre de la característica para su eliminación.
    if caract_baja_varianza:
        print("\nCaracterísticas con varianza muy baja (<0.01):")
        print(caract_baja_varianza) #Imprimimos las características de varianza más bajas.
    
    #Analizamos también las correlaciones altas entre las características.
    correlaciones = X.corr() #Calculamos la matriz de correlación entre las características.
    caract_correlacionadas = []
    for i in range(len(X.columns)):
        for j in range(i+1, len(X.columns)):
            if abs(correlaciones.iloc[i,j]) > 0.95: #Tratamos de buscar, como tal, aquellas características con un valor mayor a 0.95, indicandonos así las características más correlacionadas.
                caract_correlacionadas.append(X.columns[j]) #Agregamos las características que son redundantes a la lista (para eliminación).
    
    #Realizamos la correlación con la variable objetivo.
    correlacion_objetivo = []
    for col in X.columns:
        corr = abs(pd.get_dummies(y)[y.unique()[0]].corr(X[col]))
        if corr > 0.8:  #Umbral más estricto.
            correlacion_objetivo.append((col, corr))
    
    #Realizamos este paso para almacenar la combinación de todas las características anteriores detectadas como problemáticas para su eliminación.
    caracteristicas_a_eliminar = list(set(
        caract_baja_varianza + 
        caract_correlacionadas + 
        [col for col, _ in correlacion_objetivo]
    ))
    
    return caracteristicas_a_eliminar #Devolvemos una lista de las características que podrían eliminarse.

def entrenar_y_evaluar_modelo(X_train_scaled, X_test_scaled, y_train, y_test):
    #Entrenamos el modelo de Random Forest usando GridSearchCV y evaluamos su rendimiento.
    
    #Definimos los parámetros para la búsqueda.
    param_grid = {
        "n_estimators": [50],
        "max_features": ['sqrt'],
        "max_depth": [3, 5],           # Profundidad más limitada
        "min_samples_split": [20, 30],  # Valores más altos
        "min_samples_leaf": [10, 15],   # Valores más altos
        "bootstrap": [True]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier( #Realizamos el algoritmo de Random Forest.
            random_state=42, #Aplicamos para el random_state el número mágico de ML = 42.
            class_weight='balanced'  # Manejamos el desbalance que pueda llegar a haber.
        ),
        param_grid, #Añadimos el diccionario de hiperparámetros que hemos definido antes.
        cv=5, #Realizamos un cross validation de 5 particiones.
        scoring='recall_macro', #Es util en casos de desequilibrio de clases, ya que da el mismo peso a todas las clases (Además es la métrica que nos interesa para nuestro modelo).
        verbose=2, #Nos imprime un resumen detallado de la búsqueda de hiperparámetros.
        n_jobs=-1 #Se emplean todos los núcleos del procesador para acelerar la búsqueda de los hiperparámetros.
    )
    
    grid_search.fit(X_train_scaled, y_train)
    return grid_search

def main():
    #Antes de comenzar a realizar el modelo de Random Forest, creamos un directorio para resultados si no existe llamado "resultados".
    os.makedirs('algoritmos/resultados', exist_ok=True)
    
    try:
        #El primer paso que debemos de realizar es cargar nuestro dataset (Que se tratará del dataset, con mis features seleccionadas).
        print("Cargando datos...")
        df = pd.read_csv('./features_seleccionadas.csv')
        
        #Analizamos y eliminamos las características problemáticas (empleando para ello, el método "analizar_caracteristicas").
        caracteristicas_a_eliminar = analizar_caracteristicas(df)
        if caracteristicas_a_eliminar:
            print("\nEliminando características problemáticas:", caracteristicas_a_eliminar)
            df = df.drop(columns=caracteristicas_a_eliminar)
        
        #Preparamos y escalamos los datos.
        X_train, X_test, y_train, y_test = preparar_datos_antes_modelo(df)
        X_train_scaled, X_test_scaled = escalar_datos(X_train, X_test)
        
        #Entrenamos y evaluamos nuestro modelo.
        grid_search = entrenar_y_evaluar_modelo(X_train_scaled, X_test_scaled, y_train, y_test)
        
        #Evaluamos el modelo.
        print("\nEvaluando modelo en conjunto de prueba...")
        y_pred = grid_search.predict(X_test_scaled)
        
        #Obtenemos las probabilidades para la clase positiva (DDoS)
        y_pred_proba = grid_search.predict_proba(X_test_scaled)[:, 1]
        
        #Calculamos y mostramos la curva ROC.
        roc_auc_score = plot_roc_auc(y_test, y_pred_proba)
        print(f"\nAUC-ROC Score: {roc_auc_score:.3f}")
        
        #Mostramos los resultados.
        print("\nMejores parámetros encontrados:")
        print(grid_search.best_params_)
        print(f"\nMejor puntuación de validación cruzada: {grid_search.best_score_:.3f}")
        
        print("\nReporte de clasificación en conjunto de prueba:")
        print(classification_report(y_test, y_pred))
        
        #Visualizamos y guardamos la matriz de confusión (método definido arriba).
        print("\nGenerando matriz de confusión...")
        plot_confusion_matrix(y_test, y_pred)
        
        # Visualizamos y guardamos la importancia de las características.(Las features).
        print("\nGenerando importancia de las características...")
        plot_feature_importance(grid_search, df.drop('Label', axis=1).columns)
        
        #Guardamos los resultados en un archivo ".txt" llamado "logistic_results.txt" (Con codificación 'utf-8').
        with open('algoritmos/resultados/random_forest_results.txt', 'w', encoding='utf-8') as f:
            f.write("=== Resultados del modelo de Random Forest ===\n\n")
            f.write(f"Mejores parámetros: {grid_search.best_params_}\n")
            f.write(f"Mejor puntuación CV: {grid_search.best_score_:.3f}\n\n")
            
            #Agregamos información sobre overfitting.
            train_score = grid_search.score(X_train_scaled, y_train)
            test_score = grid_search.score(X_test_scaled, y_test)
            f.write("\nAnálisis de Overfitting:\n")
            f.write(f"Puntuación en entrenamiento: {train_score:.3f}\n")
            f.write(f"Puntuación en prueba: {test_score:.3f}\n")
            f.write(f"Diferencia (train - test): {train_score - test_score:.3f}\n\n")
            
            f.write("Reporte de clasificación en prueba:\n")
            f.write(classification_report(y_test, y_pred))
            
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")

if __name__ == "__main__":
    main()
