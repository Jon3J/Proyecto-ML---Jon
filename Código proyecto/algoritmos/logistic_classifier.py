#Importación de las librerías.
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    plt.title('Matriz de Confusión - Regresión Logística')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.savefig('algoritmos/resultados/confusion_matrix_logistic.png')
    plt.show()

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
        if corr > 0.8:  #Aplicamos un umbral
            correlacion_objetivo.append((col, corr))
    
    #Realizamos este paso para almacenar la combinación de todas las características anteriores detectadas como problemáticas para su eliminación.
    caracteristicas_a_eliminar = list(set(
        caract_baja_varianza + 
        caract_correlacionadas + 
        [col for col, _ in correlacion_objetivo]
    ))
    
    return caracteristicas_a_eliminar #Devolvemos una lista de las características que podrían eliminarse.

def entrenar_y_evaluar_modelo(X_train_scaled, X_test_scaled, y_train, y_test):
    #Entrenamos el modelo de Regresión Logística usando GridSearchCV y evaluamos su rendimiento.
    
    #Definimos los parámetros para la búsqueda.
    param_grid = {
        'C': [0.1, 1, 10, 100], #Este parámetro, controla la regularización.
        'penalty': ['l1', 'l2'], #Nos especifica, el tipo de penalización, que se aplica durante el entrenamiento para la regularización.
        'solver': ['liblinear'], #Parámetro para optimizar los parámetros del modelo durante el entrenamiento.
        'max_iter': [1000] #Es el número máximo de iteraciones que se realiza para entrenar el modelo.
    }
    
    print("\nIniciando búsqueda de hiperparámetros...")
    n_combinaciones = len(param_grid['C']) * len(param_grid['penalty'])
    print(f"Configuraciones a probar: {n_combinaciones} combinaciones")
    
    # Búsqueda de mejores parámetros
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42,class_weight='balanced'), #Aplicamos el modelo de Regresión Logística.
        param_grid, #Añadimos el diccionario de hiperparámetros que hemos definido antes.
        cv=5, #Realizamos un cross validation de 5 particiones.
        scoring='recall_macro', #Es util en casos de desequilibrio de clases, ya que da el mismo peso a todas las clases (Además es la métrica que nos interesa para nuestro modelo).
        n_jobs=-1, #Se emplean todos los núcleos del procesador para acelerar la búsqueda de los hiperparámetros.
        verbose=2 #Nos imprime un resumen detallado de la búsqueda de hiperparámetros.
    )
    
    print("\nEntrenando modelo con validación cruzada...")
    grid_search.fit(X_train_scaled, y_train) #Entrenamos el modelo.
    
    return grid_search

def main():
    #Antes de comenzar a realizar el modelo de Regresión Logística, creamos un directorio para resultados si no existe llamado "resultados".
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
        
        #Mostramos los resultados.
        print("\nMejores parámetros encontrados:")
        print(grid_search.best_params_)
        print(f"\nMejor puntuación de validación cruzada: {grid_search.best_score_:.3f}")
        
        print("\nReporte de clasificación en conjunto de prueba:")
        print(classification_report(y_test, y_pred))
        
        #Visualizamos y guardamos la matriz de confusión (método definido arriba).
        print("\nGenerando matriz de confusión...")
        plot_confusion_matrix(y_test, y_pred)
        
        #Guardamos los resultados en un archivo ".txt" llamado "logistic_results.txt".
        with open('algoritmos/resultados/logistic_results.txt', 'w',encoding='utf-8') as f:
            f.write("=== Resultados del modelo de Regresión Logística ===\n\n")
            f.write(f"Mejores parámetros: {grid_search.best_params_}\n")
            f.write(f"Mejor puntuación CV: {grid_search.best_score_:.3f}\n\n")
            f.write("Reporte de clasificación:\n")
            f.write(classification_report(y_test, y_pred))
            
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")

if __name__ == "__main__":
    main() 