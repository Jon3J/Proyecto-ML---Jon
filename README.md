
# Predicción de Ataques DDoS en Tráfico de Datos

Este proyecto utiliza técnicas de Machine Learning para analizar y predecir posibles ataques DDoS (Distributed Denial of Service) en el tráfico de red. A través de la detección temprana de patrones anómalos, el modelo puede identificar amenazas antes de que causen daños significativos.

## Tabla de Contenidos

- Introducción

- Objetivos del Proyecto

- Datos

- Modelos de Machine Learning

- Requisitos

### Introducción

Los ataques DDoS representan una amenaza significativa para la infraestructura digital, interrumpiendo servicios y causando pérdidas económicas y reputacionales. Este proyecto está diseñado para mejorar las capacidades de detección utilizando algoritmos de Machine Learning que analizan grandes volúmenes de datos de tráfico en tiempo real.

### Objetivos del Proyecto

Diseñar un modelo capaz de detectar patrones anómalos asociados con ataques DDoS en tráfico de red.

Proporcionar un sistema robusto y escalable para implementar en entornos reales.

Poder llegar a minimizar falsos positivos y negativos para garantizar la eficacia de la detección.

### Datos

El modelo está entrenado con un dataset que contiene tráfico normal y tráfico asociado con ataques DDoS. El dataset empleado es:

CIC-IDS2017: Dataset perteneciente a "Instituto Canadiense de Ciberseguridad (UNB)" con datos etiquetados de tráfico normal y de ataques DDoS.

Los datos se preprocesan para eliminar valores nulos, normalizar las características y manejar desequilibrios en las clases.

### Modelo de Machine Learning

Se evaluaron múltiples algoritmos de Machine Learning para encontrar el modelo más efectivo, incluyendo:

Random Forest: Ofrece interpretabilidad y buen rendimiento con datos tabulares.

Support Vector Machine (SVM): Eficaz para separar clases con datos balanceados.

Logistic Regression: Estima la probabilidad de que ocurra o no ocurra un evento. Empleado para casos de clasificación.

La mejor arquitectura se seleccionó en función de la métrica de clasificación recall.

### Requisitos

* Software y Librerías

* Python 3.8+

Librerías principales:

- pandas

- numpy

- scikit-learn

- matplotlib y seaborn (para visualización de datos)


## Screenshots

![App Screenshot](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR0CU4iDUk6e_BcpY6pOtFaT2VWDkmH2OIzXw&s)

