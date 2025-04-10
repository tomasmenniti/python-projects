## Clasificación de Emails como Spam/No Spam:

    Este proyecto utiliza un dataset ficticio de emails (texto simplificado y etiquetas) para entrenar un modelo de clasificación que determine si un email es spam o no spam. Se utiliza un modelo de Naive Bayes, que es común para problemas de texto, y se incluyen prácticas de preprocesamiento, entrenamiento y evaluación.



## Instalación 
    pip install numpy pandas seaborn scikit-learn matplotlib



## Información al ejecutar el código:

    - Dataset inicial con 8 emails.

    - Matriz de conteo de palabras.

    - Predicciones y métricas (precisión ~0.5-1.0 con este dataset pequeño).

    - Mapa de calor de la matriz de confusión.

    - Predicción para un nuevo email.



## Explicación:

    - Generación de un dataset ficticio:

        Se crea un conjunto pequeño de emails y etiquetas (spam = 1, no spam = 0) para simular un problema real.

    - Preprocesamiento de los datos:

        CountVectorizer convierte el texto en una matriz numérica contando la frecuencia de palabras. Crucial para modelos de ML, ya que no pueden procesar texto directamente.

        stop_words='english' elimina palabras comunes que no aportan mucho significado (por ejemplo, "and", "the").

    - División de los datos:

        train_test_split separa los datos en entrenamiento (80%) y prueba (20%) para evaluar el modelo.

    - Entrenamiento del modelo:

        MultinomialNB es un algoritmo de Naive Bayes adecuado para datos de conteo (como frecuencias de palabras). fit() entrena el modelo con los datos de entrenamiento.

    - Predicciones:

        predict() usa el modelo entrenado para clasificar los emails de prueba.

    - Evaluación del modelo:

        accuracy_score mide la proporción de predicciones correctas.

        confusion_matrix muestra verdaderos positivos, falsos positivos, etc.

        classification_report da precisión, recall y F1-score por clase.

    - Visualización:

        Un mapa de calor (sns.heatmap) ilustra la matriz de confusión, ayudando a interpretar los errores.

    - Predicción personalizada:

        Permite probar el modelo con un nuevo email, reforzando su uso práctico.