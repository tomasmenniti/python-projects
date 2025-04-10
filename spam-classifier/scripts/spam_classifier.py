import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import os

# 1. Generar un dataset ficticio más grande
print("1. Generación de un dataset ficticio")

# Datos simulados de emails (aumentamos el tamaño)
emails = [
    "Gana dinero rápido, haz clic aquí",  # Spam
    "Reunión mañana a las 10",           # No spam
    "¡Oferta especial, compre ahora!",   # Spam
    "Recordatorio: entrega el informe",  # No spam
    "¡Ganaste un premio, reclama ahora!", # Spam
    "Hola, ¿puedes ayudarme con el proyecto?", # No spam
    "Compra barata, enlace exclusivo",   # Spam
    "Invitación a la fiesta del viernes", # No spam
    "Dinero gratis, regístrate hoy",     # Spam
    "Envío de reporte semanal",          # No spam
    "Promoción increíble, únete",        # Spam
    "Llamada de equipo mañana",          # No spam
] * 5  # Repetimos los datos 5 veces para tener 60 emails

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5  # 1 = Spam, 0 = No spam

# Crear DataFrame
data = pd.DataFrame({'Email': emails, 'Etiqueta': labels})

# Guardar el dataset en un archivo CSV en la carpeta data
if not os.path.exists('data'):
    os.makedirs('data')  # Crear carpeta data si no existe
data.to_csv('data/spam_emails.csv', index=False)
print("\nDataset guardado como 'data/spam_emails.csv'")

print("\nDataset inicial (primeras 10 filas):")
print(data.head(10))

# 2. Preprocesamiento de los datos
print("\n2. Preprocesamiento de los datos")

# Convertir texto a una matriz de conteo de palabras
vectorizer = CountVectorizer(stop_words='english')  # Elimina palabras comunes como "the", "is"
X = vectorizer.fit_transform(data['Email']).toarray()
y = data['Etiqueta']

# Ver las palabras extraídas y su representación
print("\nPalabras extraídas (vocabulario):")
print(vectorizer.get_feature_names_out())
print("\nMatriz de conteo (X) - primeras 5 filas:")
print(X[:5])

# 3. Dividir los datos en entrenamiento y prueba
print("\n3. División de los datos")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTamaño del conjunto de entrenamiento:", X_train.shape)
print("Tamaño del conjunto de prueba:", X_test.shape)

# 4. Entrenar el modelo
print("\n4. Entrenamiento del modelo")

# Usar Naive Bayes para clasificación
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Hacer predicciones
print("\n5. Predicciones")

y_pred = model.predict(X_test)
print("\nPredicciones vs Valores reales (primeras 5):")
predictions_df = pd.DataFrame({'Real': y_test[:5], 'Predicho': y_pred[:5]})
print(predictions_df)

# 6. Evaluar el modelo
print("\n6. Evaluación del modelo")

# Calcular precisión
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión: {accuracy:.2f}")

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusión:")
print(conf_matrix)

# Reporte de clasificación (especificar labels para evitar el error)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['No Spam', 'Spam']))

# 7. Visualización
print("\n7. Visualización")

# Gráfico de la matriz de confusión
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Spam', 'Spam'], yticklabels=['No Spam', 'Spam'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.show()

# 8. Experimenta: Predicción de un nuevo email
print("\n8. Predicción de un nuevo email")
new_email = ["¡Gana dinero fácil ahora!"]
new_email_vectorized = vectorizer.transform(new_email).toarray()
prediction = model.predict(new_email_vectorized)
print(f"Predicción para el nuevo email: {'Spam' if prediction[0] == 1 else 'No Spam'}")