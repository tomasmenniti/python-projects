import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import os  # Para manejar directorios
import joblib  # Para guardar el modelo

# 1. Descargar datos en tiempo real/históricos usando yfinance
ticker = 'TSLA'
end_date = datetime.now()
start_date = end_date - timedelta(days=365)  # Datos de los últimos 365 días

stock_data = yf.download(ticker, start=start_date, end=end_date)

# Verificar si los datos se descargaron correctamente
if stock_data.empty:
    print("No se pudieron descargar los datos. Verifica el ticker o la conexión.")
    exit()

# 2. Exploración inicial de los datos
print(f"Datos de {ticker}:")
print("Primeras 5 filas del dataset:")
print(stock_data.head())
print("\nInformación del dataset:")
print(stock_data.info())
print("\nEstadísticas descriptivas:")
print(stock_data.describe())

# 3. Preprocesamiento de datos
# Seleccionar columnas relevantes como features y target
features = ['Open', 'High', 'Low', 'Volume', 'Close']  # Incluimos Close temporalmente
target = 'Close'  # Variable objetivo: precio de cierre

# Eliminar filas con valores nulos (aunque yfinance suele ser confiable)
stock_data = stock_data.dropna()

# Crear un lag (desplazamiento) para predecir el próximo cierre
stock_data['Target'] = stock_data['Close'].shift(-1)  # El próximo precio de cierre como target

# Eliminar la última fila (ya que no tendrá target)
stock_data = stock_data.dropna()

# Definir X (features) y y (target)
X = stock_data[['Open', 'High', 'Low', 'Volume', 'Close']]  # Usamos Close como feature para predecir el siguiente Close
y = stock_data['Target']  # Precio de cierre futuro

# 4. Dividir los datos en entrenamiento y prueba
# Usaremos una división temporal en lugar de aleatoria para simular predicciones en tiempo real
train_size = int(len(X) * 0.8)  # 80% para entrenamiento, 20% para prueba

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# 5. Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Hacer predicciones
y_pred = model.predict(X_test)

# 7. Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nResultados del modelo:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# 8. Visualizar los resultados
plt.figure(figsize=(12, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', alpha=0.5, label='Precios reales')
plt.scatter(range(len(y_pred)), y_pred, color='red', alpha=0.5, label='Precios predichos')
plt.title(f'Comparación: Precios reales vs. Predichos para {ticker}')
plt.xlabel('Índice de tiempo')
plt.ylabel('Precio de cierre')
plt.legend()
plt.tight_layout()
plt.show()

# Opcional: Graficar los precios históricos
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, stock_data['Close'], label='Precio de cierre histórico')
plt.title(f'Precios históricos de {ticker}')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9. Guardar el modelo y los datos
data_folder = 'data'
os.makedirs(data_folder, exist_ok=True)  # Crea la carpeta 'data' si no existe

# Guardar el archivo CSV en la carpeta 'data'
stock_data.to_csv(os.path.join(data_folder, f'{ticker}_stock_data.csv'))
print(f"Datos guardados como '{os.path.join(data_folder, f'{ticker}_stock_data.csv')}'")

# Guardar el modelo en la carpeta 'data'
joblib.dump(model, os.path.join(data_folder, f'{ticker}_stock_price_model.pkl'))
print(f"Modelo guardado como '{os.path.join(data_folder, f'{ticker}_stock_price_model.pkl')}'")

# Opcional: Mostrar coeficientes del modelo
coef_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
print("\nCoefficients del modelo:")
print(coef_df)