import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import joblib

# Шлях до моделей
os.makedirs("models/linear", exist_ok=True)

# Шлях до файлів
datasets = {
    "original": "../data/dataset.xlsx",
    "best": "../data/best_dataset.xlsx",
    "average": "../data/average_dataset.xlsx"
}

# Колонки, які не використовуються
drop_cols = ["id", "molar_ic50", "nano_ic50", "smiles", "source"]

# Цільова змінна
target = "pIC50"

def fingerprint_to_array(fp):
    return np.array([int(x) for x in fp])

def process_and_train(name, path):
    print(f"\n=== Training on {name.upper()} dataset ===")
    df = pd.read_excel(path)

    # Видалити Unnamed-колонки
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Видаляємо зайві колонки
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    # Витягаємо фінгерпринти
    fingerprint_array = df["morgan_fingerprint"].apply(fingerprint_to_array)
    fingerprint_df = pd.DataFrame(fingerprint_array.tolist(), index=df.index)

    # Знаходимо дескриптори автоматично (числові, без pIC50 і fingerprint)
    descriptor_cols = [col for col in df.select_dtypes(include='number').columns if col != target]

    # Формуємо X та y
    # Формуємо X та y
    X = pd.concat([df[descriptor_cols], fingerprint_df], axis=1)
    X.columns = X.columns.astype(str)  # ← ВАЖЛИВО: імена колонок мають бути рядками!
    y = df[target]


    # Розділення
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Масштабування дескрипторів (тільки descriptor_cols, не fingerprint)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[descriptor_cols] = scaler.fit_transform(X_train[descriptor_cols])
    X_test_scaled[descriptor_cols] = scaler.transform(X_test[descriptor_cols])

    # Навчання моделі
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Прогноз
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Метрики
    def print_metrics(y_true, y_pred, label):

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        med_ae = median_absolute_error(y_true, y_pred)
        print(f"{label}: R²: {r2:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | Median AE: {med_ae:.4f}")

    print_metrics(y_train, y_train_pred, "Train")
    print_metrics(y_test, y_test_pred, "Test")

    # Зберігаємо модель і скейлер
    joblib.dump(model, f"models/linear/linear_model_{name}.pkl")
    joblib.dump(scaler, f"models/linear/linear_scaler_{name}.pkl")

# Запуск для кожного варіанту
for name, path in datasets.items():
    process_and_train(name, path)
