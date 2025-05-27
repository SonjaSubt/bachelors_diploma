import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

# Створюємо папку для моделей
os.makedirs("models/rf", exist_ok=True)

# Списки файлів і назви для виводу
datasets = {
    "original": "../data/dataset.xlsx",
    "best": "../data/best_dataset.xlsx",
    "average": "../data/average_dataset.xlsx"
}

# Стовпці дескрипторів
descriptor_cols = [
    "Molecular Weight", "LogP", "TPSA", "Rotatable Bonds", "HBD", "HBA",
    "Aromatic Rings", "Fraction CSP3",
    "Heavy Atom Count", "Heavy Atom Count (no H)", "Molar Refractivity",
    "Bertz Index", "Balaban Index", "Chi0", "Chi1", "Chiral Centers"
]

# Функція для перетворення фінгерпринта у масив
def fingerprint_to_array(fp):
    return np.array([int(x) for x in fp])

# Основна функція
def train_on_dataset(name, filename):
    print(f"\n=== Training on {name.upper()} dataset ===")

    df = pd.read_excel(filename)

    # Видалення зайвих колонок
    df = df.drop(columns=[col for col in df.columns if "Unnamed" in col or col in ["id", "molar_ic50", "nano_ic50"]],
                 errors="ignore")

    # Обробка фінгерпринтів
    fingerprint_array = df["morgan_fingerprint"].apply(fingerprint_to_array)
    fingerprint_df = pd.DataFrame(fingerprint_array.tolist())

    # Формування X, y
    X = pd.concat([df[descriptor_cols], fingerprint_df], axis=1)
    X.columns = X.columns.astype(str)
    y = df["pIC50"]

    # Розділення
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)

    # Масштабування лише дескрипторів
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[descriptor_cols] = scaler.fit_transform(X_train[descriptor_cols])
    X_test_scaled[descriptor_cols] = scaler.transform(X_test[descriptor_cols])

    # Навчання Random Forest моделі
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    #
    # # Важливість ознак
    # feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    # feature_importances = feature_importances.sort_values(ascending=False)
    #
    # # Збереження важливості у CSV
    # feature_importances.to_csv(f"models/rf/feature_importance_{name}.csv")
    #
    # # Вивід ТОП-10 ознак
    # print("\nTop 15 важливих ознак:")
    # print(feature_importances.head(15))
    #
    # import matplotlib.pyplot as plt
    #
    # def plot_feature_importance(importances, title, filename):
    #     top_n = importances.head(15)
    #     plt.figure(figsize=(10, 6))
    #     top_n.plot(kind='barh')
    #     plt.title(title)
    #     plt.gca().invert_yaxis()
    #     plt.tight_layout()
    #     plt.savefig(filename)
    #     plt.close()
    #
    # # Виклик у функції train_on_dataset
    # plot_feature_importance(
    #     feature_importances,
    #     f"Top 15 Features Importance - {name}",
    #     f"models/rf/feature_importance_{name}.png"
    # )

    # Передбачення
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Метрики
    def evaluate(y_true, y_pred, label):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        med_ae = median_absolute_error(y_true, y_pred)
        print(f"{label}: R²: {r2:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | Median AE: {med_ae:.4f}")

    evaluate(y_train, y_train_pred, "Train")
    evaluate(y_test, y_test_pred, "Test")

    # Збереження моделі і скейлера
    joblib.dump(model, f"models/rf/rf_model_{name}_02.pkl")
    joblib.dump(scaler, f"models/rf/rf_scaler_{name}_02.pkl")

# Запуск для всіх трьох датасетів
for name, path in datasets.items():
    train_on_dataset(name, path)
