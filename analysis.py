
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Зчитування Excel-файлу
file_path = "data/average_dataset.xlsx"  # ← Заміни на назву свого файлу
df = pd.read_excel(file_path)

# Замінюємо кому на крапку для числових колонок і перетворюємо в float
for col in df.columns:
    try:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False).astype(float)
    except:
        continue  # Пропускаємо нечислові колонки (наприклад, SMILES, source, fingerprint)

# 🔍 Аналіз розподілу pIC50
plt.figure(figsize=(8, 5))
sns.histplot(df['pIC50'], kde=True, bins=20, color='skyblue')
plt.title('Розподіл pIC50 (середнє значення)')
plt.xlabel('pIC50')
plt.ylabel('Частота')
plt.grid(True)
plt.show()
if 'morgan_fingerprint' in df.columns:
    df_corr = df.drop(columns=['morgan_fingerprint'])
else:
    df_corr = df.copy()

# Розрахунок кореляційної матриці (тільки числові колонки)
corr = df_corr.corr(numeric_only=True)

# Побудова графіка
fig, ax = plt.subplots(figsize=(14, 12))  # Збільшуємо розмір для уникнення накладень
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", square=True, ax=ax)

# Зсув графіка: більше простору зліва і зверху
plt.subplots_adjust(left=0.25, right=0.95, top=0.92, bottom=0.15)

# Назва
ax.set_title('Кореляційна матриця дескрипторів', fontsize=16, pad=20)

plt.show()
# 📈 Топ 10 дескрипторів, корельованих із pIC50
corr_pIC50 = corr['pIC50'].drop('pIC50').sort_values(key=abs, ascending=False)
corr_pIC50.head(10).plot(kind='bar', title='Топ 10 дескрипторів за кореляцією з pIC50')
plt.ylabel('Коефіцієнт кореляції')
plt.grid(True)
plt.show()

# 🔗 Pairplot для 3 найважливіших дескрипторів
top_features = corr_pIC50.head(3).index.tolist()
sns.pairplot(df[top_features + ['pIC50']])
plt.suptitle("Парні залежності між топ-дескрипторами і pIC50", y=1.02)
plt.show()

# 📦 Аналіз активності (pIC50 > 6)
df['active'] = df['pIC50'] > 6

# Boxplot для LogP як приклад
if 'LogP' in df.columns:
    sns.boxplot(x='active', y='LogP', data=df)
    plt.title('Розподіл LogP між активними і неактивними сполуками')
    plt.xlabel('Активність (pIC50 > 6)')
    plt.ylabel('LogP')
    plt.show()

for col in top_features:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=col, y="pIC50")
    sns.regplot(data=df, x=col, y="pIC50", scatter=False, color='red')
    plt.title(f"Залежність pIC50 від {col}")
    plt.show()

# # 🔹 5. Парна кореляція дескрипторів
# sns.pairplot(df[top_features.tolist() + ["pIC50"]])
# plt.suptitle("Парні залежності між top дескрипторами і pIC50", y=1.02)
# plt.show()

#6
sns.histplot(df['pIC50'], bins=30, kde=True)
plt.title('Розподіл pIC50')
plt.show()
