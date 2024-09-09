# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print(plt.style.available)

plt.style.use("seaborn-v0_8-dark")


# %%
df = pd.read_csv("stores_sales_forecasting.csv", encoding="Latin-1")

print(df.head())

print(df.info())

print(df.isnull().sum())


# %%
# Tarih sütunlarını datetime formatına çevirelim
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Değişiklikleri kontrol edelim
print(df[['Order Date', 'Ship Date']].head())


# %%
# Kategorik değişkenlerin sınıf dağılımını görelim
categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    print(f"{col} - Sınıf sayısı: {df[col].nunique()}")
    print(df[col].value_counts())
    print("\n")


# %%
# Sayısal değişkenlerin dağılımlarını görselleştirelim
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

df[numerical_columns].hist(figsize=(12, 8), bins=20, color='skyblue', edgecolor='black')
plt.tight_layout()
plt.show()


# %%
# Sevkiyat süresini hesaplayalım (Ship Date - Order Date)
df['Shipping Duration'] = (df['Ship Date'] - df['Order Date']).dt.days

# Sevkiyat süresi ile ilgili istatistiksel bilgiler
print(df['Shipping Duration'].describe())

# Sevkiyat süresinin dağılımını görselleştirelim
plt.figure(figsize=(8, 6))
sns.histplot(df['Shipping Duration'], bins=20, kde=True, color='green')
plt.title("Sevkiyat Süresi Dağılımı")
plt.show()


# %%
# Object türündeki sütunları seçelim
categorical_columns = ["Ship Mode", "Segment", "Sub-Category" ,"Region"]
# Bu sütunları category veri tipine dönüştürelim
df[categorical_columns] = df[categorical_columns].astype('category')

# Değişiklikleri kontrol edelim
print(df.info())


# %%
# Temel istatistiksel özet
print(df.describe().T)


# %%
# Kategorik değişkenlerin dağılımlarını görselleştirelim
import seaborn as sns
import matplotlib.pyplot as plt

for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col, palette='Set2')
    plt.title(f'{col} Dağılımı')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
# Sayısal değişkenlerin histogramlarını çizelim
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

df[numerical_columns].hist(figsize=(12, 8), bins=20, color='skyblue', edgecolor='black')
plt.tight_layout()
plt.show()


# %%
# Korelasyon matrisini hesaplayalım
corr_matrix = df[numerical_columns].corr()

# Korelasyon matrisini görselleştirelim
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Korelasyon Matrisi')
plt.show()


# %%
# Zaman bazında satışların değişimini inceleyelim
df.set_index('Order Date', inplace=True)

# Aylık satış trendi
monthly_sales = df['Sales'].resample('M').sum()

plt.figure(figsize=(10, 6))
monthly_sales.plot(color='purple')
plt.title('Aylık Satış Trendi')
plt.ylabel('Toplam Satış')
plt.xlabel('Tarih')
plt.show()

# Veri setini eski haline getirelim (index olarak tekrar Row ID kullanalım)
df.reset_index(inplace=True)


# %%
# Sevkiyat süresinin kâr ve satış üzerindeki etkisi
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Shipping Duration', y='Profit', data=df, hue='Sales', palette='viridis', size='Sales')
plt.title('Sevkiyat Süresi ile Kâr ve Satış İlişkisi')
plt.show()


# %%
# Segment ve ürün kategorisi bazında toplam satış ve kâr
plt.figure(figsize=(12, 6))

# Segment bazında
plt.subplot(1, 2, 1)
sns.barplot(x='Segment', y='Sales', data=df, estimator=sum, palette='pastel')
plt.title('Segment Bazında Toplam Satış')

# Ürün kategorisi bazında
plt.subplot(1, 2, 2)
sns.barplot(x='Category', y='Sales', data=df, estimator=sum, palette='pastel')
plt.title('Ürün Kategorisi Bazında Toplam Satış')

plt.tight_layout()
plt.show()

# Kar analizi
plt.figure(figsize=(12, 6))

# Segment bazında
plt.subplot(1, 2, 1)
sns.barplot(x='Segment', y='Profit', data=df, estimator=sum, palette='pastel')
plt.title('Segment Bazında Toplam Kâr')

# Ürün kategorisi bazında
plt.subplot(1, 2, 2)
sns.barplot(x='Category', y='Profit', data=df, estimator=sum, palette='pastel')
plt.title('Ürün Kategorisi Bazında Toplam Kâr')

plt.tight_layout()
plt.show()


# %%
plt.figure(figsize=(12, 6))

# Bölge bazında toplam satış
plt.subplot(1, 2, 1)
sns.barplot(x='Region', y='Sales', data=df, estimator=sum, palette='pastel')
plt.title('Bölge Bazında Toplam Satış')

# Bölge bazında toplam kâr
plt.subplot(1, 2, 2)
sns.barplot(x='Region', y='Profit', data=df, estimator=sum, palette='pastel')
plt.title('Bölge Bazında Toplam Kâr')

plt.tight_layout()
plt.show()


# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Discount', y='Sales', data=df, color='blue', alpha=0.6)
plt.title('İndirimler ve Satışlar Arasındaki İlişki')
plt.xlabel('İndirim')
plt.ylabel('Satış')
plt.show()


# %%
# Tarih sütununu indeks olarak ayarlama
df.set_index('Order Date', inplace=True)
# Aylık toplam kârı hesaplayalım
monthly_profit = df['Profit'].resample('M').sum()

# Aylık toplam kârı çizdirelim
plt.figure(figsize=(10, 6))
monthly_profit.plot(color='green')
plt.title('Aylık Toplam Kâr')
plt.ylabel('Toplam Kâr')
plt.xlabel('Tarih')
plt.grid(True)
plt.show()


# %%
# Sevkiyat tarihi sütununu datetime formatına çevir
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Sevkiyat tarihini indeks olarak ayarlama
df.set_index('Ship Date', inplace=True)

# Aylık toplam kârı hesaplayalım
monthly_profit = df['Profit'].resample('M').sum()

# Aylık toplam kârı çizdirelim
plt.figure(figsize=(10, 6))
monthly_profit.plot(color='green')
plt.title('Aylık Toplam Kâr')
plt.ylabel('Toplam Kâr')
plt.xlabel('Tarih')
plt.grid(True)
plt.show()


# %%
from sklearn.model_selection import train_test_split

# Özellikler ve hedef değişkeni tanımlama
features = df[['Sales', 'Quantity', 'Discount']]  # Özellikler
target = df['Profit']  # Hedef değişken

# Eğitim ve test kümesine ayırma
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Modeli oluşturma
model = LinearRegression()

# Modeli eğitim verisi ile eğitme
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Sonuçları değerlendirme
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")


# %%
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Gerçek Kâr')
plt.ylabel('Tahmin Edilen Kâr')
plt.title('Gerçek ve Tahmin Edilen Kârlar')
plt.grid(True)
plt.show()


# %% [markdown]
# # Satış ve Kâr Analizi Raporu
# **Tarih:** 2024-09-10  
# **Yazar:** Hakan Çelik
# 
# ## Giriş
# Bu rapor, Store Sales Forecasting veri seti üzerinde yapılan analiz ve modelleme sürecini kapsamaktadır. Amacımız, satış ve kâr ilişkisini anlamak ve gelecekteki kârı tahmin etmek.
# 

# %% [markdown]
# ## Veri Analizi
# 
# 
# ### Veri Temizleme ve Ön İşleme
# - Eksik değerlerin işlenmesi
# - Veri dönüşümleri
# 
# ### Analiz ve Görselleştirmeler
# - Satış ve kâr trendleri
# - Kârın aylık değişimi
# 

# %%
plt.figure(figsize=(10, 6))
monthly_profit.plot(color='green')
plt.title('Aylık Toplam Kâr')
plt.ylabel('Toplam Kâr')
plt.xlabel('Tarih')
plt.grid(True)
plt.show()


# %% [markdown]
# ## Modelleme
# 
# ### Kullanılan Modeller
# - Lineer Regresyon
# 
# ### Model Eğitim Süreci
# Kod parçaları ve eğitim süreci hakkında bilgi.
# 
# ### Model Performansı
# Performans değerlendirme sonuçları ve grafikler.
# 

# %%
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")



