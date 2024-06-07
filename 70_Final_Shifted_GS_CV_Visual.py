#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pandas as pd

# Excel dosyalarını oku
Vitra = pd.read_excel("Vitra_Forecast.xlsx")
Vitra.set_index(Vitra.columns[0], inplace=True)
Vitra


# In[2]:


import pandas as pd
import numpy as np
import pandas as pd

# Excel dosyalarını oku
df_index_r = pd.read_excel("Index_R.xlsx")
df_50_call = pd.read_excel("50_Call.xlsx")

# 50 farklı dataframe oluştur
dataframes = []
for column_name in df_50_call.columns[1:]:  # İlk sütunu atla
    # Her bir Dependent değişken için yeni bir dataframe oluştur
    df = pd.DataFrame()
    
    df[column_name] = df_50_call[column_name]
    
    # df_index_r ile birleştir
    df = pd.concat([df_index_r, df], axis=1)
    
    # İlk sütunu indeks olarak ayarla
    df.set_index(df.columns[0], inplace=True)
    
    df.loc['JAN 2024'] = np.nan
   
    df_shifted = df.shift(1)
    
    dataframes.append(df_shifted)

# Oluşturulan 50 farklı dataframe'i göster
for i, df in enumerate(dataframes):
    print(f"Dataframe {i+1}:")
    print(df.head())  # İlk beş satırı göstermek için
    print("\n")


# In[3]:


df_index_r.info()


# In[4]:


df_index_r.describe()


# In[5]:


# İlk sütunu indeks olarak ayarla
df_index_r.set_index(df_index_r.columns[0], inplace=True)

# df_50_call DataFrame'inin ikinci sütununu seçme
second_column = df_50_call.iloc[:, 1]  # 1, ikinci sütunun index numarasıdır

# İki DataFrame'i birleştirme
vis_df = pd.concat([df_index_r, second_column], axis=1)

vis_df.loc['JAN 2024'] = np.nan
   
f_shifted = vis_df.shift(1)

f_shifted.iloc[:, -1] = f_shifted.iloc[:, -1].shift(-1)
f_shifted.drop(f_shifted.index[-1], inplace=True)
f_shifted.drop(['FEB 2019'], inplace=True)

import matplotlib.pyplot as plt
import seaborn as sns

# Örnek bir DataFrame'in korelasyon matrisi üzerinde çalışıyorsunuz varsayalım
# df = ... (veri setiniz)

plt.figure(figsize=(20,20))
# Korelasyon matrisini çizdir
ax = sns.heatmap(f_shifted.corr(), annot=True, cmap='RdYlGn', square=True, 
                 annot_kws={"size": 14}, fmt=".2f") 

# X ve Y eksenlerindeki etiketlerin boyutunu ayarla
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=15)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45, fontsize=15)

# Göster
plt.show()


# In[6]:


import numpy as np

# Ağırlıkları tanımla
weights = np.array([0.25, 0.5, 0.75, 1])

# Her bir dataframe için işlemleri uygula ve sonuçları sakla
processed_dataframes = []

for df in dataframes:
    
    
    # 2. rolling(window=2, min_periods=2).sum() ve shift(periods=2) uygula
    df_roll_sum_shifted_2 = df.rolling(window=2, min_periods=2).sum()
    df_roll_sum_shifted_2.columns = [col + '_rolling_sum_1_shift_2' for col in df_roll_sum_shifted_2.columns]
    
    # 3. rolling(window=3, min_periods=3).sum() ve shift(periods=1) uygula
    df_roll_sum_shifted_1 = df.rolling(window=3, min_periods=3).sum()
    df_roll_sum_shifted_1.columns = [col + '_rolling_sum_2_shift_1' for col in df_roll_sum_shifted_1.columns]
    
    # 4. rolling(window=4, min_periods=4).sum() uygula
    df_roll_sum = df.rolling(window=4, min_periods=4).sum()
    df_roll_sum.columns = [col + '_rolling_sum_3' for col in df_roll_sum.columns]
    
    # 5. rolling(window=4, min_periods=4).apply(lambda x: (weights*x).sum()/np.sum(weights), raw=True) uygula
    df_roll_weighted_sum = df.rolling(window=4, min_periods=4).apply(lambda x: (weights*x).sum()/np.sum(weights), raw=True)
    df_roll_weighted_sum.columns = [col + '_wma_3' for col in df_roll_weighted_sum.columns]
    
    # 6. rolling(window=2, min_periods=2).min() ve shift(periods=2) uygula
    df_roll_min_shifted_2 = df.rolling(window=2, min_periods=2).min()
    df_roll_min_shifted_2.columns = [col + '_min_1_shift_2' for col in df_roll_min_shifted_2.columns]
    
    # 7. rolling(window=3, min_periods=3).min() ve shift(periods=1) uygula
    df_roll_min_shifted_1 = df.rolling(window=3, min_periods=3).min()
    df_roll_min_shifted_1.columns = [col + '_min_2_shift_1' for col in df_roll_min_shifted_1.columns]
    
    # 8. rolling(window=3, min_periods=3).min() uygula
    df_roll_min = df.rolling(window=4, min_periods=4).min()
    df_roll_min.columns = [col + '_min_3' for col in df_roll_min.columns]
    
    # 9. rolling(window=2, min_periods=2).max() ve shift(periods=2) uygula
    df_roll_max_shifted_2 = df.rolling(window=2, min_periods=2).max()
    df_roll_max_shifted_2.columns = [col + '_max_1_shift_2' for col in df_roll_max_shifted_2.columns]
    
    # 10. rolling(window=3, min_periods=3).max() ve shift(periods=1) uygula
    df_roll_max_shifted_1 = df.rolling(window=3, min_periods=3).max()
    df_roll_max_shifted_1.columns = [col + '_max_2_shift_1' for col in df_roll_max_shifted_1.columns]
    
    
    
    # Her bir işlem sonucunu sakla
    processed_dataframes.append([df_roll_sum_shifted_2, df_roll_sum_shifted_1, df_roll_sum, 
                                 df_roll_weighted_sum, df_roll_min_shifted_2, df_roll_min_shifted_1, 
                                 df_roll_min, df_roll_max_shifted_2, df_roll_max_shifted_1
                                 ])

# Her bir dataframe'in sonuçlarını içeren bir liste döndür
processed_dataframes


# In[7]:


import pandas as pd

# Yeni bir liste oluştur
combined_dataframes = []

# İki listenin uzunluğu aynı olmalıdır, aksi halde bir hata alabilirsiniz.
for i in range(min(len(processed_dataframes), len(dataframes))):
    # processed_dataframes içindeki i. öğeyi DataFrame haline getir
    processed_df = pd.concat(processed_dataframes[i], axis=1)
    
    # dataframes içindeki i. öğe zaten bir DataFrame
    original_df = dataframes[i]
    
    # İki DataFrame'i birleştir ve yeni DataFrame'i listeye ekle
    combined_df = pd.concat([processed_df, original_df], axis=1)
    combined_dataframes.append(combined_df)


# In[8]:


for df in combined_dataframes:
    # Son sütunu bir satır yukarı kaydır
    df.iloc[:, -1] = df.iloc[:, -1].shift(-1)
    df.drop(df.index[-1], inplace=True)


# In[9]:


for df in combined_dataframes:
    df.drop([ 'FEB 2019','MAR 2019', 'APR 2019'], inplace=True)


# In[10]:


for df in combined_dataframes:
    # Eksik değerleri sütun ortalaması ile doldur
    df.fillna(df.mean(), inplace=True)


# In[11]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

# StandardScaler nesnesini oluştur
scaler = StandardScaler()

# Her DataFrame için standartlaştırma işlemini uygula ve orijinal listeyi güncelle
for i in range(len(combined_dataframes)):
    # DataFrame'i seç
    df = combined_dataframes[i]
    
    # Son sütunu dışarıda bırak
    features = df.iloc[:, :-1]
    last_column = df.iloc[:, -1]
    
    # Özellikleri standartlaştır
    scaled_features = scaler.fit_transform(features)
    
    # Standartlaştırılmış özellikleri içeren yeni bir DataFrame oluştur ve orijinal DataFrame'in indekslerini kullan
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns, index=df.index)
    
    # Son sütunu orijinal haliyle geri ekle
    scaled_df[last_column.name] = last_column
    
    # Güncellenmiş DataFrame'i orijinal listeye yerleştir
    combined_dataframes[i] = scaled_df


# In[12]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

all_predictions_df = pd.DataFrame()
comparison_list = []  # Boş bir liste tanımla
aa = []

for i, df in enumerate(combined_dataframes):  # enumerate kullanarak her df için index (i) al
    # Bağımsız ve bağımlı değişkenleri ayır
    independent_variables = df.iloc[:, :-1]
    dependent_variable = df.iloc[:, -1]
    
    target_column_name = df.columns[-1]
    
    X = independent_variables
    y = dependent_variable
    
    # Eğitim ve test setlerine ayırma -- %30 Test için ayrıldı ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    # Model ve çapraz doğrulama ayarı
    model = Ridge()
    tscv = TimeSeriesSplit(n_splits=2)
    
    # Hiperparametre optimizasyonu
    param_grid = {'alpha': np.logspace(-1, 2, 10)}
    ridge_cv = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_train, y_train)

    # En iyi modeli ve seçilen özellikleri kullanarak eğitim
    model_best = ridge_cv.best_estimator_
    model_best.fit(X_train, y_train)
    
    # Tahmin ve performans değerlendirme
    y_pred = model_best.predict(X_test)
    
    config = y_pred/3  # Ölçekleme

    #for j in range(len(config)):
        #if y_pred[j] < 0:
            # Negatif değeri önceki üç tahminin ortalamasının yarısıyla değiştir
            #prev_3_preds = config[max(0, j-3):j]
            #replacement_value = 0.5 * prev_3_preds.mean() if len(prev_3_preds) > 0 else 0
            #config[j] = replacement_value  
            
    

       
    
    
    # Tahmin edilen değerleri DataFrame'e ekle
    predictions_series = pd.Series(config, index=X_test.index)
    all_predictions_df[f"{target_column_name} Model {i+1} Predictions"] = predictions_series
    
    # Her döngüde yeni bir comparison_df oluştur ve listeye ekle
    comparison_df = pd.DataFrame({f'{target_column_name} Gerçek Değerler': y_test, f'{target_column_name} EY Tahmin Edilen Değerler': config})
    comparison_list.append(comparison_df)  # Listeye ekle
    aa.append(comparison_df)
    
# comparison_list'i kontrol etmek için:
for df in comparison_list:
    print(df)

# DataFrame'i Excel dosyasına yaz


# In[13]:


for comparison_df in aa:
    # comparison_df'deki ilk sütunun adının ilk kelimesini alın
    first_column_name = comparison_df.columns[0]
    keyword = first_column_name.split()[0]

    # Vitra DataFrame'inde bu kelimeyi içeren bir sütun bulun
    matching_column = [col for col in Vitra.columns if keyword in col]
    
    if matching_column:
        # Sütun adı eşleşen ilk sütun olarak alınır (normalde tek bir eşleşme beklenir)
        column_to_add = matching_column[0]
        
        # Eşleşen sütunu, Vitra DataFrame'inden alıp comparison_df'e ekleyin.
        # Burada reindex_like kullanarak Vitra'dan alınan verilerin indexlerinin, comparison_df ile aynı olmasını sağlıyoruz.
        comparison_df[column_to_add] = Vitra[column_to_add].reindex_like(comparison_df)
        
        # Eklenen sütun için yeni ad oluştur ve güncelle
        new_column_name = f"{column_to_add} Vitra Tahmin Değerler"
        comparison_df.rename(columns={column_to_add: new_column_name}, inplace=True)
    else:
        print(f"{keyword} kelimesini içeren bir sütun Vitra DataFrame'inde bulunamadı.")


# In[14]:


# comparison_list içindeki tüm DataFrame'ler için ilk 6 satırı sil
for i in range(len(aa)):
    # İlk 6 satırı atla ve güncellenmiş DataFrame'i aynı indekste sakla
    aa[i] = aa[i].iloc[6:]


# In[15]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge
import seaborn as sns

# İki farklı for döngüsünü birleştirdim
for i, (df1, df2) in enumerate(zip(aa, combined_dataframes)):
    # Birinci grafik
    plt.figure(figsize=(10, 5))  # Grafik boyutunu ayarla
    
    for column in df1.columns:
        plt.plot(df1.index, df1[column], label=column)  # Her sütunu DataFrame'deki index ile çiz
    
    plt.title(f"Line Graph for DataFrame {i+1}")  # Grafik başlığını ekle
    plt.xlabel("Index")  # X ekseni başlığı
    plt.ylabel("Values")  # Y ekseni başlığı
    plt.legend()  # Açıklamaları ekle
    plt.grid(True)  # Izgara çizgilerini ekle
    plt.show()  # Grafikleri göster

    # İkinci kod parçasının görselleri
    
    all_predictions_df = pd.DataFrame()
    comparison_list = []

    independent_variables = df2.iloc[:, :-1]
    dependent_variable = df2.iloc[:, -1]

    target_column_name = df2.columns[-1]

    X = independent_variables
    y = dependent_variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    model = Ridge()
    tscv = TimeSeriesSplit(n_splits=2)

    param_grid = {'alpha': np.logspace(-1, 2, 10)}
    ridge_cv = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_train, y_train)

    model_best = ridge_cv.best_estimator_
    model_best.fit(X_train, y_train)

    y_pred = model_best.predict(X_test)

    config = y_pred / 3

    predictions_series = pd.Series(config, index=X_test.index)
    all_predictions_df[f"{target_column_name} Model {i+1} Predictions"] = predictions_series

    comparison_df = pd.DataFrame({
        f'{target_column_name} Gerçek Değerler': y_test,
        f'{target_column_name} EY Tahmin Edilen Değerler': config
    })
    comparison_list.append(comparison_df)

    coefficients = pd.Series(model_best.coef_, index=X.columns)
    top_coefficients = coefficients.abs().sort_values(ascending=False).head(5)

    top_features = coefficients.nlargest(5).index.tolist()
    correlation_matrix = df2[top_features + [target_column_name]].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title(f'Heatmap for Model {i+1}')
    plt.show()

    print(f"Model {i+1} Top Coefficients:")
    print(top_coefficients)
    print("\n")


# In[ ]:




