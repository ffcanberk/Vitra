#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Excel dosyalarını oku
df_50_call = pd.read_excel("51_Call.xlsx")

df_50_call.set_index(df_50_call.columns[0], inplace=True)

df_50_call.index.name = 'DATES'
    
# 50 farklı dataframe oluştur
dfs = []
for column_name in df_50_call.columns[:]:  # İlk sütunu atla
    # Her bir Dependent değişken için yeni bir dataframe oluştur
    df = pd.DataFrame()
    
    df[column_name] = df_50_call[column_name]
    
    dfs.append(df)

# Oluşturulan 50 farklı dataframe'i göster
for i, df in enumerate(dfs):
    print(f"Dataframe {i+1}:")
    print(df.head())  # İlk beş satırı göstermek için
    print("\n")


# In[2]:


for df in dfs:
    # Eksik değerleri sütun ortalaması ile doldur
    df.fillna(df.mean(), inplace=True)


# In[3]:


dfs[0]


# In[4]:


import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error


# In[5]:


import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

all_predictions_df = pd.DataFrame()


for df in dfs:
    # Veriyi eğitim ve test olarak bölmek
    train_size = int(len(df) * 0.7)
    train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]
    
    # Son sütunun adını al
    target_column_name = df.columns[-1]

    # Ağırlıklı exponential smoothing uygulama
    model = ExponentialSmoothing(train_data.iloc[:, 0], trend='add', seasonal='add', damped_trend=True)
    fitted_model = model.fit(optimized=True)

    # Test verisi üzerinde modeli kullanma
    predictions = fitted_model.forecast(len(test_data))
    

    # Hata hesaplama
    mse = mean_squared_error(test_data.iloc[:, 0], predictions)
    print("Mean Squared Error:", mse)

    # Gerçek ve tahmin edilen değerleri gösterme
    results = pd.DataFrame({'Gerçek Değerler': test_data.iloc[:, 0].values,
                            'Tahmin Edilen Değerler {}'.format(train_data.columns[0]): predictions})
    
    all_predictions_df[f"{target_column_name} Model {i+1} Predictions"] = predictions
    
    
    print(results)
    
# DataFrame'i Excel dosyasına yaz
all_predictions_df.to_excel("tum_tahmin_degerleri_last_call2222.xlsx", index=False)


# In[6]:


import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

all_predictions_df = pd.DataFrame()

# dfs is assumed to be defined somewhere before this loop
for i, df in enumerate(dfs):
    # Veriyi eğitim ve test olarak bölmek
    train_size = int(len(df) * 0.7)
    train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]
    
    # Son sütunun adını al
    target_column_name = df.columns[-1]

    # Ağırlıklı exponential smoothing uygulama
    model = ExponentialSmoothing(train_data.iloc[:, 0], trend='add', seasonal='add', damped_trend=False)
    fitted_model = model.fit(optimized=True)

    # Test verisi üzerinde modeli kullanma
    predictions = fitted_model.forecast(len(test_data))
    
    predictions = predictions.clip(lower=0)

    # Hata hesaplama
    mse = mean_squared_error(test_data.iloc[:, 0], predictions)
    print("Mean Squared Error:", mse)

    # Gerçek ve tahmin edilen değerleri gösterme
    results = pd.DataFrame({'Gerçek Değerler': test_data.iloc[:, 0].values,
                            f'Tahmin Edilen Değerler {train_data.columns[0]}': predictions})
    
    all_predictions_df[f"{target_column_name} Model {i+1} Predictions"] = predictions
    
    print(results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, test_data.iloc[:, 0], label='Gerçek Değerler')
    plt.plot(test_data.index, predictions, label='Tahmin Edilen Değerler')
    plt.title('Gerçek ve Tahmin Edilen Değerler')
    plt.xlabel('Zaman')
    plt.ylabel('Değer')
    plt.legend()
    plt.show()
    


# In[7]:


import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error




    # Veriyi eğitim ve test olarak bölmek
train_size = int(len(dfs[2]) * 0.7)
train_data, test_data = dfs[2].iloc[:train_size], dfs[2].iloc[train_size:]
    
    # Son sütunun adını al
target_column_name = dfs[2].columns[-1]

    # Ağırlıklı exponential smoothing uygulama
model = ExponentialSmoothing(train_data.iloc[:, 0], trend='add', seasonal='add', damped_trend=True)
fitted_model = model.fit(optimized=True)

    # Test verisi üzerinde modeli kullanma
predictions = fitted_model.forecast(len(test_data))
    

    # Hata hesaplama
mse = mean_squared_error(test_data.iloc[:, 0], predictions)
print("Mean Squared Error:", mse)

    # Gerçek ve tahmin edilen değerleri gösterme
results = pd.DataFrame({'Gerçek Değerler': test_data.iloc[:, 0].values,
                            'Tahmin Edilen Değerler {}'.format(train_data.columns[0]): predictions})
    

    
    
print(results)
    


# In[8]:


import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error




    # Veriyi eğitim ve test olarak bölmek
train_size = int(len(dfs[48]) * 0.7)
train_data, test_data = dfs[48].iloc[:train_size], dfs[48].iloc[train_size:]
    
    # Son sütunun adını al
target_column_name = dfs[48].columns[-1]

    # Ağırlıklı exponential smoothing uygulama
model = ExponentialSmoothing(train_data.iloc[:, 0], trend='add', seasonal='add', damped_trend=True)
fitted_model = model.fit(optimized=True)

    # Test verisi üzerinde modeli kullanma
predictions = fitted_model.forecast(len(test_data))
    
predictions = predictions.clip(lower=0)

    # Hata hesaplama
mse = mean_squared_error(test_data.iloc[:, 0], predictions)
print("Mean Squared Error:", mse)

    # Gerçek ve tahmin edilen değerleri gösterme
results = pd.DataFrame({'Gerçek Değerler': test_data.iloc[:, 0].values,
                            'Tahmin Edilen Değerler {}'.format(train_data.columns[0]): predictions})
    

    
    
print(results)
    


# In[ ]:




