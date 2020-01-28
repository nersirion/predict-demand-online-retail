import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm_notebook as tqdm

def first_change(df):
    #Первое преобразование данных для болле удобного features creation
    df['Date_by_day'] = df['InvoiceDate'].dt.date
    df['Month'] = df['InvoiceDate'].dt.month
    df['Week'] = df['InvoiceDate'].dt.week
    df['Day_of_week'] = df['InvoiceDate'].dt.dayofweek
    df['Hour'] = df['InvoiceDate'].dt.hour
    #StockCode, которые не относятся к товарам
    SC_to_drop=['M', 'D', 'POST', 'DOT', 'gift_0001_80', 'BANK CHARGES', 'ADJUST']
    df = df[(df['Price']<1000) &(~df['StockCode'].isin(SC_to_drop))].dropna() 
    
    def group_by_price(x):
        if x<=1:
            x='Lowest price'
        elif 1<x<1.5:
            x='Low price'
        elif 1.5<=x<2.5:
            x='Fine price'      
        elif 20>x>=2.5:
            x='Normal price'
        else:
            x='High price'
            
        return x
    df['Group_by_Price'] = df['Price'].apply(group_by_price)
    
    #Create dict 
    features_for_demand = dict(set(zip(df['Country'].unique(), np.zeros(len(df['Country'].unique())))))
    features_for_demand.update(dict(set(zip(df['Country'].unique()+'_sales', np.zeros(len(df['Country'].unique()))))))
    features_for_demand.update(dict(set(zip(df['Country'].unique()+'_avg_sales', np.zeros(len(df['Country'].unique()))))))
    features_for_demand.update(dict(set(zip(df['Hour'].astype(str).unique()+'_hours', np.zeros(len(df['Hour'].unique()))))))
    features_for_demand.update(dict(set(zip(df['Group_by_Price'].unique(), np.zeros(len(df['Group_by_Price'].unique()))))))
    
    
    return df, features_for_demand

def create_features_for_demand(retail, features_for_demand):
    '''retail - DataFrame c данными ритейла и features - предварительно созданный словарь.
    Первоночальная групировка идет по каждому дню. 
    
    Полученный датасет позволяет обобщеный по пользователем EDA, ориентируясь, например, на страну. 
    Также его можно использовать для прогнозирования спроса.'''
    data = []
    features_for_demand['sum_sales']=0
    features_for_demand['avg_sale']=0
    features_for_demand['Quantity']=0
    features_for_demand['nunique_products']=0
    features_for_demand['Month'] = 0
    features_for_demand['count_customers'] = 0
    features_for_demand['Avg_quantity'] = 0
    count=0
    for i, day in tqdm(retail.groupby('Date_by_day', sort=False)):
        features=features_for_demand.copy()
       
        
        sum_sales = day['Price'].sum()
        quantity = day['Quantity'].sum()
        avg_sale = sum_sales/len(day)*quantity
        features['count_customers']=day['Customer ID'].nunique()
        features['Quantity'] = quantity
        features['Avg_quantity'] = features['Quantity']/day['Customer ID'].nunique()
        
        features['sum_sales']=avg_sale
        features['avg_sale']=avg_sale/day['Customer ID'].nunique()
        nunique_products = day['Description'].nunique()
        features['nunique_products']=nunique_products
        
        for i, country_df in day.groupby('Country', sort=False):
            country = country_df['Country'].iloc[0]
            features[country]=country_df['Customer ID'].nunique()
            country_sales = country_df['Price'].sum()/len(country_df)*country_df['Quantity'].sum()
            features[f'{country}_sales']=country_sales
            features[f'{country}_avg_sales']=country_sales/features[country]
            
        features['day_of_week'] = day['Day_of_week'].iloc[0]
        features['Day'] = day['Date_by_day'].iloc[0]
        features['Month'] = day['Month'].iloc[0]
        
        hours_counter = Counter(day['Hour'])
        for key in hours_counter.keys():
            features[str(key)+'_hours']=hours_counter[key]
            
        group_price_counts = Counter(day['Group_by_Price'])
        for key in group_price_counts.keys():
            features[key]=group_price_counts[key]
        
        
        data.append(features)
    retail_by_day = pd.DataFrame(data)    
    return retail_by_day



def create_features_by_customer(df):
    '''df - начальные данные ритейла, т.к. используется весь датасет, то словарь признаков генерируется внутри функции.
    Полученный датасет отоброжает данные агрегированные вокруг пользователя. Основной набор признаков составляют
    первые 300 самых распространненых товаров. Остальные товары отнесены в категорию Other. 
    
    Этот датасет может служить для нескольких целей машиного обучения:
        1. Прогнозировать послюдущую покупку, исходя из коллаборативной фильтрации, что позволяет не давать скидку на то,
        что и так бы купил клиент. Если же он не купил этот товар, то высока вероятность дожать покупку скидкой.
        2. Постараться рассчитать вероятность возрата покупки клиентом.
        
    В исходном датасете только данные итоговых транзакций, поэтому серьёзных успехов добиться, скорее всего, не получится,
    так как нет огромного пласта информации о поведении клиента на сайте до покупки.'''
    features_customer = dict(set(zip(df['Hour'].astype(str).unique()+'_hours', np.zeros(len(df['Hour'].unique())))))
    features_customer.update(dict(set(zip(df['Group_by_Price'].unique(), np.zeros(len(df['Group_by_Price'].unique()))))))
    features_customer.update(dict(set(zip(df['Description'].value_counts().index[:300], np.zeros(300)))))
    data=[]
    features_customer['Other'] = 0
    
    
    for i, customer in tqdm(df.groupby('Customer ID', sort=False)):
        
        features=features_customer.copy()
       
        quantity = customer['Quantity'].sum()
        features['Country']=customer['Country'].iloc[0]
        features['Quantity'] = customer['Quantity'].sum()
        #Нулевая или отрицательная сумма покупки говорит о возрате
        features['sum_sales'] = customer['Price'].sum()/len(customer)*quantity
              
        features['Customer ID'] = customer['Customer ID']
        hours_counter = Counter(customer['Hour'])
        for key in hours_counter.keys():
            features[str(key)+'_hours']=hours_counter[key]
            
        products_counter = Counter(customer['Description'])
        for key in products_counter.keys():
            if key in features:
                features[key]=products_counter[key]
            else:
                features['Other']+=products_counter[key]
             
            
        group_price_counts = Counter(customer['Group_by_Price'])
        for key in group_price_counts.keys():
            features[key]=group_price_counts[key]  
            
        last_product = customer['Description'].iloc[-1]#Последняя покупка и цель для модели
        if last_product in features:
            features['last_buy']=last_product
        else:
            features['last_buy']='Other'
      
        data.append(features)
    customers_data = pd.DataFrame(data)    
    return customers_data
            
