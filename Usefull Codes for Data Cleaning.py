# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 23:29:33 2022

@author: fdcel
"""

import seaborn as sns
import pandas as pd

df=sns.load_dataset('iris')
dfa=pd.DataFrame()
dfb=pd.DataFrame
import numpy as np

np.random.randint(0,11,3)

from sklearn.datasets import make_blobs
# Generate feature matrix and target vector
features, target = make_blobs(n_samples = 100,
 n_features = 2,
 centers = 3,
 cluster_std = 0.5,
 shuffle = True,
 random_state = 1)

# Load library
import matplotlib.pyplot as plt
# View scatterplot
plt.scatter(features[:,0], features[:,1], c=target)
plt.show()


# Load library
import pandas as pd
# Create URL
url = 'https://tinyurl.com/titanic-csv'
# Load data
dataframe = pd.read_csv(url)
# female filtresi sağlama------------------------------------------
dataframe[dataframe['Sex'] == 'female']
# iki koşul ile filtereleme-------------------------------------
dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 65)]

# sütundaki değeri değiştirme-----------------------------------
dataframe['Sex'].replace("female", "Woman")

# sütundaki 2 değeri değiştirme-----------------------------------------
dataframe['Sex'].replace(["female", "male"], ["Woman", "Man"])

# tüm değerleri değiştirme------------------------------------------------
dataframe.replace(1, "One")

# sütun adı değiştirme------------------------------------------------
dataframe.rename(columns={'PClass': 'Passenger Class', 'Sex': 'Gender'})

dataframe[dataframe['Age'].isnull()]

#boş değerleri değiştirme seçili sütudnan--------------------------------
# Replace values with NaN
dataframe['Sex'] = dataframe['Sex'].replace('male', np.nan)

dataframe.drop(dataframe.columns[1:3], axis=1)
#çoklayan verileri atma-----------------------------------------------
dataframe.drop_duplicates()

# Show number of rows
print("Number Of Rows In The Original DataFrame:", len(dataframe))
print("Number Of Rows After Deduping:", len(dataframe.drop_duplicates()))

# Drop duplicates
dataframe.drop_duplicates(subset=['Sex'])

# Group rows by the values of the column 'Sex', calculate mean
# of each group
dataframe.groupby('Sex').mean()

# Group rows, calculate mean
dataframe.groupby(['Sex','Survived'])['Age'].mean()

#Zaman Serileri -----------------------------------------------------
# Create date range
time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')
# Create DataFrame
dataframe = pd.DataFrame(index=time_index)
# Create column of random values
dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)

# Select observations between two datetimes
dataframe[(dataframe['date'] > '2002-1-1 01:00:00') &
 (dataframe['date'] <= '2002-1-1 04:00:00')]

dataframe.loc['2002-1-1 01:00:00':'2002-1-1 04:00:00']

# Create features for year, month, day, hour, and minute
dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
dataframe['hour'] = dataframe['date'].dt.hour
dataframe['minute'] = dataframe['date'].dt.minute

# Group rows by week, calculate sum per week
dataframe.resample('W').sum()
# Group by two weeks, calculate mean
dataframe.resample('2W').mean()
# Group by month, count rows
dataframe.resample('M').count()
#------------------------------------------------------------------------------


# Group by month, count rows
dataframe.resample('M', label='left').count()

#döngüye alma değerleri ve fonksiyon uygulama----------------------------------
#loop to all elements in column
[name.upper() for name in dataframe['species'][0:2]]

# apply ile fonksiyon uygulama-------------------------------------------------
def uppercase(x):
 return x.upper()
dataframe['species'].apply(uppercase)[0:2]

# lambda ile fonkisyon---------------------------------------------------------
dataframe.groupby('species').apply(lambda x: x.count())
#concat---------------------------------------------
pd.concat([dfa, dfb], axis=1)

# Append row-------------------------------------------------------------------
row = pd.Series([10, 'Chris', 'Chillon'], index=['id', 'first', 'last'])
dfa.append(row, ignore_index=True)

# Merge DataFrames-------------------------------------------------------------
pd.merge(dfa, dfb, left_on='employee_id', right_on='employee_id')


# Load libraries
import numpy as np
import pandas as pd
# Create strings
date_strings = np.array(['03-04-2005 11:35 PM',
 '23-05-2010 12:01 AM',
'04-09-2009 09:09 PM'])

# data tipini değiştirme-------------------------------------------------------
# Convert to datetimes
[pd.to_datetime(date, format='%d-%m-%Y %I:%M %p') for date in date_strings]



#timezone değiştirme-----------------------------------------------------------
# Load library
import pandas as pd
# Create datetime
pd.Timestamp('2017-05-01 06:00:00', tz='Europe/London')

# Create datetime
date = pd.Timestamp('2017-05-01 06:00:00')
# Set time zone
date_in_london = date.tz_localize('Europe/London')

date_in_london.tz_convert('Africa/Abidjan')



# Load library
import pandas as pd
# Create data frame
dataframe = pd.DataFrame()
dataframe.loc['2002-1-1 01:00:00':'2002-1-1 04:00:00']

#aradaki süreyi hesaplama------------------------------------------------------
# Calculate duration between features
pd.Series(delta.days for delta in (dataframe['Left'] - dataframe['Arrived']))
# Create two datetime features
dataframe['Arrived'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-04-2017')]
dataframe['Left'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-06-2017')]
# Calculate duration between features
dataframe['Left'] - dataframe['Arrived']

#gün ay yıl dakika sütunları oluşturma-----------------------------------------
# Create features for year, month, day, hour, and minute
dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
dataframe['hour'] = dataframe['date'].dt.hour
dataframe['minute'] = dataframe['date'].dt.minute

#lag ekleme--------------------------------------------------------------------

# Load library
import pandas as pd
# Create data frame
dataframe = pd.DataFrame()
# Create data
dataframe["dates"] = pd.date_range("1/1/2001", periods=5, freq="D")
dataframe["stock_price"] = [1.1,2.2,3.3,4.4,5.5]
# Lagged values by one row
dataframe["previous_days_stock_price"] = dataframe["stock_price"].shift(1)

# eksik veriyi değiştirme

# Load libraries
import pandas as pd
import numpy as np
# Create date
time_index = pd.date_range("01/01/2010", periods=5, freq="M")
dataframe = pd.DataFrame(index=time_index)
# Create feature with a gap of missing values
dataframe["Sales"] = [1.0,2.0,np.nan,np.nan,5.0]
# Interpolate missing values
dataframe.interpolate()

# Forward-fill
dataframe.ffill()
# Back-fill
dataframe.bfill()
