#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_excel("C:\\Users\\Anuradha jha\\Downloads\\real_time_weather_dataset (1).xlsx")




temp_columns = [col for col in df.columns if col.startswith('Temp_')]
df['Average_Temperature'] = df[temp_columns].mean(axis=1)


df['Humidity'] = df['Humidity_1']   
df['Wind Speed'] = df['WindSpeed_1'] 

df['Time'] = range(len(df))


plot_data = df[['Time', 'Average_Temperature', 'Humidity', 'Wind Speed']].head(50)


plt.figure(figsize=(12, 6))
sns.lineplot(data=plot_data, x='Time', y='Average_Temperature', label='Temperature (°C)', marker='o')
sns.lineplot(data=plot_data, x='Time', y='Humidity', label='Humidity (%)', marker='s')
sns.lineplot(data=plot_data, x='Time', y='Wind Speed', label='Wind Speed (km/h)', marker='^')

plt.title('Real-Time Weather Conditions')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_excel("C:\\Users\\Anuradha jha\\Downloads\\real_time_weather_dataset (1).xlsx")


temp_columns = [col for col in df.columns if col.startswith('Temp_')]
df['Average_Temperature'] = df[temp_columns].mean(axis=1)

df['Time'] = range(len(df))

short_avg = df['Average_Temperature'].tail(5).mean()
short_forecast = [short_avg] * 5
short_time = list(range(len(df), len(df) + 5))

long_avg = df['Average_Temperature'].tail(30).mean()
long_forecast = [long_avg] * 30
long_time = list(range(len(df), len(df) + 30))

plt.figure(figsize=(12, 6))
plt.plot(df['Time'], df['Average_Temperature'], label='Historical Temperature', marker='o')
plt.plot(short_time, short_forecast, label='Short-Term Forecast (Next 5)', marker='s', color='orange')
plt.plot(long_time, long_forecast, label='Long-Term Forecast (Next 30)', linestyle='--', color='green')

plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.title('Weather Forecast Using Moving Average')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[25]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_excel("C:\\Users\\Anuradha jha\\Downloads\\real_time_weather_dataset (1).xlsx")

temp_cols = [col for col in df.columns if col.startswith('Temp_')]
df['Average_Temperature'] = df[temp_cols].mean(axis=1)

df['Humidity'] = df['Humidity_1']
df['Wind Speed'] = df['WindSpeed_1']

df['Time'] = range(len(df))


plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Time', y='Average_Temperature', label='Avg Temperature', color='red')
sns.lineplot(data=df, x='Time', y='Humidity', label='Humidity', color='blue')
sns.lineplot(data=df, x='Time', y='Wind Speed', label='Wind Speed', color='green')
plt.title('Climate Trend Over Time')
plt.xlabel('Time')
plt.ylabel('Weather Metrics')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_excel("C:\\Users\\Anuradha jha\\Downloads\\real_time_weather_dataset (1).xlsx")

temp_cols = [col for col in df.columns if col.startswith('Temp_')]
df['Average_Temperature'] = df[temp_cols].mean(axis=1)


df['Humidity'] = df['Humidity_1']
df['Wind Speed'] = df['WindSpeed_1']
df['Time'] = range(len(df))  


sns.set_theme(style="whitegrid")


fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)


sns.lineplot(ax=axs[0], data=df, x='Time', y='Average_Temperature', color='red')
axs[0].set_title("🌡️ Temperature Trend Over Time", fontsize=14)
axs[0].set_ylabel("Temperature (°C)")


sns.lineplot(ax=axs[1], data=df, x='Time', y='Humidity', color='blue')
axs[1].set_title("💧 Humidity Levels Over Time", fontsize=14)
axs[1].set_ylabel("Humidity (%)")


sns.lineplot(ax=axs[2], data=df, x='Time', y='Wind Speed', color='green')
axs[2].set_title("🍃 Wind Speed Over Time", fontsize=14)
axs[2].set_ylabel("Speed (km/h)")
axs[2].set_xlabel("Time")

plt.tight_layout()
plt.show()


# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_excel("C:\\Users\\Anuradha jha\\Downloads\\real_time_weather_dataset (1).xlsx")


temp_cols = [col for col in df.columns if col.startswith('Temp_')]
df['Temperature'] = df[temp_cols].mean(axis=1)


df['Humidity'] = df['Humidity_1']


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Temperature', y='Humidity', color='orange', s=60)

plt.title('🌡️ Temperature vs 💧 Humidity')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




