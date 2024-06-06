import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import altair as alt
import numpy as np
import folium
from streamlit_folium import st_folium
from scipy.optimize import curve_fit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Data Preprocessing 
df = pd.read_csv("2023-10-01-2024-05-17-Middle_East-Israel-Palestine.csv")
df = df.drop_duplicates(subset=['event_id_cnty'], keep='last')
df['event_date'] = pd.to_datetime(df['event_date'])
df.sort_values(by='event_date', inplace=True)

# Daily Fatalities
daily_df = df[['event_date','fatalities','country']].groupby(['event_date','country']).sum().reset_index()
daily_df_pivot = daily_df.pivot(index='event_date', columns='country', values='fatalities')
start_date = daily_df_pivot.index.min()
end_date = daily_df_pivot.index.max()
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
daily_df_pivot = daily_df_pivot.reindex(date_range).fillna(0)
daily_df_pivot['7-day average'] = daily_df_pivot.rolling(7).mean().sum(axis=1)


# Cumulative Fatalities
df_grouped = df.groupby(['event_date', 'country']).sum().reset_index()
df_grouped['cumulative_fatalities'] = df_grouped[['country', 'fatalities']].groupby(['country']).cumsum()
pivot_df_cumsum = df_grouped.pivot(index='event_date', columns='country', values='cumulative_fatalities')
start_date = pivot_df_cumsum.index.min()
end_date = pivot_df_cumsum.index.max()
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
pivot_df_cumsum = pivot_df_cumsum.reindex(date_range).fillna(method='ffill')


# Map-wise Visualization
df_deaths = df[df['fatalities'] > 0]
df_deaths_gp = df[['longitude', 'latitude','fatalities']].groupby(['longitude', 'latitude']).sum().reset_index()
unique_loc = df[['longitude', 'latitude','country']].drop_duplicates()


# Visualising Protests
df_protests = df[df['event_type'] == 'Protests']
text_to_num = { # ... (your text_to_num dictionary) }

def convert_to_number(text): # ... (your convert_to_number function)

df_protests['crowd_size'] = df_protests['tags'].apply(convert_to_number)
daily_protests = df_protests[['event_date','crowd_size','country']].groupby(['event_date','country']).sum().reset_index()
daily_protests_pivot = daily_protests.pivot(index='event_date', columns='country', values='crowd_size')
start_date = daily_protests_pivot.index.min()
end_date = daily_protests_pivot.index.max()
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
daily_protests_pivot = daily_protests_pivot.reindex(date_range).fillna(0)



# Streamlit App

st.title('Israel-Palestine Conflict Analysis 2023-2024')

# Daily Fatalities
st.header('Daily Reported Fatalities')
st.line_chart(daily_df_pivot)

# Annotate October 7th
max_israel_fatalities_date = daily_df_pivot["Israel"].idxmax()
max_israel_fatalities = daily_df_pivot["Israel"].max()
st.markdown(f"Peak fatalities in Israel occurred on {max_israel_fatalities_date}, with {max_israel_fatalities} deaths.")

# Ceasefire Annotation
st.markdown("**Note:** The red shaded region indicates a period of ceasefire.")

# Cumulative Fatalities
st.header('Cumulative Reported Fatalities')
st.line_chart(pivot_df_cumsum)

# Types of Events
st.header('Types of Events')
st.bar_chart(df.event_type.value_counts())

# Disorders Type
st.header('Disorder Types')
st.bar_chart(df.disorder_type.value_counts())


# Map Visualization
st.header('Map of Fatalities')
m = folium.Map(location=[31.5, 34.5], tiles="CartoDB Positron", zoom_start=9)

def add_points_to_map(gdf, map_obj):
    for _, row in gdf.iterrows():

        country = unique_loc[(unique_loc['latitude'] == row['latitude']) & (unique_loc['longitude'] == row['longitude'])]['country'].values[0]
        if country == 'Palestine':
            colour_in = 'red'
        else:
            colour_in = 'blue'

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=row['fatalities'] / 10, 
            color=colour_in,
            fill=True,
            fill_color=colour_in
        ).add_to(map_obj)

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(df_deaths_gp, geometry=gpd.points_from_xy(df_deaths_gp.longitude, df_deaths_gp.latitude))

# Add points to the map
add_points_to_map(gdf, m)
st_folium(m, width=700, height=500)


# Protests Visualization
st.header('Protests')

st.line_chart(daily_protests_pivot)

# Annotations for protest events
max_israel_protests_date = daily_protests_pivot["Israel"].idxmax()
max_israel_protests = daily_protests_pivot["Israel"].max()
st.markdown(f"Peak protests in Israel occurred on {max_israel_protests_date}, with an estimated {int(max_israel_protests)} attendees.")
st.markdown(f"**Note:** Large protests were recorded 50 and 100 days after the conflict started on October 7, 2023.")


# Experimental Modeling
st.header('Experimental Modeling')
st.write("This section contains experimental modeling that may not be reliable for forecasting. It's provided for informational purposes only.")
# .... (your experimental modeling code with streamlit integration)
