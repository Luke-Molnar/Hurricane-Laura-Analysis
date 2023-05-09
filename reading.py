# Author: Xin (Bruce) Wu xwu03@villanova.edu
# Villanova University
# "Copyright 2023"

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import scipy.stats as stats

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from datetime import date, timedelta, datetime

# 1. reading hurricane data:
# Create a dictionary of centroids, where the keys are the county names and the values are the centroid points
hurricane_df = pd.read_csv("input_hurricane_laura_rawdata3.csv")
hurricane_df['nb_affected_day'] = 0
for i in range(2, 8):
    hurricane_df['nb_affected_day'] = hurricane_df['nb_affected_day'] + hurricane_df.iloc[:, i]
county2hurricane_dict = dict(zip(hurricane_df['CTFIPS'], hurricane_df['nb_affected_day']))
# Use melt to unpivot the 'salary' and 'sales' columns
melted_hurricane_df = \
    pd.melt(hurricane_df, id_vars=['CTFIPS', 'CTNAME', 'nb_affected_day'], var_name='date', value_name='affected')
# Convert the 'date' column to datetime format
melted_hurricane_df['date_index'] = pd.to_datetime(melted_hurricane_df['date'])
melted_hurricane_df['pair_id_date'] = melted_hurricane_df.apply(lambda x: str(x.CTFIPS)+"_"+x.date, axis=1)
county_date_2hurricane_dict = dict(zip(melted_hurricane_df['pair_id_date'], melted_hurricane_df['affected']))
melted_hurricane_df.to_csv("data_hurricane.csv")

# 2. reading county map:
counties = gpd.read_file("cb_2018_us_county_20m/cb_2018_us_county_20m.shp")
# Check the current CRS of the GeoDataFrame
print(counties.crs)

# # Re-project the GeoDataFrame to a projected CRS (EPSG:3857)
gdf = counties.to_crs(epsg=3857)
counties['centroid'] = counties.centroid
counties['centroid'] = counties.centroid
counties['x_coord'] = counties['centroid'].x
counties['y_coord'] = counties['centroid'].y
counties.to_csv("data_counties.csv", index=False)

# 3 read the SERA data
trip_df = pd.read_csv("input_county_sera_results.csv")

trip_df = trip_df[trip_df['STFIPS'].isin([1, 5, 22, 28, 29, 40, 47, 48])]

# Convert the 'date' column to datetime format
trip_df['date_index'] = pd.to_datetime(trip_df['date'])

# Use .dt.weekday to determine the weekday of each date
trip_df['weekday'] = trip_df['date_index'].dt.weekday

# Create a new column indicating whether each date is a weekday or not
trip_df['is_weekday'] = (trip_df['weekday'] < 5)

# Select the attributes related to mobility
focused_trip_df = trip_df[
    {'CTFIPS', 'CTNAME', 'STFIPS', '% staying home', 'Trips/person', '% out-of-county trips',
     '% out-of-state trips', 'Miles/person', 'Work trips/person',
     'Non-work trips/person', 'Population',
     'date', 'weekday', 'is_weekday', 'New cases/1000 people',
     'Active cases/1000 people', '#days: decreasing COVID cases', 'Tests done/1000 people',
     '% working from home'}].copy()
focused_trip_df.loc[:, 'Trips'] = focused_trip_df['Trips/person'] * focused_trip_df['Population']
focused_trip_df.loc[:, 'Out-of-county trips/person'] = \
    focused_trip_df['Trips/person'] * focused_trip_df['% out-of-county trips'] / 100
focused_trip_df.loc[:, 'Out-of-state trips/person'] = \
    focused_trip_df['Trips/person'] * focused_trip_df['% out-of-state trips'] / 100

focused_trip_df.to_csv("data_trips.csv", index=False)
