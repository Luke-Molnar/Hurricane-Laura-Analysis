# Author: Xin (Bruce) Wu xwu03@villanova.edu
# Villanova University
# "Copyright 2023"

import matplotlib.pyplot as plt
from datetime import timedelta, datetime, date

import pandas as pd


def _county_attributes_mapping(county_fips, date_str, sd, ed, attributes):
    date_index = datetime.strptime(date_str, "%m/%d/%Y")
    if (attributes == 'evacuation') & (date_index <= ed) & (date_index >= sd):
        value = county2evacuation_dict[county_fips]
    elif (attributes == 'hurricane') & (date_index <= ed) & (date_index >= sd):
        value = county2hurricane_dict[county_fips]
    elif (attributes == 'hurricane_date') & (date_index <= ed) & (date_index >= sd):
        value = county_date_2hurricane_dict[str(county_fips) + '_' + date_str]
    elif (attributes == 't_testing') & (date_index <= ed) & (date_index >= sd):
        value = county2t_test_dict[county_fips]
    else:
        value = 'N/A'
    return value


# 1. read hurricane data:
print("1. read hurricane data...")
# Create a dictionary of centroids, where the keys are the county names and the values are the centroid points
hurricane_df = pd.read_csv("data_hurricane.csv")
county2hurricane_dict = dict(zip(hurricane_df['CTFIPS'], hurricane_df['nb_affected_day']))
county_date_2hurricane_dict = dict(zip(hurricane_df['pair_id_date'], hurricane_df['affected']))

# 2.read t-testing results
hypothetical_df = pd.read_csv("output_hypothetical_test.csv")
county2evacuation_dict = dict(zip(hypothetical_df['CTFIPs'], hypothetical_df['evacuation_order']))

# 3 read trips
trip_df = pd.read_csv("data_trips.csv")
# Set the 'date' column as the DataFrame index
trip_df['date_index'] = pd.to_datetime(trip_df['date'])
trip_df.set_index('date_index', inplace=True)
focused_trip_df = trip_df.copy()

# 4. create date objects for the start and end dates of the hurricane
print("2. create the start and end dates of the hurricane...")
hurricane_start_date = date(2020, 8, 23)
hurricane_end_date = date(2020, 8, 27)
print('start time:', hurricane_start_date, '...')
print('end time:', hurricane_end_date, '...')
time_before = hurricane_start_date - timedelta(days=60)
time_after = hurricane_end_date + timedelta(days=60)

# 5. plot the figures for unaffected counties

print("3. plotting")
subset_df = hypothetical_df[hypothetical_df.nb_affected_days > 0]
affected_counties_list = subset_df.CTFIPs.to_list()
county2t_test_dict = dict(zip(hypothetical_df.CTFIPs, hypothetical_df.t_stat_2samp_person_trip))
group1_subset_df = subset_df[subset_df.t_stat_2samp_person_trip >= 0]
group1_counties_list = group1_subset_df.CTFIPs.to_list()
group2_subset_df = subset_df[subset_df.t_stat_2samp_person_trip < 0]
group2_counties_list = group2_subset_df.CTFIPs.to_list()

# Convert to datetime object
hurricane_start_date = datetime.combine(hurricane_start_date, datetime.min.time())
hurricane_end_date = datetime.combine(hurricane_end_date, datetime.min.time())
focused_trip_df_copy = focused_trip_df.copy()

# Use .loc to modify the original DataFrame in-place
focused_trip_df_copy.loc[:, 'group1_test>=0'] = \
    focused_trip_df.apply(lambda x: 1 if x.CTFIPS in group1_counties_list else 0, axis=1)

focused_trip_df_copy.loc[:, 'group1_test<0'] = \
    focused_trip_df.apply(lambda x: 1 if x.CTFIPS in group2_counties_list else 0, axis=1)

focused_trip_df_copy.loc[:, 'evacuation'] = \
    focused_trip_df.apply(lambda x: _county_attributes_mapping(x.CTFIPS, x.date, hurricane_start_date,
                                                               hurricane_end_date, 'evacuation'), axis=1)

focused_trip_df_copy.loc[:, 'nb_day_affected_hurricane'] = \
    focused_trip_df.apply(lambda x: _county_attributes_mapping(x.CTFIPS, x.date, hurricane_start_date,
                                                               hurricane_end_date, 'hurricane'), axis=1)

focused_trip_df_copy.loc[:, 'if_affected_hurricane'] = \
    focused_trip_df.apply(lambda x: _county_attributes_mapping(x.CTFIPS, x.date, hurricane_start_date,
                                                               hurricane_end_date, 'hurricane_date'), axis=1)

focused_trip_df_copy.loc[:, 't_testing'] = \
    focused_trip_df.apply(lambda x: _county_attributes_mapping(x.CTFIPS, x.date, hurricane_start_date,
                                                               hurricane_end_date, 't_testing'), axis=1)

focused_trip_df_copy.to_csv("output_focused_data.csv")

start_date = time_before + timedelta(days=30)
end_date = time_after - timedelta(days=30)
# Slice the DataFrame using .loc[] and .isin()
mask = focused_trip_df_copy.index.isin(pd.date_range(start_date, end_date))
focused_trip_df_copy = focused_trip_df_copy.loc[mask]
avg_trips_df = focused_trip_df_copy.groupby(['date'])['Trips/person'].mean()
avg_trips_df.plot(kind='line', color='blue')
# Add labels and title
plt.xlabel('Date')
plt.ylabel('Number of trips')
plt.title('Time-Dependent Trip Data')
plt.show()

focused_trip_df_copy_unaffected = focused_trip_df_copy[(focused_trip_df_copy['nb_day_affected_hurricane'] == 0) |
                                                       (focused_trip_df_copy['nb_day_affected_hurricane'] == 'N/A')]
avg_case = focused_trip_df_copy_unaffected.groupby(['date'])['New cases/1000 people'].mean().rolling(window=7).mean()
avg_case.plot(kind='line', color='blue')
# Add labels and title
plt.xlabel('Group 1 - Date')
plt.ylabel('Number of cases')
plt.title('Time-Dependent Case Data')
# Set the xticks to show every x value
plt.xticks(range(len(avg_case)), avg_case.index)
# Rotate the xtick labels by 45 degrees
plt.xticks(rotation=90)
# Add grid lines to the plot
plt.grid(True)
plt.show()

focused_trip_df_copy_group1 = focused_trip_df_copy[focused_trip_df_copy['group1_test>=0'] == 1]
avg_case = focused_trip_df_copy_group1.groupby(['date'])['New cases/1000 people'].mean().rolling(window=7).mean()
avg_case.plot(kind='line', color='blue')
# Add labels and title
plt.xlabel('Group 1 - Date')
plt.ylabel('Number of cases')
plt.title('Time-Dependent Case Data')
# Set the xticks to show every x value
plt.xticks(range(len(avg_case)), avg_case.index)
# Rotate the xtick labels by 45 degrees
plt.xticks(rotation=90)
# Add grid lines to the plot
plt.grid(True)
plt.show()

focused_trip_df_copy_group2 = focused_trip_df_copy[focused_trip_df_copy['group1_test<0'] == 1]
avg_case = focused_trip_df_copy_group2.groupby(['date'])['New cases/1000 people'].mean().rolling(window=7).mean()
avg_case.plot(kind='line', color='blue')
# Add labels and title
plt.xlabel('Group 2 - Date')
plt.ylabel('Number of cases')
plt.title('Time-Dependent Case Data')
# Set the xticks to show every x value
plt.xticks(range(len(avg_case)), avg_case.index)
# Rotate the xtick labels by 45 degrees
plt.xticks(rotation=90)
# Add grid lines to the plot
plt.grid(True)
plt.show()
print('END')
