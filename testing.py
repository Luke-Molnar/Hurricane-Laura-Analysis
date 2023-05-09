# Author: Xin (Bruce) Wu xwu03@villanova.edu
# Villanova University
# "Copyright 2023"

import pandas as pd
import scipy.stats as stats
from bioinfokit.analys import stat
from datetime import date, timedelta, datetime
import statsmodels.api as sm
from statsmodels.formula.api import ols
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 1. read hurricane data:
print("1. read hurricane data...")
# Create a dictionary of centroids, where the keys are the county names and the values are the centroid points
hurricane_df = pd.read_csv("data_hurricane.csv")
county2hurricane_dict = dict(zip(hurricane_df['CTFIPS'], hurricane_df['nb_affected_day']))

# 2. read evacuation data:
print("2. read evacuation data...")
evacuation_df = pd.read_csv("data_evacuation.csv")
county2evacuation_dict = dict(zip(evacuation_df['CTFIPS'], evacuation_df['ORDER']))

# 3. read county data
print("3. read county data...")
counties = pd.read_csv("data_counties.csv")
county_id2geometry_dict = dict(zip(counties['GEOID'], counties['geometry']))
# county_id2centroid_dict = dict(zip(counties['GEOID'], counties['centroid']))

# 4. create date objects for the start and end dates of the hurricane
print("4. create the start and end dates of the hurricane...")
hurricane_start_date = date(2020, 8, 23)
hurricane_end_date = date(2020, 8, 27)
print('start time:', hurricane_start_date, '...')
print('end time:', hurricane_end_date, '...')

# 5.1 create date objects for the start and end dates of the control group
# 8/23/2020 is a Sunday (the start date of the hurricane)
print("5.1 create the start and end dates of the control group (trips)...")
time_before = hurricane_start_date - timedelta(days=60)
time_after = hurricane_end_date + timedelta(days=60)
# Loop over the dates and print the day of the week
current_date = time_before
while current_date <= time_after:
    day_of_week = current_date.strftime('%A')
    print(current_date.strftime('%B %d, %Y'), ':', day_of_week)
    current_date = current_date + timedelta(days=1)

print("Time period of control group is from ", time_before, ' to', hurricane_start_date,
      " and from ", hurricane_end_date, ' to', time_after)

# 5.2 create date objects for the start and end dates of the control group
print("5.2 create the start and end dates of the control group (covid cases)...")
time_before_covid = hurricane_start_date - timedelta(days=9)
time_after_covid = hurricane_end_date + timedelta(days=14)

# 6. read trip data
print("6. read trip data...")
trip_df = pd.read_csv("data_trips.csv")

# Set the 'date' column as the DataFrame index
trip_df['date_index'] = pd.to_datetime(trip_df['date'])
trip_df.set_index('date_index', inplace=True)

# Select the rows within a date range
start_date = time_before
end_date = time_after
# Slice the DataFrame using .loc[] and .isin()
mask = trip_df.index.isin(pd.date_range(start_date, end_date))
focused_trip_df = trip_df.loc[mask].copy()

# add a new column to mark whether each date falls within the given range
focused_trip_df.loc[:, 'in_hurricane_range'] = \
    focused_trip_df.index.isin(pd.date_range(hurricane_start_date, hurricane_end_date))

# 7. hypothetical testing
print("7.1 hypothetical testing...")
iter_county_df = focused_trip_df.groupby('CTFIPS')  # group by county id
county_list = []
for county_id, subset in iter_county_df:
    print('start doing hypothetical testing for county:', str(county_id), '...')
    # county name
    county_name = subset.CTNAME[0]

    # state FIPs
    state_id = subset.STFIPS[0]

    # if the county is affected and how many days are affected:
    nb_affected_days = county2hurricane_dict[county_id]

    # evacuation order
    evacuation_order = county2evacuation_dict.setdefault(county_id, 'No evacuation order')

    # calculate mean and variance in population and samples
    pop_trip_mean = subset[subset.is_weekday]['Trips/person'].mean()
    pop_trip_var = subset[subset.is_weekday]['Trips/person'].var()
    sample_trip_mean = subset[subset.is_weekday & subset.in_hurricane_range]['Trips/person'].mean()

    # out of county trips
    pop_out_of_county_trip_mean = \
        subset[subset.is_weekday]['Out-of-county trips/person'].mean()
    sample_out_of_county_trip_mean = \
        subset[subset.is_weekday & subset.in_hurricane_range]['Out-of-county trips/person'].mean()

    # out of state trips
    pop_out_of_state_trip_mean = \
        subset[subset.is_weekday]['Out-of-county trips/person'].mean()
    sample_out_of_state_trip_mean = \
        subset[subset.is_weekday & subset.in_hurricane_range]['Out-of-state trips/person'].mean()

    # calculate miles
    pop_mile_mean = \
        subset[subset.is_weekday]['Miles/person'].mean()
    sample_mile_mean = \
        subset[subset.is_weekday & subset.in_hurricane_range]['Miles/person'].mean()

    # calculate % of working at home
    pop_working_at_home = \
        subset[subset.is_weekday]['% working from home'].mean()
    sample_working_at_home = \
        subset[subset.is_weekday & subset.in_hurricane_range]['% working from home'].mean()

    # Average cases and testing numbers before the end of hurricane
    mask = subset.index.isin(pd.date_range(time_before_covid, hurricane_end_date))
    covid_subset_before = subset.loc[mask]
    before_cases = covid_subset_before['New cases/1000 people'].mean()
    before_testing = covid_subset_before['Tests done/1000 people'].mean()

    # Average cases and testing numbers after the end of hurricane
    mask = subset.index.isin(pd.date_range(hurricane_end_date, time_after_covid))
    covid_subset_after = subset.loc[mask]
    after_cases = covid_subset_after['New cases/1000 people'].mean()
    after_testing = covid_subset_after['Tests done/1000 people'].mean()

    # case variation and test variations (please be attention the testing number is at state-level)
    case_difference = after_cases - before_cases
    test_difference = after_testing - before_testing

    # test 1: One sample t-test for trip per person
    # print('test 1')
    sample = trip_sample = subset[subset.is_weekday & subset.in_hurricane_range]['Trips/person'].to_list()
    t_stat_1samp_ptrip, p_value_1samp_ptrip = stats.ttest_1samp(a=sample, popmean=pop_trip_mean)

    # test 2: two sample t-test for trip per person
    # print('test 2')
    trip_population = subset[subset.is_weekday & (~subset.in_hurricane_range)]['Trips/person'].to_list()
    trip_sample = subset[subset.is_weekday & subset.in_hurricane_range]['Trips/person'].to_list()
    t_stat_2samp_ptrip, p_value_2samp_ptrip = stats.ttest_ind(a=trip_sample, b=trip_population)

    # test 3: two sample t-test for total trips
    # print('test 3')
    trip_population = subset[subset.is_weekday & (~subset.in_hurricane_range)]['Trips'].to_list()
    trip_sample = subset[subset.is_weekday & subset.in_hurricane_range]['Trips'].to_list()
    t_stat_2samp_ttrip, p_value_2samp_ttrip = stats.ttest_ind(a=trip_sample, b=trip_population)

    # test 4: two sample t-test for trips out of county
    # print('test 4')
    trip_population = subset[subset.is_weekday & (~subset.in_hurricane_range)]['Out-of-county trips/person'].to_list()
    trip_sample = subset[subset.is_weekday & subset.in_hurricane_range]['Out-of-county trips/person'].to_list()
    # perform two sample t-test
    t_stat_2samp_octrip, p_value_2samp_octrip = stats.ttest_ind(a=trip_sample, b=trip_population, alternative='less')

    # test 5: two sample t-test for trips out of states
    # print('test 5')
    trip_population = subset[subset.is_weekday & (~subset.in_hurricane_range)]['Out-of-state trips/person'].to_list()
    trip_sample = subset[subset.is_weekday & subset.in_hurricane_range]['Out-of-state trips/person'].to_list()
    t_stat_2samp_ostrip, p_value_2samp_ostrip = stats.ttest_ind(a=trip_sample, b=trip_population, alternative='less')

    # test 6: two sample t-test for % stay at home
    # print('test 6')
    workhome_population = subset[subset.is_weekday & (~subset.in_hurricane_range)]['% working from home'].to_list()
    workhome_sample = subset[subset.is_weekday & subset.in_hurricane_range]['% working from home'].to_list()
    t_stat_2samp_phome, p_value_2samp_phome = stats.ttest_ind(a=workhome_sample, b=workhome_population)

    # test 7: two sample t-test for miles
    # print('test 7')
    trip_population = subset[subset.is_weekday & (~subset.in_hurricane_range)]['Miles/person'].to_list()
    trip_sample = subset[subset.is_weekday & subset.in_hurricane_range]['Miles/person'].to_list()
    t_stat_2samp_pmile, p_value_2samp_pmile = stats.ttest_ind(a=trip_sample, b=trip_population)

    geometry = county_id2geometry_dict[county_id]

    each_county_list = [county_id, county_name, state_id, nb_affected_days, evacuation_order,
                        pop_trip_mean, pop_trip_var, sample_trip_mean,
                        pop_out_of_state_trip_mean, sample_out_of_county_trip_mean,
                        pop_out_of_state_trip_mean, sample_out_of_state_trip_mean,
                        pop_mile_mean, sample_mile_mean,
                        t_stat_1samp_ptrip, p_value_1samp_ptrip,
                        t_stat_2samp_ptrip, p_value_2samp_ptrip,
                        t_stat_2samp_ttrip, p_value_2samp_ttrip,
                        t_stat_2samp_octrip, p_value_2samp_octrip,
                        t_stat_2samp_ostrip, p_value_2samp_ostrip,
                        t_stat_2samp_pmile, p_value_2samp_pmile,
                        t_stat_2samp_phome, p_value_2samp_phome,
                        case_difference, after_cases, before_cases,
                        test_difference, after_testing, before_testing,
                        geometry, t_stat_2samp_ptrip > 0, nb_affected_days > 0]
    county_list.append(each_county_list)

hypothetical_df = pd.DataFrame(county_list)
hypothetical_df = hypothetical_df.rename(columns={0: 'CTFIPs',
                                                  1: 'CTNAME',
                                                  2: 'STFIPs',
                                                  3: 'nb_affected_days',
                                                  4: 'evacuation_order',
                                                  5: 'pop_trip_mean', 6: 'pop_trip_var', 7: 'sample_trip_mean',
                                                  8: 'pop_out_of_state_trip_mean', 9: 'sample_out_of_county_trip_mean',
                                                  10: 'pop_out_of_state_trip_mean', 11: 'sample_out_of_state_trip_mean',
                                                  12: 'pop_mile_mean', 13: 'sample_mile_mean',
                                                  14: 't_stat_1samp_person_trip', 15: 'p_value_1samp_person_trip',
                                                  16: 't_stat_2samp_person_trip', 17: 'p_value_2samp_person_trip',
                                                  18: 't_stat_2samp_total_trip', 19: 'p_value_2samp_total_trip',
                                                  20: 't_stat_2samp_out_ct_trip', 21: 'p_value_2samp_out_ct_trip',
                                                  22: 't_stat_2samp_out_st_trip', 23: 'p_value_2samp_out_st_trip',
                                                  24: 't_stat_2samp_person_mile', 25: 'p_value_2samp_person_mile',
                                                  26: 't_stat_2samp_perc_work_home', 27: 'p_value_2samp_perc_work_home',
                                                  28: 'case_difference', 29: 'after_cases', 30: 'before_cases',
                                                  31: 'test_difference', 32: 'after_testing', 33: 'before_testing',
                                                  34: 'geometry', 35: 'mobility_variation', 36: 'affected_hurricane'})

hypothetical_df.to_csv("output_hypothetical_test.csv")

w, pvalue = stats.shapiro(hypothetical_df.after_testing)
print(w, pvalue)

# Ordinary Least Squares (OLS) model
model = \
    ols('case_difference ~ C(mobility_variation)+C(affected_hurricane)+C(mobility_variation):C(affected_hurricane)',
        data=hypothetical_df).fit()
anova_table = sm.stats.anova_lm(model, type=2)
print(anova_table)
#
# res = stat()
# res.tukey_hsd(df=hypothetical_df, res_var='case_difference',
#               xfac_var='affected_hurricane',
#               anova_model=
#               'case_difference ~ C(affected_hurricane)')
# print(res.tukey_summary)

# res = stat()
# res.levene(df=hypothetical_df, res_var='case_difference', xfac_var='affected_hurricane')
# print(res.levene)


# test 8 compare affected counties and unaffected counties on covid cases differences
print("7.2 compare affected counties and unaffected counties on covid cases differences....")
subset_df_affected = hypothetical_df[hypothetical_df.nb_affected_days > 0]
subset_df_unaffected = hypothetical_df[hypothetical_df.nb_affected_days == 0]
case_difference_sample_affected = subset_df_affected['case_difference'].to_list()
case_difference_sample_unaffected = subset_df_unaffected['case_difference'].to_list()
t_stat_2samp_case, p_value_2samp_case = \
    stats.ttest_ind(a=case_difference_sample_affected, b=case_difference_sample_unaffected)
print("t-test:", t_stat_2samp_case)
print("p_value:", p_value_2samp_case)

# test 9 compare covid cases differences between counties with and without evacuation order
print("7.3 compare affected counties and unaffected counties on covid cases differences....")
subset_df_order = \
    hypothetical_df[(hypothetical_df.nb_affected_days > 0) &
                    ~(hypothetical_df.evacuation_order != 'No evacuation order')]
subset_df_no_order = \
    hypothetical_df[(hypothetical_df.nb_affected_days > 0) &
                    ~(hypothetical_df.evacuation_order == 'No evacuation order')]

case_difference_sample_affected = subset_df_order['case_difference'].to_list()
case_difference_sample_unaffected = subset_df_no_order['case_difference'].to_list()
t_stat_2samp_case_order, p_value_2samp_case_order = \
    stats.ttest_ind(a=case_difference_sample_affected, b=case_difference_sample_unaffected)
print("t-test:", t_stat_2samp_case_order)
print("p_value", p_value_2samp_case_order)
