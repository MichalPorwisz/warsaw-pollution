import os

from wwo_hist import retrieve_hist_data

os.chdir("../data_processed/forecasts")

frequency = 1
start_date = '25-DEC-2014'
end_date = '10-JAN-2020'
# TODO: Put your key
api_key = 'apikey'
location_list = ['warsaw']
# TODO: inside they filter out some columns - maybe try seeing all
hist_weather_data = retrieve_hist_data(api_key,
                                location_list,
                                start_date,
                                end_date,
                                frequency,
                                location_label = False,
                                export_csv = True,
                                store_df = True)
print('done')

# path like this because chdir changed directory
hist_weather_data[0].to_csv('./warsaw_full.csv')