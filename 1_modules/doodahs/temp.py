import time
import os
from datetime import datetime, timedelta

import amphora_client as amphora_client
from amphora_client.rest import ApiException
from amphora_client.configuration import Configuration

import json

configuration = Configuration()
configuration.host = "https://beta.amphoradata.com"

params = {'id': '57d6593f-1889-410a-b1fb-631b6f9c9c85',
        #'token': useful['API-token'],
        'start_time': datetime.now(),#(2019,11,4,11),#YYYY,MM,DD,hh,mm
        'end_time': datetime.now()+ timedelta(hours=24)#datetime(2019,12,6,11)
        }    
signal_dict = {'property': 'temperature'}
useful = {'API-token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJodHRwOi8vc2NoZW1hcy54bWxzb2FwLm9yZy93cy8yMDA1LzA1L2lkZW50aXR5L2NsYWltcy9uYW1lIjoiTUFSS1VTLkRPTExNQU5OQEVMSUlaQS5DT00uQVUiLCJodHRwOi8vc2NoZW1hcy54bWxzb2FwLm9yZy93cy8yMDA1LzA1L2lkZW50aXR5L2NsYWltcy9uYW1laWRlbnRpZmllciI6IjhiMjk1OWM3LTY0ZWUtNGQ1YS1hN2E0LTQ0NjA1NWQ0MjljZSIsImV4cCI6MTU3NTYxMTI5MiwiaXNzIjoiYW1waG9yYWRhdGEuY29tIiwiYXVkIjoiYW1waG9yYWRhdGEsY29tIn0.J3wbxxFb-91GcBD9BT1_I7Y5K0ZA3w-kAJx1xSG-zfI',
  'username': 'markus.dollmann@eliiza.com.au',
  'password': 'fobra1-penqef-Bebnuj'}


id = params['id']
type_ = '$event.'+str(signal_dict['property'])
start_time = params['start_time']
end_time = params['end_time']

# Create an instance of the API class
auth_api = amphora_client.AuthenticationApi(amphora_client.ApiClient(configuration))

token_request = amphora_client.TokenRequest(username='markus.dollmann@eliiza.com.au', password='fobra1-penqef-Bebnuj' ) 

# https://beta.amphoradata.com/Amphorae/Detail?id=57d6593f-1889-410a-b1fb-631b6f9c9c85
# id = "57d6593f-1889-410a-b1fb-631b6f9c9c85" 

try:
    # Gets a token
    t1_start = time.perf_counter()  
    #res = auth_api.authentication_request_token(token_request = token_request )
    t1_stop = time.perf_counter() 
    print("Elapsed time:", t1_stop - t1_start) # print performance indicator
    configuration.api_key["Authorization"] = "Bearer " + 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJodHRwOi8vc2NoZW1hcy54bWxzb2FwLm9yZy93cy8yMDA1LzA1L2lkZW50aXR5L2NsYWltcy9uYW1lIjoiTUFSS1VTLkRPTExNQU5OQEVMSUlaQS5DT00uQVUiLCJodHRwOi8vc2NoZW1hcy54bWxzb2FwLm9yZy93cy8yMDA1LzA1L2lkZW50aXR5L2NsYWltcy9uYW1laWRlbnRpZmllciI6IjhiMjk1OWM3LTY0ZWUtNGQ1YS1hN2E0LTQ0NjA1NWQ0MjljZSIsImV4cCI6MTU3NTYxMTI5MiwiaXNzIjoiYW1waG9yYWRhdGEuY29tIiwiYXVkIjoiYW1waG9yYWRhdGEsY29tIn0.J3wbxxFb-91GcBD9BT1_I7Y5K0ZA3w-kAJx1xSG-zfI'

    amphora_api = amphora_client.AmphoraeApi(amphora_client.ApiClient(configuration))
    print(f'Getting signals for: {amphora_api.amphorae_read(id).name}')
    signals = amphora_api.amphorae_get_signals(id)
    properties=list((s._property for s in signals))

    ts_api = amphora_client.TimeSeriesApi(amphora_client.ApiClient(configuration)) # the API for interacting with time series
    tomorrow = datetime.now() + timedelta(hours=24)

    time_range = amphora_client.DateTimeRange(_from = start_time, to= end_time)
    # get tomorrow's temperatures
    variable = amphora_client.NumericVariable( kind="numeric", 
        value=amphora_client.Tsx(tsx=type_), 
        aggregation=amphora_client.Tsx("avg($value)"))
    get_series = amphora_client.GetSeries([id], search_span= time_range, inline_variables={"temperature": variable})
    time_series_data = ts_api.time_series_query_time_series( amphora_client.QueryRequest(get_series= get_series))

    print(f'Got {len(time_series_data.timestamps)} datapoints and {len(time_series_data.properties)} properties')
    # access the data in time_series_data.properties
    print("-----------")

    # get average of tomorrow's rainfall probablility
    variable = amphora_client.AggregateVariable( kind="aggregate", 
        aggregation=amphora_client.Tsx("avg($event.rainProb)"))
    aggregate_series = amphora_client.AggregateSeries([id], 
        search_span= time_range, 
        inline_variables={"rainProbAvg": variable}, 
        interval= "PT24H") # 24 hour buckets
    time_series_data = ts_api.time_series_query_time_series( amphora_client.QueryRequest(aggregate_series= aggregate_series))
    event_count = next(value for value in time_series_data.properties if value.name == 'EventCount')
    rain_prob_average = next(value for value in time_series_data.properties if value.name == 'rainProbAvg')

    for i in range(len(time_series_data.timestamps)):
        print(f'For date: {time_series_data.timestamps[i]}')
        print(f'There has been {event_count.values[i]} events ingested')
        print(f'And an average rain probability of {rain_prob_average.values[i]} in 24 hour buckets')
        print("----------")
    

except ApiException as e:
    print("Exception when calling API: %s\n" % e)