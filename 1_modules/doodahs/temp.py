import time
import os
from datetime import datetime, timedelta

import amphora_client as a10a
from amphora_client.rest import ApiException
from amphora_client.configuration import Configuration

import json

configuration = Configuration()
configuration.host = "https://beta.amphoradata.com"

# Create an instance of the API class
auth_api = a10a.AuthenticationApi(a10a.ApiClient(configuration))

token_request = a10a.TokenRequest(username='markus.dollmann@eliiza.com.au', password='fobra1-penqef-Bebnuj') 

# https://beta.amphoradata.com/Amphorae/Detail?id=57d6593f-1889-410a-b1fb-631b6f9c9c85
id = "57d6593f-1889-410a-b1fb-631b6f9c9c85" 

try:
    # Gets a token
    t1_start = time.perf_counter()  
    res = auth_api.authentication_request_token(token_request = token_request )
    t1_stop = time.perf_counter() 
    print("Elapsed time:", t1_stop - t1_start) # print performance indicator
    configuration.api_key["Authorization"] = "Bearer " + res

    amphora_api = a10a.AmphoraeApi(a10a.ApiClient(configuration))
    print(f'Getting signals for: {amphora_api.amphorae_read(id).name}')
    signals = amphora_api.amphorae_get_signals(id)
    properties=list((s._property for s in signals))

    ts_api = a10a.TimeSeriesApi(a10a.ApiClient(configuration)) # the API for interacting with time series
    tomorrow = datetime.now() + timedelta(hours=24)

    time_range = a10a.DateTimeRange(_from = datetime.now(), to= tomorrow)
    # get tomorrow's temperatures
    variable = a10a.NumericVariable( kind="numeric", 
        value=a10a.Tsx(tsx="$event.temperature"), 
        aggregation=a10a.Tsx("avg($value)"))
    get_series = a10a.GetSeries([id], search_span= time_range, inline_variables={"temperature": variable})
    time_series_data = ts_api.time_series_query_time_series( a10a.QueryRequest(get_series= get_series))
    print(time_series_data)
    print(f'Got {len(time_series_data.timestamps)} datapoints and {len(time_series_data.properties)} properties')
    # access the data in time_series_data.properties
    print("-----------")

    # get average of tomorrow's rainfall probablility
    variable = a10a.AggregateVariable( kind="aggregate", 
        aggregation=a10a.Tsx("avg($event.rainProb)"))
    aggregate_series = a10a.AggregateSeries([id], 
        search_span= time_range, 
        inline_variables={"rainProbAvg": variable}, 
        interval= "PT24H") # 24 hour buckets
    time_series_data = ts_api.time_series_query_time_series( a10a.QueryRequest(aggregate_series= aggregate_series))
    event_count = next(value for value in time_series_data.properties if value.name == 'EventCount')
    rain_prob_average = next(value for value in time_series_data.properties if value.name == 'rainProbAvg')

    for i in range(len(time_series_data.timestamps)):
        print(f'For date: {time_series_data.timestamps[i]}')
        print(f'There has been {event_count.values[i]} events ingested')
        print(f'And an average rain probability of {rain_prob_average.values[i]} in 24 hour buckets')
        print("----------")
    

except ApiException as e:
    print("Exception when calling API: %s\n" % e)