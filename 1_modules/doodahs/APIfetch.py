import json
import pandas as pd
import numpy as np
import os
import time

import amphora_client
from amphora_client.configuration import Configuration
from amphora_client.api_client import ApiClient
from amphora_client.rest import ApiException
from amphora_client.configuration import Configuration
from datetime import datetime, timedelta

useful = {'API-token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJodHRwOi8vc2NoZW1hcy54bWxzb2FwLm9yZy93cy8yMDA1LzA1L2lkZW50aXR5L2NsYWltcy9uYW1lIjoiTUFSS1VTLkRPTExNQU5OQEVMSUlaQS5DT00uQVUiLCJodHRwOi8vc2NoZW1hcy54bWxzb2FwLm9yZy93cy8yMDA1LzA1L2lkZW50aXR5L2NsYWltcy9uYW1laWRlbnRpZmllciI6IjhiMjk1OWM3LTY0ZWUtNGQ1YS1hN2E0LTQ0NjA1NWQ0MjljZSIsImV4cCI6MTU3NTUwNzc3NSwiaXNzIjoiYW1waG9yYWRhdGEuY29tIiwiYXVkIjoiYW1waG9yYWRhdGEsY29tIn0.rdpeD7zOrsLTlkeOnX-p5fjf5jxZZE8TKmf5DJ0ElPo',
  'username': 'markus.dollmann@eliiza.com.au',
  'password': 'fobra1-penqef-Bebnuj'}

def establish_connection():
    '''
    create token and return it

    status: working with v0.3.0!
    '''
    configuration = Configuration()
    configuration.host = 'https://beta.amphoradata.com'
    # create instance of API class, creates an unauthenticated client
    auth_api = amphora_client.AuthenticationApi(amphora_client.ApiClient(configuration)) 
    token = auth_api.authentication_request_token(token_request = amphora_client.TokenRequest(username = useful['username'], password = useful['password']))
    
    return token

def contained_signals(id, token):
  """
  Check what signals are contained within amphora and return them as a list of strings.
  """
  # create api config class instance:
  configuration = Configuration()
  configuration.host = "https://beta.amphoradata.com"
  # set the authorization token for requests and pass configs
  configuration.api_key["Authorization"] = "Bearer " + token
  amphora_api = amphora_client.AmphoraeApi(amphora_client.ApiClient(configuration))
  # ask what data categories ("signals") are contained:
  signals = amphora_api.amphorae_get_signals(id)
  properties=list((s._property for s in signals))
  return properties, signals

def fetch_data(params):
    '''
    Fetches dataset in amphora ids, returns a pd.DataFrame of the API response
    
    params: parameter dict containing keys
      id          e.g. ecc5263e-83b6-42d6-8852-64beffdf204e
      token       as created by establish_connection()
      start_time  datetime object
      end_time    datetime object
      
    returns
    pd.DataFrame

    TODO: implement muli-ID fetch
    '''
    id = params['id']
    start_time = params['start_time']
    end_time = params['end_time']
  
    ids= {'electricity_NSW': 'ecc5263e-83b6-42d6-8852-64beffdf204e',
          'weather_QLD': 'a46f461f-f7ee-4cc5-a1e4-569960ea5ed8',
          'electricity_SA': '89c2e30d-78c8-46ef-b591-140edd84ddb6',
          'electricity_QLD': 'ef22fa0f-010c-4ab1-8a28-a8963f838ce9',
          'electricity_VIC': '3b66da5a-0723-4778-98fc-02d619c70664',
          'weather_NSW': '11fd3d6a-12e4-4767-9d52-03271b543c66',
          'weather_VIC': 'd48ac35f-c658-41c1-909a-f662d6f3a972',
          'weather_SA': '860ba45-9dda-41e0-91aa-73901a323318'}

    configuration = Configuration()
    configuration.host = "https://beta.amphoradata.com"
    configuration.api_key["Authorization"] = "Bearer " + establish_connection()
    
    # To see amphora belonging to id:
    # print("https://beta.amphoradata.com/Amphorae/Detail?id={}".format(id)) 
    # id = "57d6593f-1889-410a-b1fb-631b6f9c9c85" # weather Albury Wodonga (VIC)
    # id = '3b66da5a-0723-4778-98fc-02d619c70664' # electricity VIC
    
    
    try:
      amphora_api = amphora_client.AmphoraeApi(amphora_client.ApiClient(configuration))
      #print(f'Getting signals for: {amphora_api.amphorae_read(id).name}')
      signals = amphora_api.amphorae_get_signals(id)
      # grab all the different timeseries names using a listcomprehension
      properties = [s._property for s in signals]

      # the API for interacting with time series
      ts_api = amphora_client.TimeSeriesApi(amphora_client.ApiClient(configuration)) 

      time_range = amphora_client.DateTimeRange(_from = start_time, to = end_time)
      
      # get tomorrow's temperatures
      
      property_ = []
      value_dict = {}
      for _ in properties:
        property_ = '$event.'+_ 
        variable1 = amphora_client.NumericVariable(kind="numeric",
                           value=amphora_client.Tsx(tsx=property_),
                           aggregation=amphora_client.Tsx("avg($value)"))
        value_dict[_] = variable1

      # variable = amphora_client.NumericVariable( kind="numeric", 
      #     value=amphora_client.Tsx(tsx='$event.temperature'), 
      #     aggregation=amphora_client.Tsx("avg($value)"))
      # get the series (plural, if so defined above in value_dict!), save to time_series_data
      get_series = amphora_client.GetSeries([id], search_span= time_range, inline_variables=value_dict)
      time_series_data = ts_api.time_series_query_time_series(amphora_client.QueryRequest(get_series= get_series))
      
      # ingest data into pd.DataFrame
      apple = []
      # create list of returned timeseries
      for _ in range(len(time_series_data.properties)):
        apple.append(np.array(time_series_data.properties[_].values).reshape((len(time_series_data.properties[_].values))))

      # dataframe containing all data for further stuff
      banana = pd.DataFrame(apple).T
      banana.index =  np.array(time_series_data.timestamps)
      # delete weird artefact
      del banana[0]
      banana.columns = properties
      return banana

    except ApiException as e:
      print("Exception when calling API: %s\n" % e)
