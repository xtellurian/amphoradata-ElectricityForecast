import json
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

import amphora_client
from amphora_client.configuration import Configuration
from amphora_client.api_client import ApiClient
from amphora_client.rest import ApiException
from amphora_client.configuration import Configuration
from amphora_extensions.file_uploader import FileUploader
from amphora_client import SignalDto
import os

useful = {'username': os.environ.get('usrname'),
          'password': os.environ.get('password')}

def establish_connection():
    '''
    create token and return it

    status: working with v0.3.0!
    '''
    configuration = Configuration()
    configuration.host = 'https://beta.amphoradata.com'
    # create instance of API class, creates an unauthenticated client
    auth_api = amphora_client.AuthenticationApi(amphora_client.ApiClient(configuration)) 
    # if function throws ValueError 'username'/'password' must not be none, then the environment
    # variables are not set, c.f. setup-EForecast.sh and readme.MD section Quickstart
    token = auth_api.authentication_request_token(token_request = amphora_client.TokenRequest(username = useful['username'], password = useful['password']))
    
    return token

def _contained_signals(amphora_api,id_):
  """
  Check what signals are contained within amphora and return them as a list of strings.
  """
  # ask what data categories ("signals") are contained:
  signals = amphora_api.amphorae_get_signals(id_)
  properties= [s._property for s in signals]
  return properties, signals

def fetch_data(params):
    '''
    Fetches dataset in amphora ids, returns a pd.DataFrame of the API response
    
    params: parameter dict containing keys
      id          e.g. ecc5263e-83b6-42d6-8852-64beffdf204e
      token       as created by establish_connection()
      start_time  datetime object
      end_time    datetime object
      filter      optional parameter to filter results by
    returns
    pd.DataFrame

    '''
    id = params['id']
    start_time = params['start_time']
    end_time = params['end_time']
    if 'filter' in params.keys():
      filter_ = params['filter']
    else:
      filter_ = False
  
    # example ids to test run this function with:
    # ids= {'electricity_NSW': 'ecc5263e-83b6-42d6-8852-64beffdf204e',
    #       'weather_QLD': 'a46f461f-f7ee-4cc5-a1e4-569960ea5ed8',
    #       'electricity_SA': '89c2e30d-78c8-46ef-b591-140edd84ddb6',
    #       'electricity_QLD': 'ef22fa0f-010c-4ab1-8a28-a8963f838ce9',
    #       'electricity_VIC': '3b66da5a-0723-4778-98fc-02d619c70664',
    #       'weather_NSW': '11fd3d6a-12e4-4767-9d52-03271b543c66',
    #       'weather_VIC': 'd48ac35f-c658-41c1-909a-f662d6f3a972',
    #       'weather_SA': 'f860ba45-9dda-41e0-91aa-73901a323318'}

    configuration = Configuration()
    configuration.host = "https://beta.amphoradata.com"
    configuration.api_key["Authorization"] = "Bearer " + establish_connection()
    
    # To see amphora belonging to the id on the web:
    # print("https://beta.amphoradata.com/Amphorae/Detail?id={}".format(id)) 
    
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
      if filter_:
        for _ in properties:
          property_ = '$event.'+_
          variable1 = amphora_client.NumericVariable(kind="numeric",
                           value=amphora_client.Tsx(tsx=property_),
                           filter=amphora_client.Tsx("$event.periodType.String = '{}'".format(filter_)),
                           aggregation=amphora_client.Tsx("avg($value)"))
          value_dict[_] = variable1
      else:
        for _ in properties:
          property_ = '$event.'+_ 
          variable1 = amphora_client.NumericVariable(kind="numeric",
                           value=amphora_client.Tsx(tsx=property_),
                           aggregation=amphora_client.Tsx("avg($value)"))
          value_dict[_] = variable1

      # if some of the data needs to be filtered, e.g., ecc5263e-83b6-42d6-8852-64beffdf204e
      # has 'Forecast' and 'Actual' values for spot prices, also get the second one if
      # the parameter is passed 

      # if you dont understand wtf is going on:
      # https://docs.microsoft.com/en-au/rest/api/time-series-insights/dataaccess(preview)/query/execute#getseries

      # get the series (plural, if so defined above in value_dict!), save to time_series_data
      get_series = amphora_client.GetSeries([id], search_span= time_range, inline_variables=value_dict)
      time_series_data = ts_api.time_series_query_time_series(amphora_client.QueryRequest(get_series= get_series))
        
      # ingest data into pd.DataFrame
      # create list of returned timeseries
      apple = []
      for _ in range(len(time_series_data.properties)):
        apple.append(np.array(time_series_data.properties[_].values).reshape((len(time_series_data.properties[_].values))))

      # dataframe containing all data for further stuff
      banana = pd.DataFrame(apple).T
      banana.index =  np.array(time_series_data.timestamps)
      # delete weird artefact
      del banana[0]
      if filter_:
        properties_filtered = [p+'.'+filter_ for p in properties]
        banana.columns = properties_filtered
      else:
        banana.columns = properties
      return banana

    except ApiException as e:
      print("Exception when calling API: %s\n" % e)


def upload_series(df, params1, id_=''):
    '''
    Fetches dataset in amphora ids, returns a pd.DataFrame of the API response
    
    df:       pandas DataFrame
    id_:   (str) if amphora exists pass id, else pass empty string
    params:   parameter dict containing keys
      name        (str) name of amphora e.g. 'Price Predictions Avocado Futures'
      description (str) description of amphora e.g. 'Spot Price Predictions for Futures of 
                  Australian grown Hass Avocadoes.'
      price       (int) price of Amphora, can be >=0
      file_path   (str) path to files, or filename of file to be uploaded.
    

    returns:
      failure/success message
    '''
    name = params1['name']
    description = params1['description']
    price = params1['price']
    # file_path = params1['file_path']

    configuration = Configuration()
    configuration.host = "https://beta.amphoradata.com"
    configuration.api_key["Authorization"] = "Bearer " + establish_connection()

    try:
      # get API token and create authorized client:
      configuration.api_key["Authorization"] = "Bearer " + establish_connection()
      amphora_api = amphora_client.AmphoraeApi(amphora_client.ApiClient(configuration))

      # create amphora description (dto), then create amphora (amphora)
      if not id_:
        dto = amphora_client.CreateAmphoraDto(name = name, price = price, 
          description = description)
        amphora = amphora_api.amphorae_create(create_amphora_dto=dto)
        signals_liste = []
        for col in df.columns:
            signals_liste.append(SignalDto(_property=col, value_type='Numeric'))
        for _ in signals_liste:
          amphora_api.amphorae_create_signal(amphora.id, signal_dto=_)
      else:
        amphora = amphora_api.amphorae_read(id_)
        signal = _contained_signals(amphora_api, id_)[0]
        
        # test that all signals are exist both in the amphora and the df:
        cols = [n for n in df.columns]
        assert all(elem in signal for elem in cols), f'Content, i.e. signals, in Amphora do not match with DataFrame columns. Amphora contains: {signal}, df contains {cols}'
      
      # transform pd.DataFrame into API compatible format of a list of dicts (1 dict per df row)
      signal_list = []
      cats = [x for x in df.columns]
      for row in range(df.shape[0]):
        temp_dict = dict(t=df.index[row])
        for _ in range(len(cats)):
          temp_dict[cats[_]] = float(df.iloc[row,_])
        signal_list.append(temp_dict)
      # upload of signals
      amphora_api.amphorae_upload_signal_batch(amphora.id, request_body = signal_list)
      print(f'Uploaded {len(signal_list)} signals to amphora "{amphora.name}" under "{amphora.id}"')
      
      # if one where to upload a file:
      # uploader = FileUploader(amphora_api)
      # 
      # the way it is set ip, file_path can either be an entire directory 
      # corresponding to head in (head, tail = ntpath.split(path)),
      # or to tail, the filename, if there is a valid filename!
      # uploader.create_and_upload_file(amphora.id, file_path)

    except ApiException as e:
      print("Exception when calling API: %s\n" % e)




if __name__ == "__main__":
  ## very lazy testing of upload functionality:
  # Create test DataFrame to be upload
  ananas = pd.DataFrame([[0,1,2],[3,4,5],[6,7,8],[9,10,11]],
    index=[datetime(2019,12,2,11,0,0),datetime(2018,12,2,12,0,0),
    datetime(2017,12,2,13,0,0),datetime(2016,12,2,14,0,0)], 
    columns=['banana','appl','citron'])

  # create test settings for the test amphora:
  params = {'name': "Test - now also with signals",
        'description': "Day 501 - still no signals.",
        'price': 0
        }
  # upload to amphora data:      
  upload_series(ananas, params, id_='bab66317-3e8a-4e3f-94b2-aa5fe890dc11')