import amphora_client
#from amphora_client.api import amphora_api
from amphora_client.configuration import Configuration
from amphora_client.api_client import ApiClient
from amphora_client.rest import ApiException
from amphora_client.configuration import Configuration
import json
import pandas as pd

useful = {'API-token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJodHRwOi8vc2NoZW1hcy54bWxzb2FwLm9yZy93cy8yMDA1LzA1L2lkZW50aXR5L2NsYWltcy9uYW1lIjoiTUFSS1VTLkRPTExNQU5OQEVMSUlaQS5DT00uQVUiLCJodHRwOi8vc2NoZW1hcy54bWxzb2FwLm9yZy93cy8yMDA1LzA1L2lkZW50aXR5L2NsYWltcy9uYW1laWRlbnRpZmllciI6IjhiMjk1OWM3LTY0ZWUtNGQ1YS1hN2E0LTQ0NjA1NWQ0MjljZSIsImV4cCI6MTU3NTUwNzc3NSwiaXNzIjoiYW1waG9yYWRhdGEuY29tIiwiYXVkIjoiYW1waG9yYWRhdGEsY29tIn0.rdpeD7zOrsLTlkeOnX-p5fjf5jxZZE8TKmf5DJ0ElPo',
  'username': 'markus.dollmann@eliiza.com.au',
  'password': 'fobra1-penqef-Bebnuj'}

def establish_connection():
    '''
    create token and return it

    status: working with v0.3.0!
    '''
    configuration = Configuration()
    configuration.host = 'http://beta.amphoradata.com'
    # create instance of API class, creates an unauthenticated client
    auth_api = amphora_client.AuthenticationApi(amphora_client.ApiClient(configuration)) 
    token = auth_api.authentication_request_token(token_request = amphora_client.TokenRequest(username = useful['username'], password = useful['password']))
    configuration.api_key["Authorization"] = "Bearer " + token
    # read user info
    users_api = amphora_client.UsersApi(amphora_client.ApiClient(configuration)) # creates an authenticated client
    
    return users_api , token

# def fetch_data(users_api, id):
#     '''
#     Fetches dataset in amphora ids, returns a pd.DataFrame of the API response

#     TODO: implement muli-ID fetch
#     '''
#     ids= {'electricity_NSW': 'ecc5263e-83b6-42d6-8852-64beffdf204e',
#           'weather_QLD': 'a46f461f-f7ee-4cc5-a1e4-569960ea5ed8',
#           'electricity_SA': '89c2e30d-78c8-46ef-b591-140edd84ddb6',
#           'electricity_QLD': 'ef22fa0f-010c-4ab1-8a28-a8963f838ce9',
#           'electricity_VIC': '3b66da5a-0723-4778-98fc-02d619c70664',
#           'weather_NSW': '11fd3d6a-12e4-4767-9d52-03271b543c66',
#           'weather_VIC': 'd48ac35f-c658-41c1-909a-f662d6f3a972',
#           'weather_SA': '860ba45-9dda-41e0-91aa-73901a323318'}
#     print(f'Getting signals for: {amphora_api.amphorae_read(ids[id]).name}')    
#     try:
#         api_response = api_instance.amphorae_get_signals(id)
#     except ApiException as e:
#         print('Exception when calling AmphoraeAPI->amphorae_get_signals: {}\n'.format(e))
#     # for id in ids:
#     #     print(f'Getting signals for: {amphora_api.amphorae_read(ids[id]).name}')    
#     #     try:
#     #         api_response = api_instance.amphorae_get_signals(id, x_amphoradata_version=x_amphoradata_version)
#     #     except ApiException as e:
#     #         print('Exception when calling AmphoraeAPI->amphorae_get_signals: {}\n'.format(e))
    
#     return  pd.DataFrame(api_response)  
    
# line 30
# https://github.com/amphoradata/python-sdk/blob/master/samples/query_signal.py

# def fetch_amphora(dataID, start_date, end_date):
#     """
#     Fetch data from the amphora API in json (else modify accept code), returns timeseries in dataframe.
#     or:
#     Fetches data (requests), unpacks XML into dict, forwards to pandas.DataFrame, names columns neatly
    
#     Data: (str) specifies dataset identifier (beta.amphora.com for id)
#     start_date: (str) 'YYYY-MM-DD' no earlier than 1999-01-04
#     end_date: (str) same format
#     """
#     #query API
#     header = {"Accept":"application/json", "Content-Type": "application/json"}
#     param = {"updatedAfter":start_date,'startPeriod': start_date, 'endPeriod': end_date}
#     if not data:
#         pass
#     else:
#         data = '/' + data
#         param = {}
#     data_raw = requests.get("https://sdw-wsrest.ecb.europa.eu/service/data/EXR" + data, params=param, headers=header)
#     assert data_raw.status_code==200, 'Communication with ECB API failed. Error Code: '+str(data_raw.status_code)
    
#     #get data out of API response
#     pre_dict_ = parse(data_raw.text)
#     df_ = pd.DataFrame()
#     temp = pre_dict_['message:GenericData']['message:DataSet']['generic:Series']
#     for i,i_tmp in enumerate(temp):
#         column_name = temp[i]['generic:SeriesKey']['generic:Value'][1]['@value']
#         pre_dict_2 = temp[i]
#         data_dict = {_['generic:Obs']['generic:ObsDimension']['@value']: float(_['generic:Obs']['generic:ObsValue']['@value']) for _ in pre_dict_2}
#         temp = pd.DataFrame.from_dict(data_dict, orient='index')
#         if i != 0: 
#             df_[column_name] = temp.iloc[:,-1]
#         else:
#             df_ = temp
#             df_.rename(columns={"index": "dates", list(df_)[-1]:column_name},inplace=True)
#     etl.save_df_to_csv(df_,type_, dates)
#     return df_    