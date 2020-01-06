# Predicting Electricity Spot Prices using Amphora Data
Talk about why amphoradata, why spot prices of interest

## quick start on the project
1. create folder for project, run `git clone https://github.com/1112114641/amphoradata-ElectricityForecast.git`, or download manually from `https://github.com/1112114641/amphoradata-ElectricityForecast`
2. modify `0_setup/setup-EForecast.sh` to contain right amphora data password and username
3. run `bash 0_setup/setup-EForecast.sh` to setup the environment, install python packages and set environment variables
4. take care to ensure the variables $usrname and $password are set, so connection to amphora API possible. To test this, run `echo $usrname` or `echo $password` in your shell

## talk about data importance, why amphora
For data science, and specifically for forecasting problems, accurate predictions require large amounts of data. Making a prediction on something not seen before by the model, will be difficult in most situations. Amphora offers...
<!-- screenshot of UI of amphora + description-->

##  talking about koz.ai
Kozai, offers a very facile method for cloud data science with plugins allowing for direct git versioning of the entire project.
Moreover, it allows for the easy move between different work locations, obviating the need for constant fickle fights with updates of environments and packages, as the kozai environments is identical, irrespective of where you log in from. 

 - comment on management of teams? (covered during meeting today)
 - management of instances?
 - compute CPUs/GPUs available, multi processor
 - running of task montoring idk

<!-- screenshot of prelim UI kozai + description of to come-->

## talking about the datascience
structure of eda, API access to amphora, revisit modelling/multiple models, choice of model by lowest error criterion

### prelims: amount of data
data was limited so results jumpy, ideally at least 3yrs of data, not just 3 months (possible project on the murray river basin water levels)

After efforts to create new features with a higher predicitive power was scrapped, as their power showed high variance from day to day, as new data came in.

### prelims: model selection
Often, the central question for data science is the choice of the model, where several competing factors have to be considered: between explainability, model performance, model inference time, and model train time.
linear model, RFR, Dense, CNN model, all compared for the chose criterion of rmse, as large errors are punished more severly

## predictions
predictions showed to be reasonable for the prediction of trends in spot prices, whereas the precise value mostly was off.
<!-- 2x4 grid of date (4x) vs ((QLD,NSW, True),((VIC,SA,True)))-->

## final note
feature engineering, more involved models, etc.