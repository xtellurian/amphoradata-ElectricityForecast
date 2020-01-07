# Predicting Electricity Spot Prices using Amphora Data
Give outline of what happened

Talk about why amphoradata, why spot prices of interest

## quick start on the project
1. Run `git clone https://github.com/1112114641/amphoradata-ElectricityForecast.git` in kozai terminal, or download from github `https://github.com/1112114641/amphoradata-ElectricityForecast` and then manually upload to kozai
2. Run `cd 0_setup`, then execute `export usrname='yourAmphoraUsername'`, and `export password='yourAmphoraPassword'` to set the amphora data password and username
3. Run `conda env create -f EForecast.yml` to setup the environment, install python packages and then activate the env with `conda activate EForecast`
 - if a `permission denied` error occurs, change rights with `chmod a+x EForecast.yml` to make the file executable, run 3., then change back to base dir with `cd ..`
4. Take care to ensure the variables $usrname and $password are set, so connection to amphora API possible. To test this, run `echo $usrname` or `echo $password` in your shell
5. Open the `ipynb` file in the base directory, which is now ready to be run

## talk about data importance, why amphora
For data science, and specifically for forecasting problems, accurate predictions require large amounts of data. Making a prediction on something not seen before by the model, will be difficult in most situations. Often, more data is not sufficient though, and other data, other variables help to improve model performance. Amphora offers...

![Example of the layout of the contents of an Amphora](4_images/amphoradataUI.png)
<!-- screenshot of UI of amphora + description-->

##  talking about koz.ai
Kozai, offers a very facile method for cloud data science with plugins allowing for direct git versioning of the entire project.
Moreover, it allows for the easy move between different work locations, obviating the need for constant fickle fights with updates of environments and packages, as the kozai environments is identical, irrespective of where you log in from. 

 - comment on management of teams? (covered during meeting today)
 - management of instances?
 - compute CPUs/GPUs available, multi processor
 - running of task montoring idk

<!-- screenshot of prelim UI kozai + description of to come-->

<!--### Connecting e.g. VSCode to a kozai sesh
 follow e.g. [here](https://blog.ouseful.info/2019/02/11/connecting-to-a-remote-jupyter-notebook-server-running-on-digital-ocean-from-microsoft-vs-code/ "Connect VSCode to external Kernel")
-->

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