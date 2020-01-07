# Predicting Electricity Spot Prices using [Amphora Data](https://amphoradata.com) and Koz.ai

## Cui Bono?

Predicting the price of electricity is increasingly important. The wholesale market of electricity is extremely variable and many companies and organisations are highly exposed. Whilst they can benefit from periods of low prices (e.g. < $ 50 /MWh) they also face prices of up to $ 14000 /MWh. There is a fundamental conflict - businesses want price certainty to plan their daily operations, but the NEM requires price variability to shift production along the supply/ demand curve. Here, we present a small contribution to resolving this conflict with methods and tools for better predicting price fluctuations, which can help companies manage the use of their electricity accordingly.

## What is the NEM?

The Australian National Electricity Market (NEM) is a wholesale electricity market in which generators sell electricity and retailers buy it to on-sell to consumers. There are over 100 generators and retailers participating in the market, so it’s highly competitive and therefore an efficient way of maintaining relatively competitive electricity prices in the wholesale market.

All electricity sales are traded through the NEM. It is a wholesale market and prices fluctuate in response to supply and demand at any point in time. The NEM is split into 5 regions (QLD, NSW, VIC, TAS, SA), and prices are set per-region.

![Map of the NEM regions](images/map-NEM.png)

## What do the prices look like?

> All plots and data below are available at [Amphora Data](https://amphoradata.com).

Electricity prices are set every 5 minutes (the dispatch price) and 6 dispatch prices are averaged every half hour to generate the spot price. Below are the prices for a 'normal' week in South Australia. We observe that the price remains relatively close to $50, although occasionally dropping and [becoming negative](https://www.afr.com/companies/energy/why-electricity-spot-prices-are-hitting-zero-20190723-p52a08). At these times, consumers of electricity are actually paid to consume.

![Plot of normal electricity price and scheduled generation in a week](images/plot-normalElectricityPriceAndGen-SA.png)

Now compare the 'normal' week above, to the week that immediately followed it in December of 2019. Clearly things are very different. The 'normal' variability ~$50 is no longer visible, as the graph is dominated by a huge spike on December 19, when the price jumped to over $8,000, approximately 2 orders of magnitude larger than the maximum for the days either side. We also observe that the maxiumum scheduled generation has doubled compared with the previous week. So what's going on?

![Plot of abnormal electricity price and scheduled generation in a week](images/plot-spikeElectricityPriceAndGen-SA.png)

A major driver of electricity demand (and therefore supply, as the two must be equal) is the weather. The plot of temperature below shows that Adelaide was experiencing a serious heatwave from December 17 to December 20, with the price spike occuring the evening before the hottest and last day. Keen observers will also note that the minimum overnight temperature was high at 28 degrees.

We've highlighted the same period for each day of the heatwave (7 - 9 pm), which coincides with the price spike on the last day. Temperatures are significant, because higher temperatures mean more air-conditioning (i.e. greater consumption) and often less efficient plant. Wind speed is also significant (although not necessarily just in Adelaide) because of the high penetration of wind energy in South Australia.

![Plot of temperature and windspeed during abnormal week](images/plot-weatherAroundSpike-Adelaide.png)

## Counting the cost.

If your business was regularly running plant at 7pm, and that plant consumes ~250 kW of power, on a normal day you'd spend about $15/hour to run the plant. Compare that to the abnormal day on December 19. On that day, at 7pm, you'd spend $1,500 in a single hour!

## Why hasn't this happened to me?

Luckily for regular consumers, electricity retailers (think AGL, Origin etc.) protect you from price variability by hedging against shortfalls in generation. Therefore, no ridiculously expensive days. Consumers *are* paying for the price spikes, it's just averaged out over the entire year.

## So why would anyone want to be exposed to the variable price?

Remember that during the normal week of operations, the price actually went negative, meaning that anyone exposed and consuming electricity during that period was actually paid by the market. For our imaginary business, for the hour when price dropped to $-30, they would have actually been paid about $7 to run their plant - a $22 improvement on the typical situation.

You may also wish to be exposed because as long as you don't consume during periods of high prices, your total electricity bill will be significantly smaller because you aren't paying a retailer to hedge the market for you.

## How to manage costs when exposed to the wholesale price.

Best case scenario, you need an automated and robust prediction of the price, and an automated go/no-go decision framework. [Amphora Data](https://AmphoraData.com) and Koz.ai have partnered to develop this open source experimentation tool for achieving just that. We hope you find it useful :)

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