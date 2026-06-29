# ericsoc
# MatrixOrigin Eric Summer Internship Project

This repository contains my summer internship project at MatrixOrigin.
The project uses Chicago Transportation Network Provider (TNP) rideshare trip data to explore demand patterns, price dynamics, and simple demand forecasting.

The main goal is to prepare and analyze ride-hailing data for future dynamic pricing and supply forecasting work.

## Data

The dataset is based on Chicago rideshare trip records from 2022 to 2024.

I used monthly samples with about 100,000 trips per month. In total, the raw data contains about 3.6 million rows.

Important columns include:

* trip start time
* trip distance
* trip duration
* pickup and dropoff area
* fare
* additional charges
* total trip price
* pickup and dropoff coordinates

Because the monthly files are sampled, the sample data should not be used directly as the true total trip volume. For real yearly or monthly trip volume, I used the Chicago Data Portal API.

## Files

### 1. `chicago_tnp_eda.ipynb`

This notebook is the first step of the project.

It does exploratory data analysis and data preparation.

Main work:

* Loaded 36 monthly CSV files from 2022 to 2024.
* Combined them into one DataFrame with about 3.6 million rows.
* Standardized column names.
* Checked data types, missing values, and time coverage.
* Found that each month has exactly 100,000 sampled rows.
* Found some invalid or extreme records, such as zero fare, zero miles, and very long trips.
* Cleaned the data by removing invalid rows and extreme outliers.
* Created time features such as hour, day of week, weekend, night, and rush hour.
* Checked demand patterns by hour and day of week.
* Checked fare distribution.
* Converted pickup and dropoff locations into H3 cells.
* Saved the cleaned data.
* Loaded the cleaned data into MatrixOne.

Main findings:

* The raw sample has 3,600,000 rows and 21 columns.
* After cleaning, about 3,582,404 valid rows remained.
* Fare, trip miles, and trip seconds had some zero values, so these rows were removed.
* Census tract columns had many missing values, so H3 cells and community areas are more useful for location analysis.
* Pickup locations were converted into 856 distinct H3 cells.
* Finally, 3,560,474 cleaned rows were loaded into the MatrixOne `trips` table.

This notebook builds the data foundation for the later price analysis and demand forecasting notebooks.

---

### 2. `chicago_tnp_price_analysis.ipynb`

This notebook studies price dynamics.

The goal is to separate each fare into two parts:

1. A base price explained by distance and time.
2. A variable part that may reflect demand pressure or surge pricing.

Main work:

* Loaded trip-level data from the MatrixOne `trips` table.
* Removed shared/pooled trips because pooled fares may distort per-trip price.
* Created time features such as year, month, hour, weekend, night, and rush hour.
* Used linear regression to estimate a rule-like base fare for each year.
* Calculated:

```text
variable = actual fare - predicted base fare
```

* Checked whether the variable part behaves like demand-driven pricing.
* Compared dynamic pricing strength across 2022, 2023, and 2024.
* Pulled real yearly trip volume from the Chicago Data Portal API.

Main findings:

* The estimated base fare is mainly explained by trip distance and trip duration.
* The variable part is higher during rush hour and night, which supports the idea that it captures demand pressure.
* From 2022 to 2024, real Chicago rideshare demand increased from about 69 million trips to about 91 million trips.
* However, the average night premium became much smaller.
* Night premium dropped from about +$2.48 in 2022 to about +$0.05 in 2024.
* Price volatility also became smaller.
* This suggests that demand increased, but dynamic pricing became flatter.

Important limitation:

This notebook can show what happened, but not exactly why it happened.
To explain why dynamic pricing weakened, we would need driver supply data, platform policy data, weather, and event information.

---

### 3. `chicago_tnp_demand_forecast.ipynb`

This notebook builds a simple daily demand forecasting model.

The goal is to predict daily trip demand for the busiest H3 pickup cells.

Main work:

* Loaded trip data from MatrixOne.
* Counted trips by pickup H3 cell and date.
* Used daily demand instead of hourly demand because the dataset is only a sample, so hourly cell-level data would be too noisy.
* Kept the top 50 busiest H3 cells.
* Filled missing cell-day combinations with zero demand.
* Created features including:

  * day of week
  * weekend flag
  * month
  * day of year
  * H3 cell code
  * lagged demand from previous days
  * 7-day and 28-day rolling average demand
  * average demand level for each cell
* Used a time-based train/test split:

  * train: 2022–2023
  * test: 2024
* Built two models:

  * Linear Regression
  * LightGBM

Main findings:

* Linear Regression performed better than LightGBM.
* Linear Regression reached MAE around 7.8 trips, about 23% of mean demand.
* LightGBM had MAE around 10.2 trips and did not beat the simpler model.
* This suggests that daily demand in the top cells is mostly driven by stable and near-linear patterns.
* Recent average demand is a very strong signal.
* A more complex model is not always better if the available features are simple and stable.

Important limitation:

The model only predicts completed trips, not unmet demand.
It also does not include weather, events, or real-time driver supply yet. These features could improve future forecasting.

## Overall Project Flow

The project follows this order:

1. Start with raw Chicago rideshare CSV files.
2. Clean and validate the data.
3. Create useful time and location features.
4. Convert pickup and dropoff locations into H3 cells.
5. Load the cleaned dataset into MatrixOne.
6. Analyze price dynamics using MatrixOne data.
7. Build a simple daily demand forecasting model.

## Main Conclusions

* The Chicago TNP dataset is usable for early-stage ride-hailing analysis.
* H3 cells are useful for location-based demand and pricing analysis.
* Distance and time explain a large part of fare.
* The remaining fare residual behaves like a demand-driven price component.
* Chicago rideshare demand grew from 2022 to 2024, but dynamic pricing signals became weaker.
* For daily demand forecasting, a simple linear model performed better than LightGBM.
* The next step should be adding external features such as weather, events, holidays, and driver supply.

## Next Steps

Possible next steps:

* Add weather data.
* Add event data, such as sports games and concerts.
* Add airport and downtown distance features.
* Build hourly models if more complete data is available.
* Add driver supply data if available.
* Try shortage prediction using demand and supply features.
* Improve model evaluation with more time periods and more locations.

## Tools Used

* Python
* pandas
* matplotlib
* scikit-learn
* LightGBM
* H3
* PyMySQL
* MatrixOne
* Chicago Data Portal API

