# Model Card

The model discussed in this card, identifies whether a US citizen is in a 'high' or 'low' income bracket with typical census bureau data. Salary above 50K USD characterises the 'high' bracket and below or equals the 'low'.

## Model Details

The model used, is a Random Forest with the following specification:

- n_estimators: 1000
- max_depth: 3
- min_samples_leaf: 4
- random_state: 0

The remaining parameters are the default ones that can be found at the scikit-learn documentation for version 1.1.2, which was the RF implementation that was used.

## Intended Use

The model intends to allow predicting the income category of an american citizen, according to data points typicaly collected by a census survey. There are two categories 'high' and 'low', and the split is at 50K USD, where above it is considered 'high' and below or equals, 'low'. The inference can be accessed via a POST request at `https://ml-model-ci-cd.herokuapp.com/predict/`. The body of the POST request will have to contain the features discussed in the following section.

## Training Data

The model was trained with the public dataset that can be found at https://archive.ics.uci.edu/ml/datasets/census+income, with 80% of the data. The dataset comes from census survey data before 01-05-1996 (the day it was donated to the repository) and the split for train and test data was made with a random_state of 0, and a stratification so that the ratio of the 2 classes of the target variable is maintained between train and test data.

The data received some basic cleaning of removing initial spaces on categorical features and removing rows with NaNs. After cleaning, the number of rows of the train data was 24129, and they contained 14 columns and a binary target variable. Out of the 14 columns, 8 were categorical and 6 continuous. The categorical ones were:
```
workclass, education, marital-status, occupation, relationship, race, sex, native-country
```

and the continuous:
```
age, fnlwgt (a weight referring to how representative the specific sample is), education-num, capital-gain, capital-loss, hours-per-week
```

For categorical features, an OneHotEncoder was used, and a LabelBinarizer for the target variable. The final number of columns of the data after all the pre-processing was 103.

## Evaluation Data

The evaluation data follow a similar specification with the train data in terms of origin and features. The test dataset was 20% of the entire dataset, and contained 6033 rows. The split was done with stratification on the traget variable so that both classes are appropriately represented.

## Metrics

The model was evaluated using precision, recall and f-beta score with the following results:

precision: 0.97
recall: 0.19
f-beta: 0.32

## Ethical Considerations

Models that are predicting salary can have significant effect on someone's life. In various situations someone with a higher income would have more opportunities, like for example getting a loan, renting a house and more. For that reason it is important that our process has the least possible bias to be used in a setting with importance.

It is especially important that, slices that have to do with minorities are not under-represented. And that the underlying characteristics that determine salary are included, rather than ones that have absolutely no causal effect to the salary, like race, sex etc. However this is not always the case for the current model as we will also discuss in the next section.

## Caveats and Recommendations

With the current state of the model there are various caveats, most notably the various ways of bias in the data.

- outdated data: The economic landscape has significantly changed the years following 1996 with various crises around the world and inflation being on the rise. Most likely the data at most slices would be different already a few years after 1996
- US only data: the economic situation of various slices might be different in other places around the world
- fnlwgt: there are indications that it has been been produced differently by different census bureaus accross the different US states (see https://www.kaggle.com/datasets/uciml/adult-census-income/discussion/32698)
- data collection: the data are collected from different bureaus in different times, but we are uncertain of what bias might come from the collection process. For example inconsistencies might be present between the bureaus or collection periods in general (we only know the time the data were published at)
- performance: The performance of the model is obviously not good at this point especially when it comes to identifying the 'high' salary class
- under-represented slices: In many cases there are not enough datapoints in the various classes. For example if we look at the 'workclass' feature, we have 4402 observations in the 'Private' category and only 5 at 'Without-Pay'.

This model, at this point in time, is only recommended for use with only historical data of a short period before 1996, and should certainly be avoided in any critical or sensitive situation where its income prediction could have signifficant effect on someone's life.