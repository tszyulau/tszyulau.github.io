**Machine Learning Models to Analyze the Impact of Traffic Emissions on PM2.5 Levels in New York City**

I applied machine learning techniques to investigate... Below is my report.

***

## Introduction 

###Here is a summary description of the topic. Here is the problem. This is why the problem is important.

Air quality in urban environments is significantly impacted by fine particulate matter (PM2.5), a complex mixture of extremely small particles and liquid droplets that are present in the atmosphere. These particles, which are 2.5 micrometers or smaller in diameter, are of particular concern in densely populated metropolitan areas like New York City, where millions of residents are exposed to varying levels of air pollution daily. Here is the problem: Traffic emissions, particularly nitric oxide (NO) and nitrogen dioxide (NO2), are major contributors to PM2.5 levels in urban areas, but the precise nature and strength of these relationships remain inadequately quantified and understood. This lack of understanding hinders the development of effective pollution control strategies. This is why the problem is important: PM2.5 particles are especially dangerous because their microscopic size allows them to penetrate deep into the respiratory system, crossing into the bloodstream and potentially affecting multiple organ systems. This exposure has been linked to a wide range of serious health impacts, including respiratory diseases, cardiovascular complications, and premature mortality. In vulnerable populations such as children, elderly individuals, and those with pre-existing health conditions, these health risks are even more pronounced. Furthermore, the economic burden of PM2.5-related health issues places significant strain on healthcare systems and communities.

###There is some dataset that we can use to help solve this problem. This allows a machine learning approach. This is how I will solve the problem using supervised/unsupervised/reinforcement/etc. machine learning.

We have access to comprehensive environmental monitoring datasets from the New York City Environmental Health Data Portal. These datasets provide detailed annual average concentrations of PM2.5, NO2, and NO across various community districts throughout New York City. The data spans multiple years and includes measurements from different locations, capturing the spatial and temporal variations in pollutant concentrations. The datasets also account for various community districts with different traffic patterns, population densities, and urban characteristics. This allows a machine learning approach: The structured nature of this data, combined with its comprehensive coverage of multiple pollutants and locations, makes it ideal for applying advanced statistical and machine learning techniques to uncover complex relationships between traffic-related emissions and PM2.5 levels. This is how I will solve the problem using supervised machine learning: We will implement a systematic approach using both Linear Regression and Ridge Regression models. These models will be trained to predict PM2.5 levels based on NO and NO2 concentrations, incorporating sophisticated data preprocessing techniques including handling missing values, feature scaling, and normalization. The choice of regression models allows us to quantify the relative importance of different traffic-related pollutants while accounting for potential multicollinearity between predictors.

###We did this to solve the problem. We concluded that...

First, we performed extensive data preprocessing, including merging multiple datasets, handling missing values through appropriate imputation techniques, and normalizing features to ensure comparable scales. We then conducted comprehensive exploratory data analysis to understand the distributions of pollutants across different community districts and visualize potential correlations between variables. Following this, we developed and trained our regression models, carefully tuning hyperparameters to optimize performance. We evaluated the models using a robust set of metrics including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R² score, and cross-validation scores to ensure reliable and generalizable results. We concluded that: Our analysis revealed several significant findings about the relationship between traffic emissions and PM2.5 levels in New York City. First, we found that traffic-related emissions have a substantial and quantifiable impact on PM2.5 concentrations, with our models explaining a significant portion of the variance in PM2.5 levels. Notably, NO2 emerged as a stronger predictor of PM2.5 levels compared to NO, suggesting that certain types of traffic emissions may have more significant impacts on air quality than others. The spatial analysis revealed hotspots where the relationship between traffic emissions and PM2.5 is particularly strong, often corresponding to areas with high traffic density. These findings provide crucial insights for policymakers, highlighting the importance of targeted traffic management policies and suggesting specific areas where interventions might be most effective in reducing urban air pollution and protecting public health.


## Data
- Code: https://colab.research.google.com/drive/1KB060R_tsXEkppfVi7MSNn2mA84aMhvV?usp=sharing
- Dataset: PM2.5: https://drive.google.com/file/d/1o1ClooaRVEIVjdf0LPbrHsmJjEBvfdMG/view?usp=sharing
           NO2  : https://drive.google.com/file/d/1Nd-4dSLHyB5qAiBMFIUNLCTntloMfRpW/view?usp=drive_link
           NO   : https://drive.google.com/file/d/16bnMVJRpuHmC4f5JNGMFJaUZ9_xEyHaG/view?usp=drive_link
## Modelling

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

## Results

Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] DALL-E 3

[back](./)

