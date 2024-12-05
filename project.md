**Machine Learning Models to Analyze the Impact of Traffic Emissions on PM2.5 Levels in New York City**

I applied machine learning techniques to investigate... Below is my report.

***

## Introduction 

  Air pollution is one of the leading environmental challenges.  In urban areas, traffic emissions are a primary contributor to PM2.5 levels, with pollutants like nitric oxide (NO) and nitrogen dioxide (NO2) playing key roles in the formation of secondary particulate matter, which is posing significant health risks. PM2.5 particles, which are small enough to penetrate deep into the lungs and bloodstream, are linked to respiratory and cardiovascular diseases, and even premature mortality[1].  In New York city, 14 percent of current PM2.5 concentration comes from local vehicle emissions, which place it as the second largest PM2.5 emission source[2]. Specifically, while researches have stated that 4% to 37% of NOx, a major vehicle emission, is known to contribute to the formation of PM2.5 through chemical reactions in the atmosphere, the degree to which reducing traffic emissions might lower PM2.5 concentrations is not straightforward [3]. Traffic emissions regulation is critical for formulating effective air quality management strategies. While local vehicles are something that can go under restrictions and regulate the emission, we can regulate vehicle emissions by turning gasoline and diesel cars into EVs. The problem relies on quantifying this relationship between traffic-related emissions and PM2.5 levels. Using machine learning can help us in understanding this problem in depth by predicting patterns and effects and therefore can inform policies aimed at reducing traffic emissions, eventually improving urban air quality.
  This project focuses on the air quality challenge in New York City. We have datasets from the New York City Environmental Health Data Portal[4]. The datasets include PM2.5, NO, and NO2 levels for 2022. By cleaning the data and using machine learning techniques, we can modelize the trends and analyze the correlations between these pollutants and predict the impact of hypothetical traffic reductions. By using models like linear regression and gradient boosting and applying weighting method, we can quantify the contribution of traffic emissions to PM2.5 and simulate the effects of reduced traffic emissions, for instance, a 5%, 10%, 15% and 90% of reduction in PM2.5, NO and NO2 levels. The results showed a measurable reduction in PM2.5 levels, emphasizing the importance of managing traffic emissions for better air quality. This project in machine learning aims to provides insights of current air quality situation and predict situations under different scenarios, highlighting the value of machine learning in environmental analysis.

###There is some dataset that we can use to help solve this problem. This allows a machine learning approach. This is how I will solve the problem using supervised/unsupervised/reinforcement/etc. machine learning.

We have access to comprehensive environmental monitoring datasets from the New York City Environmental Health Data Portal. These datasets provide detailed annual average concentrations of PM2.5, NO2, and NO across various community districts throughout New York City. The data spans multiple years and includes measurements from different locations, capturing the spatial and temporal variations in pollutant concentrations. The datasets also account for various community districts with different traffic patterns, population densities, and urban characteristics. This allows a machine learning approach: The structured nature of this data, combined with its comprehensive coverage of multiple pollutants and locations, makes it ideal for applying advanced statistical and machine learning techniques to uncover complex relationships between traffic-related emissions and PM2.5 levels. This is how I will solve the problem using supervised machine learning: We will implement a systematic approach using both Linear Regression and Ridge Regression models. These models will be trained to predict PM2.5 levels based on NO and NO2 concentrations, incorporating sophisticated data preprocessing techniques including handling missing values, feature scaling, and normalization. The choice of regression models allows us to quantify the relative importance of different traffic-related pollutants while accounting for potential multicollinearity between predictors.

###We did this to solve the problem. We concluded that...

First, we performed extensive data preprocessing, including merging multiple datasets, handling missing values through appropriate imputation techniques, and normalizing features to ensure comparable scales. We then conducted comprehensive exploratory data analysis to understand the distributions of pollutants across different community districts and visualize potential correlations between variables. Following this, we developed and trained our regression models, carefully tuning hyperparameters to optimize performance. We evaluated the models using a robust set of metrics including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R² score, and cross-validation scores to ensure reliable and generalizable results. We concluded that: Our analysis revealed several significant findings about the relationship between traffic emissions and PM2.5 levels in New York City. First, we found that traffic-related emissions have a substantial and quantifiable impact on PM2.5 concentrations, with our models explaining a significant portion of the variance in PM2.5 levels. Notably, NO2 emerged as a stronger predictor of PM2.5 levels compared to NO, suggesting that certain types of traffic emissions may have more significant impacts on air quality than others. The spatial analysis revealed hotspots where the relationship between traffic emissions and PM2.5 is particularly strong, often corresponding to areas with high traffic density. These findings provide crucial insights for policymakers, highlighting the importance of targeted traffic management policies and suggesting specific areas where interventions might be most effective in reducing urban air pollution and protecting public health.


## Data
- Code: https://colab.research.google.com/drive/1KB060R_tsXEkppfVi7MSNn2mA84aMhvV?usp=sharing
- Dataset:
- PM2.5 : https://drive.google.com/file/d/1o1ClooaRVEIVjdf0LPbrHsmJjEBvfdMG/view?usp=drive_link
- NO2   : https://drive.google.com/file/d/1Nd-4dSLHyB5qAiBMFIUNLCTntloMfRpW/view?usp=drive_link
- NO    : https://drive.google.com/file/d/16bnMVJRpuHmC4f5JNGMFJaUZ9_xEyHaG/view?usp=drive_link

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

