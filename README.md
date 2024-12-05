## Machine Learning Models to Analyze the Impact of Traffic Emissions on PM2.5 Levels in New York City
To predict PM2.5 levels in urban environments, an ensemble machine learning approach is applied using linear regression, ridge regression, random forest, and gradient boosting decision trees.



## Introduction 

  Air pollution is one of the leading environmental challenges. In urban areas, traffic emissions are a primary contributor to PM2.5 levels, with pollutants like nitric oxide (NO) and nitrogen dioxide (NO2) playing key roles in the formation of secondary particulate matter, which is posing significant health risks. PM2.5 particles, which are small enough to penetrate deep into the lungs and bloodstream, are linked to respiratory and cardiovascular diseases, and even premature mortality [1].  In New York city, 14 percent of current PM2.5 concentration comes from local vehicle emissions, which place it as the second largest PM2.5 emission source [2]. Specifically, while researches have stated that 4% to 37% of NOx, a major vehicle emission, is known to contribute to the formation of PM2.5 through chemical reactions in the atmosphere, the degree to which reducing traffic emissions might lower PM2.5 concentrations is not straightforward [3]. Traffic emissions regulation is critical for formulating effective air quality management strategies. While local vehicles are something that can go under restrictions and regulate the emission, we can regulate vehicle emissions by turning gasoline and diesel cars into EVs. The problem relies on quantifying this relationship between traffic-related emissions and PM2.5 levels. Using machine learning can help us in understanding this problem in depth by predicting patterns and effects and therefore can inform policies aimed at reducing traffic emissions, eventually improving urban air quality.

  This project focuses on the air quality challenge in New York City. We have datasets from the New York City Environmental Health Data Portal [4]. The datasets include PM2.5, NO, and NO2 levels for 2022. By cleaning the data and using machine learning techniques, we can modelize the trends and analyze the correlations between these pollutants and predict the impact of hypothetical traffic reductions. By using models like linear regression and gradient boosting and applying weighting method, we can quantify the contribution of traffic emissions to PM2.5 and simulate the effects of reduced traffic emissions, for instance, a 5%, 10%, 15% and 90% of reduction in PM2.5, NO and NO2 levels. The results showed a measurable reduction in PM2.5 levels, emphasizing the importance of managing traffic emissions for better air quality. This project in machine learning aims to provides insights of current air quality situation and predict situations under different scenarios, highlighting the value of machine learning in environmental analysis.

## Links
Code: https://colab.research.google.com/drive/1dlk3SE7zYaYOlahEPSb_FWe1lZSBwSi-?usp=sharing

Data: https://drive.google.com/drive/folders/1s9moHgI5mlNNIf35P_yqRgZ-jTYVy5sY?usp=sharing

## Data
Input data features selected for this project are offshore water level (WL), deep water significant wave height (HSo), peak period (TP), peak direction (DP), and foreshore beach slope (b). Off-shore wave conditions (HSo, Tp, and DP) are publicly available at 30 minute intervals through the Coastal Data Information Program's (CDIP, https://cdip.ucsd.edu/) nearest deep water wave buoy (Point Loma South, 191). Water levels at 6 minute intervals are available through the National Oceanic and Atmospheric Administration's (NOAA, https://tidesandcurrents.noaa.gov/) nearest open coast tide gauge (La Jolla, 9410230). 

Beach topography is taken from an ATV survey conducted February 18th, 2015. Foreshore beach slope varies as water levels move up and down the beach with tides and waves, as the beach face is convex. Therefore, an empirical parameterization of mean water level and wave runup presented by Stockdon et al., 2006 [10] is used to determine the range over which the foreshore beach slope is calculated:

![](assets/IMG/eqn2.png)

The groundwater head measured at a pressure sensor buried 30m inland from the shoreline is taken to be the target data. All input and target data are interpolated to match the lowest sample rate of 30 minutes in the CDIP data, and normalized to range from 0 to 1.

![](assets/IMG/DataIn.png)
*Figure 1: Input and target data collected at Imperial beach in February 2015.*


## Setting up your Project Website

### How to copy this site as a template
1. Create [a GitHub account](https://github.com/)
2.	Go to [https://github.com/atmosalex/atmosalex.github.io/](https://github.com/atmosalex/atmosalex.github.io/) and click *Use this template*, then **Create a new repository**. [![screenshot][1]][1]
3.	In the box that says *Repository name*, write your **Github username**, followed by **.github.io**, as shown in the screenshot below. Then click **Create repository** at the bottom. [![screenshot][2]][2]
4.	Go to the *Settings* tab, then click *Pages* (under *Code and automation*). In the *Build and deployment* section, under **Branch**, select "main" and click save (if it isn't already selected). It should look like this: [![screenshot][3]][3]
5.	Click the *Actions* tab at the top of the page and check that the build and deployment action has finished. Once it has, navigate to **[your username].github.io** to see your site, which should be a copy of this one! If you cannot see an *Actions* tab, just wait a few minutes then go to your URL to check it is live.

Now you are ready to customize your site! To add your name to the site, go to your repository page on Github, click `_config.yml`, and edit it to replace the temporary title with your name, etc. When we make changes to a project on Github, we have to **commit** the new version of each file. Github keeps track of all the changes we make, making it easy to roll back (i.e. return the project to a previous commit).

[1]: /assets/IMG/instr_new.png
[2]: /assets/IMG/instr_template.png
[3]: /assets/IMG/instr_bd.png

### How to change the theme (optional)
1.	You can choose any theme [listed on this page](https://pages.github.com/themes/), though some do not work as well on mobile devices.
2.	From GitHub, edit `_config.yml` and replace the `theme:` line with `theme: jekyll-theme-name` where `name` is the name of the theme from the above list. **For the `minima` theme, use a shortened preface like so `theme: minima`**, the others seem to need the whole preface `theme: jekyll-theme-`. You can check the *Actions* tab (as in step 5. above) to make sure the site is building successfully.

### How to change your site logo (optional)
1. Some themes, such as `jekyll-theme-minimal`, show a logo. In your repository, upload a logo or profile picture to the `assets/IMG/` directory
2. Open `_config.yml` and modify the line `logo: /assets/IMG/template_logo.png` to point to your new image

***

## Guide to Adding Content
* Your repository's `README.md` file (the file you are reading now) acts like a home page. Replace its contents with whatever you want the world to see by editing the file on GitHub.
* If you want to turn this page into a CV or blog, etc., it may be useful to refer to a [guide for writing Markdown](https://www.markdownguide.org/basic-syntax/).
* You can create other markdown files (.md) in your repository and navigate to them from this page using links, i.e.: [here is a link to another file, `project.md`](project.md)
* When editing a markdown file on GitHub, it is useful to wrap text by selecting the *Soft wrap* option as shown: ![screenshot](/assets/IMG/instr_wrap.png)
* If you want to get even more technical, you can also write HTML in your .md files, and GitHub Pages will render it. For example, the image below is displayed by writing the following (edit this file to see!): `<img align="right" width="200" height="200" src="/assets/IMG/template_frog.png">`
<img align="right" width="337" height="200" src="/assets/IMG/template_frog.png"> 

***

## Delivering your Project

Your final project is delivered in two components: a report and your code.

### Report

Your report should be **delivered via your website**. Submit a link to your website on BruinLearn so that your instructor can browse it to find your report. 

To make this simple, you can write the report using a word processor or Latex, then export it as a .pdf file and upload it to the `assets` directory. You can then link to it [like so](/assets/project_demo.pdf). However, you can also type the report directly onto the website using another markdown page - [here is](/project.md) a template for that.

### Code

A link to your code must be submitted on BruinLearn, and the course instructor must be able to download your code to mark it. The code could be in a Google Colab notebook (make sure to *share* the notebook so access is set to **Anyone with the link**), or you could upload the code into a separate GitHub repository, or you could upload the code into the `assets` directory of your website and link to it. 
