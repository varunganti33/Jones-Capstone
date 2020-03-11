# Financial Statement Audit: Predicting Total Accruals: 
### Modifications to Jones Model made by Varun Ganti
Author: [Varun Ganti](https://github.com/varunganti33)


## Table of Contents

- [Problem Statement](#Problem-Statemet)
- [Executive Summary](#Executive-Summary)
- [Data Dictionary](#Data-Dictionary)
- [Feature Engineering](#Feature-Engineering)
- [Model Selection](#Model-Selection)
- [Conclusion](#Model-Selection)
- [Recommendations](#Reccomendations)
- [Resources](#Resources)



## Problem Statement

During election time, we see the amount of restatements on accruals go up for company, Accrual accounting is required by GAAP and companies have to spend loads of money to adapt to these standards. Our team wants to create various predictive models, such as linear regression, ridge, knn, decision trees, bagging, boosting, and random forests to predict the amount of discretionary accruals for a company, so we can plan the audit process accordingly. The metric to determine which model we deploy during our audit is R2, which will show how much our models explain the variance in the data. We chose to mirror the Jones Model is in order to predict the total net accruals for a company. This model is going to input variables from 2016 Q1-Q4 data into the regression model. After We create the model, we are going to train the model on a prior period data. After choosing the model with the highest r2, we are going to use the model to validate the model on the next period(2017 Q3).
## The Executive Summary

Our team found that the data is readily avaliable on the SEC website with published quarterly financial statments. From our previous audit process, we hypothesized that the difference between the change in revenue and the change in recievables, and the book value of property plant and equipment variables would have the most influence over predicting total accruals. The difference of revenue and recievables will show the amount of income recieved by a company that is not based on recievables(cash). The abundance of variables, forced me to dive deeper into the data in order to find the most optimal combinations in order to find the highest scoring model. We want a high majority of the data to be explained by our model. 


EDA helped our team drastically not only set up a preprocessing plan for our model, but helped us understand the health of the data One example was that “NetCashOperating” showed up high on the correlation heat map, meaning that this variable will be useful in predicting total net accruals. For preprocessing, we chose to drop all null values with na, assuming that those values actually reperesented none of that type. Most of our eda consisted of examining the financial statmenets, and engineer features accordingly to fit our model. We decided to log all numeric columns to normalize the distributions. Furthermore, we had to create lagged variables in order to find the differences of sales and recievables from the prior quarter. On top of that, we needed to engineer our target variable based on the difference between net income and net cash from operating activities. The way companies transition from cash to accrual basis is to find the differences between assets and liabilities and either add that to cash income or subtract from accruals.  

|Model|Type|Train R2|Testing R2|
|---|---|---|---|
|**Model 1**|*Linear*|0.947|0.547| 


Our approach to picking the best model, is by assessing the R2 Score and the cross validation score on the whole data set. We decided to deploy our linear model into the the audit process because of the ability to examine coeficients and continously improve our model. Once given more time, we can examine non cash activites such as depreciation to see how that can affect accruals

## Data Dictionary
- Refer to Sec for Data Description

- 1.SUB is identifies all the EDGAR submissions in the data set, with each row having the unique (primary) key adsh, a 20 character EDGAR Accession Number with dashes in positions 11 and 14.

2.TAG is a data set of all tags used in the submissions, both standard and custom. A unique key of each row is a combination of these fields:

 1)    tag – tag used by the filer

 2)    version – if a standard tag, the taxonomy of origin, otherwise equal to adsh.
3.NUM is a data set of all numeric XBRL facts presented on the primary financial statements. A unique key of each row is a combination of the following fields:

 1)    adsh- EDGAR accession number

 2)    tag – tag used by the filer

 3)    version – if a standard tag, the taxonomy of origin, otherwise equal to adsh.

 4)    ddate - period end date

 5)    qtrs - duration in number of quarters

 6)    uom - unit of measure

 7)    coreg - coregistrant of the parent company registrant (if applicable)
4.PRE is a data set that provides the text assigned by the filer to each line item in the primary financial statements, the order in which the line item appeared, and the tag assigned to it. A unique key of each row is a combination of the following fields:

  1)    adsh – EDGAR accession number

  2)    report – sequential number of report within the statements

  3)    line – sequential number of line within a report.

## Feature Engineering

#### Basic Jones Model


𝑌(𝑇𝑜𝑡𝑎𝑙𝑁𝑒𝑡𝐴𝑐𝑐𝑟𝑢𝑎𝑙𝑠)=𝛽0+𝛽1(1)+𝛽2((𝐶ℎ𝑎𝑛𝑔𝑒𝑖𝑛𝑅𝑒𝑣−𝐶ℎ𝑎𝑛𝑔𝑒𝑖𝑛𝐴𝑅))+𝛽3(𝑃𝑟𝑜𝑝𝑒𝑟𝑡𝑦𝑃𝑙𝑎𝑛𝑡𝐸𝑞𝑢𝑖𝑝𝑚𝑒𝑛𝑡)+𝜀

#### Varuns Jones Model Modified Equation

𝑌(𝑇𝑜𝑡𝑎𝑙𝑁𝑒𝑡𝐴𝑐𝑐𝑟𝑢𝑎𝑙𝑠)=𝛽0+𝛽1(1)+𝛽2((𝐶ℎ𝑎𝑛𝑔𝑒𝑖𝑛𝑅𝑒𝑣−𝐶ℎ𝑎𝑛𝑔𝑒𝑖𝑛𝐴𝑅))+𝛽3(𝑃𝑟𝑜𝑝𝑒𝑟𝑡𝑦𝑃𝑙𝑎𝑛𝑡𝐸𝑞𝑢𝑖𝑝𝑚𝑒𝑛𝑡)+𝛽4(𝐴𝑠𝑠𝑒𝑡𝑠)+𝛽4(𝑅𝑒𝑣𝑒𝑛𝑢𝑒)+𝛽5(𝑁𝑒𝑡𝐶𝑎𝑠ℎ𝑃𝑟𝑜𝑣𝑖𝑑𝑒𝑑𝐵𝑦𝑈𝑠𝑒𝑑𝐼𝑛𝑂𝑝𝑒𝑟𝑎𝑡𝑖𝑛𝑔𝐴𝑐𝑡𝑖𝑣𝑖𝑡𝑖𝑒𝑠𝑃𝑟𝑒𝑠𝑒𝑛𝑡)+𝜀


The audit team created a function to engineer tags in order to fit equation above. We can multiply through average total assets, but its is useful to engineer a lag variable for assets. Total Net Accruals will equal net income - net cash. Net Cash present will be used in the model as a modification. The original jones model is just the first two items up until property plant eq. I decided to modify it slightly and add assets and revenue as I believed this is a huge part of determining Accruals. To find the change of revenue and change in AR we have to use lag variables and the subtract the current from the prior. Once we accomplish that we can create the change in sales- change in accounts rec column for our equation. We dont need to engineer PP&E as the value on the quarterly statement is in the book value

## Model Selection

|Model|Type|Train R2|Testing R2|
|---|---|---|---|
|**Model 1**|*Linear*|0.947|0.547| 
|**Model 2**|*Ridge*|0.949|0.513|
|**Model 4**|*KNN Regressor*|0.5465|0.4800|
|**Model 5**|*Decision Trees*|0.5295|0.3782|
|**Model 6**|*Bagged Decision Trees*|0.5491|0.5026|
|**Model 7**|*Random Forrest Trees*|0.6307|0.4517|
|**Model 8**|*SVM*|-0.0615|0.4517|


Based on R2 scores listed above, We have decided to choose model 1 to deliver to our audit team in predicting total net accruals. This model will be best suited for our auditors to prepare for year end audit planning in the for the 2016 audit during election time. Model 1 does the best in predicting total accrual prices, and more of the variance in the data can be explained by our model. We want to deliver high level tax solutions to our clients with the least amount of residuals.

## Conclusion
In conclusion, the jones model is a way for companies to predict non discretionary accruals across firms using time series data. The basic theory of jones model(Change in sales + PPE) to predict accruals gave me such a low r2 that adding revenue, assets and present cash flow increased the explained variance of the data. The basic jones model does not cover certain non financing activites that have an huge impact on total accruals. To further my outside research, "The inclusion of few factors such as revenue, depreciation expenses, retirement benefit expenses, asset disposal gains/losses with the modified model was very effective in detecting earning management" . Due to time constraints, I made simple modifications by adding assets and revenue to the model but adding non cash activities would help improve the model drastically. Now that we have created generalized functions and complex models, we can make simple modifications to the cleaning and engineering of the data to deploy the modifications of the jones model. Being able to extract industry wide financial data, and predict future companys amounts based off multiple regression. We want to be able to find relationships between relationships, and create models to help conclude on hypothesis and test variables that are in interest of management. If we were to audit big companies, we could use financial data for individual customers to predict certain financial information based off prior data. Although, our model has a lot of variance, that comes to show that we have to continue to find new variables to test to find the least amount of variance. We can look at p values that are lower than .05 which will show the level of responsivness for certain variables. We saw, on average, that the change in revenues and change in accounts recievables have the biggest effect on the change in accruals. We see in many fraud cases, that trends in recievables and sales can be examined to see any overstatements in receivables and revenue. If given more time as our busy season is starting, we want to add non cash financing activites, like depreciation, as variables into the model to see if there is any relationship to accruals. Many companies have different ways to identify tags in the database, so it is useful to extract data from the sec taxonomy to see the most frequent tags used in the industry. This will allow us analysts to find more relationships in accruals across the industry. 

## Recommendation

The Recommendations to improve the model is to add more non cash financing activities such as depreciation, ammortization, and ppe gains and losses. When fitting the model, we are using X features without taking into account GAAP laws and other factors that effect earnings managmenet. Predicting Accruals is a great way to get a deep understanding of what we can expect in the following year. Our audit team will use professional skepticsm in determining high volume accounts. We want to highlight what affects accruals, and how we can predict accruals in order to give a proper scope to our audit team. "The inclusion of few factors such as revenue, depreciation expenses, retirement benefit expenses, asset disposal gains/losses with the modified model was very effective in detecting earning management in the context of Bangladesh." If our team had more time, we can substantially improve the model by adding non cash activities. The additional steps needed for this is to extract the custom tags and map them to the standard tags, to generalize the tagging process and then extract those tags to engineer features for the modified model. Change in Revenue to Recievables and PPE does not explain Total Accruals as much as non cash activites do in the context of earnings management.

## Refrences

[Jones Formula](http://www.studyland.nl/materials/Pdf/EN%20Formulas%20Modified%20Jones%20Model.pdf)

[Modification](https://www.researchgate.net/publication/228429634_Is_Modified_Jones_Model_Effective_in_Detecting_Earnings_Management_Evidence_from_A_Developing_Economy)

[Accounting Reporting Association on Jones Model](http://lib.cufe.edu.cn/upload_files/other/4_20140516025030_9.pdf)
