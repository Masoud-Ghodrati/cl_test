# Unit sales prediction 

The aim is to predict the unit sales for different items sold at different stores over a time period. The main data was provided in a Kaggle competition [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data) 

The training data includes dates, store and item information, whether that item was being promoted, as well as the unit sales. Additional files include supplementary information that may be useful in building your models (see [here](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data) for more information).

- **Data overview**  
- **Data visualization**
- **Data prediction**

## Data overview  
In total, the data includes:

|# stores | # states | # cities |# unique items | # item families  | # classes |
|---------|----------|----------|---------------|------------------|-----------|
|54       | 16       |22        | 4100          | 33               | 337       |

The test data have 60 unique items that are not included in the training data. Note that the following analysis was done based on training data just to validate the predictions (i.e., dividing train data to train and validation). To get this summary simply run this code: [`xx.py`](). Note that you need to change the paths and download and unzip the data into paths. Also, as the data is big it might take some time to run the code depending on the computer in use. 

## Data visualization
Let's first visualize the data. This helps us to get an idea about the sales, how stores are distributed, how sales (e.g., item families) are distributed geographically, and how they change over time (e.g., do sales have seasonal dependency, or do holidays affect them and how these are reflected on the type of the items/geographical regions, and etc). I used [Tableau Software](https://www.tableau.com) as it's a nice and quick way of visualization without any coding and loading hassles. Finally, everything can be saved in a Tableau package and used by any user. However, as the data was big the corresponding Tableau package was also big, bigger than the Github upload limit. So, I only include screen shuts here.

The map visualization shows that Favorita has stores in many states (not all though) and the state of Pichincha has significantly higher number of stores (19, in which 18 of them are located in the capital, Quito) than others.

![](images/img_01.png)


The total unit sales are also significantly higher in Pichincha than other states:



![](images/img_03.png)


I also aggregated the total weekly unit sales and visualized across the years to see if there is a pattern/trend in the sale data. Overall, it can be seen that the total weekly unit sales increases during the last few weeks of the year which end in the Christmas holidays. The overall sales across the years also follow season-dependent pattern with an upward trend toward the end of the year. There is also a large decrease in sales during mid-2015, which can be associated with the volcanic eruption in Aug 2015 in the state of Pichincha. Also, the sharp, spiky increase in sales on ~18 April 2016 might be associated with the earthquake on 16 April 2016.


![](images/img_04.png)


Grouping the unit sales by their families shows that Grocery 1  has the highest unit sales followed by Beverages and Produce, and this trend is kept during the whole year. The unit sales trend for items from other families do not fluctuate a lot during the year and have a lower seasonal dependency.


![](images/img_05.png)




















## Data prediction 

