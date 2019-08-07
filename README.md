# Unit sale prediction 

The aim is to predict the unit sales for different items sold at different stores over a time period. The main data was provided in a Kaggle competition [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data) 

The training data includes dates, store and item information, whether that item was being promoted, as well as the unit sales. Additional files include supplementary information that may be useful in building your models (see [here](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data) for more information).

- Data overview  
- Data visualization
- Data prediction 

## Data overview  
In total, the data includes:

|# stores | # states | # cities |# unique items | # item families  | # classes |
|---------|----------|----------|---------------|------------------|-----------|
|54       | 16       |22        | 4100          | 33               | 337       |

The test data have 60 unique items that are not included in the training data. Note that the following analysis was done based on training data just to validate the predictions (i.e., dividing train data to train and validation). To get this summary simply run this code: [`xx.py`](). Note that you need to change the paths and download and unzip the data into paths. Also, as the data is big it might take some time to run the code depending on the computer in use. 

## Data visualization
Let's first visualize the data. This helps us to get an idea about the sales, how stores are distributed, how sales (e.g., item families) are distributed geographically, and how they change over time (e.g., do sales have seasonal dependency, or do holidays affect them and how these are reflected on the type of the items/geographical regions, and etc). I used [Tableau Software](https://www.tableau.com) as it's a nice and quick way of visualization without any coding and loading hassles. Finally, everything can be saved in a Tableau package and used by any user. However, as the data was big the corresponding Tableau package was also big, bigger than the Github upload limit. So, I only include screen shuts here.

![](images/img_01.png)

## Data prediction 

