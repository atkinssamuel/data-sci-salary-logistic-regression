# Data Analysis Concepts Learned from Project
## Data Cleaning
### Converting Categorical Data Into Numerical Data
https://pbpython.com/categorical-encoding.html

There are a few options for encoding categorical data into numerical data. Given a categorical column we can choose to either encode each element numerically, encode the column using one-hot encoding, or encode the column using a custom one-hot encoding. 

Suppose the dataset in question is a dataset that contains information regarding which car each respondent owns. The column in question is ```car_type```. This column can take on 3 different values: ```tesla```, ```toyota```, and ```bmw```. The following will demonstrate three different types of encoding methods.

#### Numerical Encoding 
To numerically encode the ```car_type``` category we would assign the possible values that ```car_type``` could be as numerical values. We could encode ```tesla``` as 0, ```toyota``` as 1, and ```bmw``` as 2. The problem with this encoding method is the weighting implied by the numerical values. Are BMW's twice as valuable, heavy, or fast as Toyotas? If the data does not relate to the encoding in any way it is better to use one-hot encoding instead.

#### One-Hot Encoding
To encode a column using one-hot encoding we assign each possible value in a particular column its own column. Then, the values in each of these new columns is either 0 or 1 depending on if that particular entry possesses that property. ```tesla```, ```toyota```, and ```bmw``` would all be separate columns with boolean values in each entry. The drawback of one-hot encoding is the number of columns explodes with the number of unique values in each categorical column. In this rather simple example we added an additional 2 columns. If we repeated this process for a large number of categorical columns with many unique values in each column we would end up with a very large dataset that would take a long time to operate on. A potential solution is delinated by the next method.

#### Customized One-Hot Encoding
Suppose that in our example there are a total of 1000 respondents. 975 respondents own Teslas, 15 own BMWs, and 10 own Toyotas. In this case, it might be wiser to refactor our original ```car_type``` column to a more simple binary column like ```owns_tesla```. This would prevent an explosion of columns and would retain the attractive categorical implications of one-hot encoding. One could also situationally create a column called ```other``` that would encompass all of the other entires for a given dataset to refrain from adding several columns containing little information. Critically thinking about which encoding method you should use is essential for generating accurate models. 