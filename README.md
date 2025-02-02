# Titanic: Machine Learning from Disaster

* **Load Dataset**
* **Feature engineering and data preproccessing**
* **Data visualization**
* **Prepare dataset for modeling**
* **Modeling**
* **Testing**


```python
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set() # setting seaborn default for plots

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
```

## Step 1. Load dataset

We use pandas to import our traning and test datasets, then we'll try to understand the data. 


```python
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

print('Train shape:',train.shape)
print('Test shape:', test.shape)
```

    Train shape: (891, 12)
    Test shape: (418, 11)
    


```python
train.info()
train.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



### Data Dictionary 

 * Rows and columns :  We can see that there are 891 rows and 12 columns in our training set. Test set contains 418 rows and 11 columns.
   
| Variable    | Definition                             | Key                                         |
|-------------|----------------------------------------|---------------------------------------------|
| survival    | Survival                               | 0 = No, 1 = Yes                             |
| pclass      | Ticket class                           | 1 = 1st, 2 = 2nd, 3 = 3rd                   |
| sex         | Sex                                    |                                             |
| Age         | Age in years                           |                                             |
| sibsp       | # of siblings / spouses aboard the Titanic |                                             |
| parch       | # of parents / children aboard the Titanic |                                             |
| ticket      | Ticket number                          |                                             |
| fare        | Passenger fare                         |                                             |
| cabin       | Cabin number                           |                                             |
| embarked    | Port of Embarkation                    





## Step 2. Feature engineering and data preproccessing

Now we'll try to analyze the input to see if we need some preprocessing steps in our data:

* Descriptive Statistics. Use `train.describe()` to see the statistical properties of the data
* Handle missing values. Find missing values using `train.isnull().sum`. 
* Encode categorical variables. We use `train['column_name'].map()` function to map a categorical features to numeric.
* Remove unneccesary columns.
* Scale numerical features


```python
# Descriptive statistics
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Handle missing values, train set
train.isnull().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64



We can see that Age values is missing for 177 rows from training set, Cabin values are also missing in many rows, 687 and 2 rows missing Embarked information.


```python
bar_chart('Sex')
```


    
![png](Titanic%20Survival%20Prediction%20Model_files/Titanic%20Survival%20Prediction%20Model_12_0.png)
    


The Chart confirms **Women** more likely survivied than **Men**


```python
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
```


```python
bar_chart('Pclass')
```


    
![png](Titanic%20Survival%20Prediction%20Model_files/Titanic%20Survival%20Prediction%20Model_15_0.png)
    


The Chart confirms **1st class** more likely survivied than other classes

The Chart confirms **3rd class** more likely dead than other classes


```python
bar_chart('SibSp')
```


    
![png](Titanic%20Survival%20Prediction%20Model_files/Titanic%20Survival%20Prediction%20Model_17_0.png)
    


The Chart confirms a person aboarded with **more than 2 siblings or spouse** more likely survived

The Chart confirms **a person aboarded without siblings or spouse** more likely dead


```python
bar_chart('Parch')
```


    
![png](Titanic%20Survival%20Prediction%20Model_files/Titanic%20Survival%20Prediction%20Model_19_0.png)
    


The Chart confirms a person aboarded from **C** slightly more likely survived

The Chart confirms a person aboarded from **Q** more likely dead

The Chart confirms a person aboarded from **S** more likely dead

#### Handle missing values 


```python
# Combine train and test dataset
train_test = [train,test]

for data in train_test:
    data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False) #extract title from name

```


```python
# Map categorical features to numbers

sex_mapping = {"male": 0, "female": 1}
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for data in train_test:
    data['Sex'] = data['Sex'].map(sex_mapping)
    data['Title'] = data['Title'].map(title_mapping)
```


```python
# Median imputation for 'Age' adn 'Fare'. Fill missing 'Embarked' values with 'S' value.
train['Age'] = train['Age'].fillna(train.groupby(['Title','Pclass'])['Age'].transform('median'))
test['Age'] = test['Age'].fillna(test.groupby(['Title', 'Pclass'])['Age'].transform('median'))

train['Fare'] = train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"))
test['Fare'] = test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"))

train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].fillna('S')
```


```python
# Median imputation for 'Age'
test['Age'] = test['Age'].fillna(train.groupby(['Sex', 'Pclass'])['Age'].transform('median'))
test['Fare'] = test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"))

#Remove 'Cabin' column, too many missing values
test.drop(columns=['Cabin'], inplace = True)

#Fill missing values to 'Embarked' with mode
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])

test.isnull().sum()
```


```python
#Remove 'Cabin' column, too many missing values
train.drop(columns=['Cabin', 'Name'], inplace = True)
test.drop(columns=['Cabin', 'Name'], inplace = True)
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### Scale numerical features


```python
#Scale numerical features

scaler = StandardScaler()
train['Fare'] = scaler.fit_transform(train[['Fare']])
test['Fare']  = scaler.fit_transform(test[['Fare']])
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>-0.502445</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>0.786845</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>-0.488854</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>0.420730</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>-0.486337</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### Feature engineering

Feature engineering is the process of using domain knowledge to create new features or transform existing features in a dataset to improve the performance of a machine learning model. For the Titanic dataset, which is often used for binary classification (predicting survival or not), feature engineering can significantly impact model performance.


```python
# Create new feature 'Family_size'
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

## Family mapping
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for data in train_test:
    data['FamilySize'] = data['FamilySize'].map(family_mapping)
```


```python
# Remove unneccesary columns
train.drop(columns=['PassengerId','SibSp', 'Parch','Ticket'], inplace=True)
test.drop(columns=['SibSp', 'Parch','Ticket'], inplace=True )
```

### Step 3. Data visualization 


```python
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',fill= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
 
plt.show() 
```


    
![png](Titanic%20Survival%20Prediction%20Model_files/Titanic%20Survival%20Prediction%20Model_34_0.png)
    



```python
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'FamilySize',fill= True)
facet.set(xlim=(0, train['FamilySize'].max()))
facet.add_legend()
 
plt.show() 
```


    
![png](Titanic%20Survival%20Prediction%20Model_files/Titanic%20Survival%20Prediction%20Model_35_0.png)
    



```python
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',fill= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
 
plt.show() 
```


    
![png](Titanic%20Survival%20Prediction%20Model_files/Titanic%20Survival%20Prediction%20Model_36_0.png)
    


### Step 5. Prepare dataset for modeling

For a better evaluation of the model we'll create 3 datasets: for training, cross validation and test.


```python
# Prepare dataset for modeling
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
```


```python
X_test = test.drop('PassengerId', axis=1)
y_test = pd.read_csv("input/gender_submission.csv")
y_test = y_test.drop('PassengerId', axis=1)

X_test, y_test
```




    (     Pclass  Sex   Age      Fare  Embarked  Title  FamilySize
     0         3    0  34.5 -0.497071         2      0         0.0
     1         3    1  47.0 -0.511934         0      2         0.4
     2         2    0  62.0 -0.463762         2      0         0.0
     3         3    0  27.0 -0.482135         0      0         0.0
     4         3    1  22.0 -0.417159         0      2         0.8
     ..      ...  ...   ...       ...       ...    ...         ...
     413       3    0  25.0 -0.493113         0      0         0.0
     414       1    1  39.0  1.314555         1      3         0.0
     415       3    0  38.5 -0.507453         0      0         0.0
     416       3    0  25.0 -0.493113         0      0         0.0
     417       3    0   7.0 -0.236647         1      3         0.8
     
     [418 rows x 7 columns],
          Survived
     0           0
     1           1
     2           0
     3           0
     4           1
     ..        ...
     413         0
     414         1
     415         0
     416         0
     417         0
     
     [418 rows x 1 columns])



### Step 6. Moddeling

* Import libraries
* Cross validation
* Decision Tree Model
* Random Forest Model
* Evaluate each model


```python
# Importing Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
```


```python
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
scoring = 'accuracy'
```

#### Random Forest Algorithm


```python
# Define the parameter grid
param_grid_rf = {
    'n_estimators': [10, 20, 50],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [3, 4, 5, 6, 7, 8, 10, 11],
    'criterion': ['gini', 'entropy']
}

# Create the RandomForestClassifier
rf = RandomForestClassifier()

# Set up GridSearchCV with KFold
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=k_fold, scoring='accuracy', n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)
best_clf_rf = grid_search.best_estimator_

# Make predictions
y_pred = best_clf_rf.predict(X_test)
```


```python
# Evaluate Random Forest
average_method = 'micro'  # Change as needed
print("Precision: ", precision_score(y_test, y_pred, average='binary'))
print("Recall: ", recall_score(y_test, y_pred, average='binary'))
print("F1 Score: ", f1_score(y_test, y_pred, average='binary'))

score = cross_val_score(best_clf_rf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
round(np.mean(score)*100, 2)
```

    Precision:  0.881578947368421
    Recall:  0.881578947368421
    F1 Score:  0.881578947368421
    




    83.5



#### Decision Tree Algorithm 


```python
# Define the parameter grid for GridSearchCV
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30,40],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 5, 10, 20, 50]
}

# Create the DecisionTreeClassifier
dt = DecisionTreeClassifier()

# Set up GridSearchCV with KFold
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid_dt, cv=k_fold, scoring='accuracy', n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)
best_clf_dt = grid_search.best_estimator_

# Make predictions
y_pred = best_clf_dt.predict(X_test)
```


```python
# Evaluate Decision Tree
average_method = 'macro'  # Change as needed
print("Precision: ", precision_score(y_test, y_pred, average='binary'))
print("Recall: ", recall_score(y_test, y_pred, average='binary'))
print("F1 Score: ", f1_score(y_test, y_pred, average='binary'))

score = cross_val_score(best_clf_dt, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
round(np.mean(score)*100, 2)
```

    Precision:  0.8627450980392157
    Recall:  0.868421052631579
    F1 Score:  0.8655737704918033
    




    83.28



#### Logistic Regression


```python
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # 'liblinear' is used for small datasets, and supports 'l1' penalty
}

# Create the Logistic Regression model
lr = LogisticRegression()

# Set up GridSearchCV with KFold
grid_search_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=k_fold, scoring='accuracy', n_jobs=-1)

# Fit the model
grid_search_lr.fit(X_train, y_train)
best_clf_lr = grid_search_lr.best_estimator_

# Make predictions
y_pred_lr = best_clf_lr.predict(X_test)

# Evaluate Logistic Regression
print("Logistic Regression Evaluation:")
print("Precision: ", precision_score(y_test, y_pred_lr, average='binary'))
print("Recall: ", recall_score(y_test, y_pred_lr, average='binary'))
print("F1 Score: ", f1_score(y_test, y_pred_lr, average='binary'))

score = cross_val_score(best_clf_lr, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
round(np.mean(score)*100, 2)
```

    Logistic Regression Evaluation:
    Precision:  0.8895705521472392
    Recall:  0.9539473684210527
    F1 Score:  0.9206349206349206
    




    81.82



#### XGBoost Classifier


```python
# Define a refined parameter grid for XGBoost
param_grid_xgb_refined = {
    'n_estimators': [10, 15],    # Fine-tune this range based on initial results
    'max_depth': [4, 5, 6, 7, 8],           # Choose values around the best performing ones
    'learning_rate': [0.01, 0.05, 0.1, 0.2], # Try different learning rates
    'subsample': [0.8, 0.9, 1.0],           # Try different subsampling ratios
    'colsample_bytree': [0.8, 0.9, 1.0]     # Try different column sampling ratios
}

# Create the XGBClassifier
xgb = XGBClassifier(eval_metric='logloss')

# Set up GridSearchCV with KFold
grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=k_fold, scoring='accuracy', n_jobs=-1)

# Fit the model
grid_search_xgb.fit(X_train, y_train)
best_clf_xgb = grid_search_xgb.best_estimator_

# Make predictions
y_pred_xgb = best_clf_xgb.predict(X_test)
```


```python
# Evaluate XGBoost
print("XGBoost Evaluation:")
print("Precision: ", precision_score(y_test, y_pred_xgb, average='binary'))
print("Recall: ", recall_score(y_test, y_pred_xgb, average='binary'))
print("F1 Score: ", f1_score(y_test, y_pred_xgb, average='binary'))

score = cross_val_score(best_clf_lr, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
round(np.mean(score)*100, 2)
```

    XGBoost Evaluation:
    Precision:  0.868421052631579
    Recall:  0.868421052631579
    F1 Score:  0.868421052631579
    




    81.82



#### SVM


```python
# Create the SVM classifier
svm = SVC()
# Fit the model
svm.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm.predict(X_test)
```


```python
# Evaluate SVM
print("SVM Evaluation:")
print("Precision: ", precision_score(y_test, y_pred_svm, average='binary'))
print("Recall: ", recall_score(y_test, y_pred_svm, average='binary'))
print("F1 Score: ", f1_score(y_test, y_pred_svm, average='binary'))

score = cross_val_score(svm, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
round(np.mean(score)*100, 2)
```

    SVM Evaluation:
    Precision:  0.7340425531914894
    Recall:  0.45394736842105265
    F1 Score:  0.5609756097560976
    




    74.19



#### KNN (K-Nearest Neighbors)


```python
# Define the parameter grid for KNN
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Create the KNN classifier
knn = KNeighborsClassifier()

# Set up GridSearchCV with StratifiedKFold
grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=k_fold, scoring='accuracy', n_jobs=-1)

# Fit the model
grid_search_knn.fit(X_train, y_train)
best_clf_knn = grid_search_knn.best_estimator_

# Make predictions
y_pred_knn = best_clf_knn.predict(X_test)
```


```python
# Evaluate KNN
print("KNN Evaluation:")
print("Precision: ", precision_score(y_test, y_pred_knn, average='binary'))
print("Recall: ", recall_score(y_test, y_pred_knn, average='binary'))
print("F1 Score: ", f1_score(y_test, y_pred_knn, average='binary'))

score = cross_val_score(best_clf_knn, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
round(np.mean(score)*100, 2)
```

    KNN Evaluation:
    Precision:  0.7417218543046358
    Recall:  0.7368421052631579
    F1 Score:  0.7392739273927392
    




    81.48



### Step 7. Testing


```python
X_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>FamilySize</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0</td>
      <td>34.5</td>
      <td>-0.497071</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>1</td>
      <td>47.0</td>
      <td>-0.511934</td>
      <td>0</td>
      <td>2</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>62.0</td>
      <td>-0.463762</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>27.0</td>
      <td>-0.482135</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>-0.417159</td>
      <td>0</td>
      <td>2</td>
      <td>0.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
prediction = best_clf_knn.predict(X_test)
```


```python
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)
```


```python
submission = pd.read_csv('submission.csv')
submission.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```
