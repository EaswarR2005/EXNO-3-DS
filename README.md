## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.     
STEP 2:Clean the Data Set using Data Cleaning Process.     
STEP 3:Apply Feature Encoding for the feature in the data set.     
STEP 4:Apply Feature Transformation for the feature in the data set.     
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv('/content/Encoding Data.csv')
df
```
![image](https://github.com/user-attachments/assets/d6d61e54-63f3-4d06-9de0-882c8fb23583)

```
df.shape
```
![image](https://github.com/user-attachments/assets/05d4016f-478d-414a-80e4-e129e8d2ea83)

```
df.info()
```
![image](https://github.com/user-attachments/assets/9dc82d82-a4e9-463b-886a-c59d2ac93008)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[['ord_2']])
```
![image](https://github.com/user-attachments/assets/ef871b61-4f1c-4b3d-8332-eb25112f4318)

```
df['bo_2']=e1.fit_transform(df[['ord_2']])
df
```
![image](https://github.com/user-attachments/assets/2725896e-97d9-4ac1-9ef1-0f961d589a55)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(df['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/75931954-86f4-4bc4-9a09-ba4095ce4145)

```
dfc=df.copy()
dfc['con_2']=le.fit_transform(df['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/a50026c6-88ae-4e43-a2cd-c2199e58d979)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/620a8930-4776-42f8-bdbe-d9aaed736366)

```
pd.get_dummies(df2,columns=['nom_0'])
```
![image](https://github.com/user-attachments/assets/2d761f03-3850-44fc-8059-f495c8f3df35)

```
pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
df=pd.read_csv('/content/data.csv')
df
```
![image](https://github.com/user-attachments/assets/a3a20ece-6eed-49bd-85be-3605ed885457)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
sfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/fbef581a-26e9-42ad-88eb-a20df7cd2253)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new = te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/f3d49002-9bca-40b1-8c72-f094c3360996)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/39e1ce04-18ce-400a-a7d9-0149f0d501fa)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/41054f6d-1ec5-4809-9b1d-78c63f8e25d6)

```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/8fb6b79a-c4c3-4f49-98ee-7bd15833c1fb)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/2d950cc0-30a2-492b-a016-920d85726a4d)

```
np.reciprocal(df['Moderate Negative Skew'])
```
![image](https://github.com/user-attachments/assets/c219963a-7c26-4cd6-b054-d3b57cccafdd)

```
np.sqrt(df['Highly Positive Skew'])
```
![image](https://github.com/user-attachments/assets/eecbacaf-18b1-4566-9268-0d99a9ce5505)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/212a5d4d-f8a3-4feb-8ad0-ac5f4523717d)

```
df['Highly Positive Skew']=np.sqrt(df['Highly Positive Skew'])
df
```
![image](https://github.com/user-attachments/assets/67dee0e9-4a0d-498e-8ca3-15571d2565cf)

```
df['Highly Positive Skew_boxcox'],parameters=stats.boxcox(df['Highly Positive Skew'])
df
```
![image](https://github.com/user-attachments/assets/27dce1d6-d1a3-42cb-9202-c273f862ab55)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/210924ee-e9e7-4ffc-95e1-92f52ae5f2e8)

```
df['Moderate Negative Skew_yeojohnson'],parameters=stats.yeojohnson(df['Moderate Negative Skew'])
df
```
![image](https://github.com/user-attachments/assets/99221505-792d-4b1e-b8e6-9eb44f7d43f4)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/1913dcb3-c9a7-40fb-b90a-e86c89410296)

```
sm.qqplot(np.reciprocal(df['Moderate Negative Skew']),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c56f9a01-85a0-4abd-9350-78d0d567ecfb)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df['Moderate Negative Skew']=qt.fit_transform(df[['Moderate Negative Skew']])
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/eac19277-74a7-4ee1-ba94-5be9149e3492)


# RESULT:
      Thus Feature encodind and transformation process is performed on the given data.

       
