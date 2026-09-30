import pandas as pd 
import joblib 


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def upper_lower(data):
    
    q1=data.quantile(0.25)
    q2=data.quantile(0.50)
    q3=data.quantile(0.75)


    iqr = q3 - q1 

    lower_fence = q1 - (1.5*iqr)
    upper_fence = q3 + (1.5*iqr)

    return lower_fence,upper_fence



df = pd.read_csv(r"D:\Tops Technowlogy\Data Scince\Ml\linear_regression\temperature_linear_regression_data.csv")
temp_lower_fence,temp_upper_fence=upper_lower(df['Temperature_C'])
ice_lower,ice_upper = upper_lower(df['IceCream_Sales'])

df['IceCream_Sales'] = df['IceCream_Sales'].clip(
    upper=ice_upper,
    lower=ice_lower
)

df['Temperature_C'] = df['Temperature_C'].clip(
    upper=temp_upper_fence,
    lower=temp_lower_fence
)

df['Temperature_C'] = df['Temperature_C'].fillna(
    df['Temperature_C'].median()
)

df['IceCream_Sales'] = df['IceCream_Sales'].fillna(
    df['IceCream_Sales'].median()
)


x = df[['Temperature_C']]
y = df['IceCream_Sales']


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=32)

model = LinearRegression()

model.fit(x_train,y_train)


joblib.dump(model,"model.pkl")
print("model trained and saved successfully")


