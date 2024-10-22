import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('car.csv')

data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')

data.dropna(subset=['horsepower'], inplace=True)

data.dropna(subset = ['car name'], inplace=True)
data = data[data['car name'].str.strip() != '']

# profile = ProfileReport(data, title = "Car Report")
#
# profile.to_file('car_report.html')

target = "acceleration"
x = data.drop(target, axis=1)
y = data[target]

x = data.drop('mpg', axis = 1)
x = data.drop('weight', axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

processor = ColumnTransformer(transformers=[
    ("num_features", StandardScaler(), ["horsepower", "cylinders", "displacement", 'model year', "origin"]),
    ("nominal_feature", TfidfVectorizer(), "car name")
])

param_grid = {
    'modeling__n_estimators': [50, 100, 150, 200],
    'modeling__criterion': ['squared_error', 'poisson', 'absolute_error'],
    'modeling__max_depth': [1, 3, 5, 7]
}

model  = Pipeline([
    ("preprocessor",processor),
    ("modeling", RandomForestRegressor(n_estimators= 150, criterion= 'poisson', max_depth= 7 ,random_state=42)),
])
#
# grid_search = GridSearchCV(model, param_grid =  param_grid, scoring = 'neg_mean_squared_error', verbose = 5)
# SVR: 3.03
#Lasso: 3.82
#RandomForest: 2.90

model.fit(x_train, y_train)

y_prediction = model.predict(x_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_prediction))

#After using gridsearchcv and then we find best parameters
#MSE decreased to 2.67





