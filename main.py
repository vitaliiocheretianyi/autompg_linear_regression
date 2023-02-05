import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score


sns.set()

# load dataset
df = pd.read_csv("resources/DataSets/auto-mpg.csv")
# drop car name column
df.drop('car name', axis=1, inplace=True)

# replace '?' with NaN values representing a missing data.
df.horsepower = df.horsepower.str.replace('?', 'NaN').astype(float)

# fill the missing data with the mean value of horsepower
df.horsepower.fillna(df.horsepower.mean(), inplace=True)

# let's visualize the distribution of the features of the cars
# df.hist(figsize=(12, 8), bins=20)
# plt.show()

# Let’s visualize the relationships between the Mileage Per Galon(mpg) of a car and the other features.
# plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), cmap=plt.cm.Reds, annot=True)
# plt.title('Heatmap displaying the relationship between\nthe features of the data', fontsize=13)
# plt.show()

X1 = sm.tools.add_constant(df)
# calculate the VIF and make the results a series.
series1 = pd.Series([variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])], index=X1.columns)
# print('Series before feature selection: \n\n{}\n'.format(series1))

# Let's drop the columns that highly correlate with each other
df.drop(['cylinders', 'displacement', 'weight'], axis=1, inplace=True)

# Let's do the variance inflation factor method again after doing a feature selection
# to see if there's still multi-collinearity.
X2 = sm.tools.add_constant(df)
series2 = pd.Series([variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])], index=X2.columns)
# print('Series after feature selection: \n\n{}'.format(series2))

# create a DataFrame of independent variables
X = df.drop('mpg', axis=1)
# create a series of the dependent variable
y = df.mpg

# scaling the feature variables.
X_scaled = preprocessing.scale(X)

# preprocessing.scale() returns a 2-d array not a DataFrame so we make our scaled variables
# a DataFrame.
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# split our data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.2, random_state=0)

# train a GradientBoostingRegressor model
gradient_model = GradientBoostingRegressor()  # instantiate the model

params = {'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9],
          'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9],
          'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
          'learning_rate': [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
          'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900 ]}


gradient_search = RandomizedSearchCV(gradient_model, params, cv=5, n_jobs=-1, n_iter=50)  # initialize the search

gradient_search.fit(X_train, y_train)  # fit the model

gradient_pred = gradient_search.predict(X_test)  # make predictions with the model

# print out the best parameters and score the model
print('Best parameter found:\n{}\n'.format(gradient_search.best_params_))
print('Train score: {}\n'.format(gradient_search.score(X_train, y_train)))
print('Test score: {}\n'.format(gradient_search.score(X_test, y_test)))
print('Overall model accuracy: {}\n'.format(r2_score(y_test, gradient_pred)))
print('Mean Squared Error: {}\n'.format(mean_squared_error(y_test, gradient_pred)))
