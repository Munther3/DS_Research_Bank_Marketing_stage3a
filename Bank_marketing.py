import numpy as np
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_density, geom_line, geom_point, ggtitle
import math

# Modeling preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline

# Modeling and resampling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
import pandas as pd
import tensorflow as tf

# access data
ames = pd.read_csv("bank.csv")

# initial dimension
ames.shape


# first few observations
ames.head()

train, test = train_test_split(ames, test_size=0.3, random_state=123)

f"raw data dimensions: {ames.shape}; training dimensions: {train.shape}; testing dimensions:  {test.shape}"

(ggplot(train, aes('Deposit'))
 + geom_density()
 + geom_density(data = test, color = "red")
 + ggtitle("Random sampling with SciKit-Learn"))


 
y = attrition["age"]
train_strat, test_strat = train_test_split(age, test_size=0.3, random_state=123, stratify=y)



# response distribution for raw data
attrition["age"].value_counts(normalize=True)

# response distribution for training data
train_strat["Attrition"].value_counts(normalize=True)


# response distribution for test data
test_strat["Attrition"].value_counts(normalize=True)


# Sample data
train, test = train_test_split(ames, test_size=0.3, random_state=123)

# Extract features and response
features = train.drop(columns="Sale_Price")
label = train["Sale_Price"]



# SciKit-Learn does not automatically transform categorical features so we need to 
# apply a one-hot transformer. We will discuss this more thoroughly in the next chapter.
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('cat', categorical_transformer, selector(dtype_include="object"))])

knn_fit = Pipeline(steps=[('preprocessor', preprocessor),
                          ('knn', KNeighborsRegressor(metric='euclidean'))])

                          
# Specify resampling strategy
cv = RepeatedKFold(n_splits=10, n_repeats=5)

# Create grid of hyperparameter values
hyper_grid = {'knn__n_neighbors': range(3, 26)}

# Tune a knn model using grid search
grid_search = GridSearchCV(knn_fit, hyper_grid, cv=cv, scoring='neg_mean_squared_error')
results = grid_search.fit(features, label)


# Best model's cross validated RMSE
math.sqrt(abs(results.best_score_))

# Best model's k value
results.best_estimator_.get_params().get('knn__n_neighbors')

# Plot all RMSE results
all_rmse = pd.DataFrame({'k': range(3, 26), 'RMSE': np.sqrt(np.abs(results.cv_results_['mean_test_score']))})

(ggplot(all_rmse, aes(x='k', y='RMSE'))
 + geom_line()
 + geom_point()
 + ggtitle("Repeated CV Results"))


