import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import f1_score, make_scorer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

directory='data/'

# Load the data
labels = pd.read_csv(directory + 'train_labels.csv')
features = pd.read_csv(directory + 'train_values.csv')

# Data wrangling
labels.set_index('building_id', inplace=True)
features.set_index('building_id', inplace=True)

# Select only numerical features
num_features = features.select_dtypes(include=np.number)

# Split train and test (dev)
X_train, X_test, y_train, y_test = train_test_split(num_features, labels, test_size=0.20, random_state=33)

# Create pipeline
pipeline_xgb = Pipeline(
    [('scale', StandardScaler()),
     ('feature_reduction', PCA(n_components=10)),
     ('xgb', xgb.XGBClassifier())
])

# Set the parameters
params={}
params['xgb__learning_rate'] = [0.08]
params['xgb__objective'] = ['multi:softmax']
params['xgb__num_class'] = [3]
params['xgb__max_depth'] = [5]

# GridSearch
CV = GridSearchCV(pipeline_xgb, params, scoring = 'f1_micro', n_jobs= 1, cv=3)
CV.fit(X_train, y_train.values.ravel())

#y_pred = CV.predict(X_test)
print('caca')


# TEST SUBMISSION
test_values = pd.read_csv(directory + 'test_values.csv')
test_values.set_index('building_id', inplace=True)
num_features_test = test_values.select_dtypes(include=np.number)

results_test = CV.predict(num_features_test)
results_test = pd.DataFrame(results_test)

building_id = test_values.index
building_id = list(building_id)

results_test['building_id'] = building_id
results_test.rename(columns={0:'damage_grade'}, inplace=True)
results_test = results_test[['building_id', 'damage_grade']]

results_test.to_csv('submission_ML_pipeline.csv', index=False)