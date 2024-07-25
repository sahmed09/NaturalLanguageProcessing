from urllib.request import urlopen
import pandas as pd
import evalml
from evalml import AutoMLSearch

# EvalML is an AutoML library that builds, optimizes, and evaluates machine learning pipelines using domain-specific
# objective functions. (pip install evalml)

input_data = urlopen('https://featurelabs-static.s3.amazonaws.com/spam_text_messages_modified.csv')
data = pd.read_csv(input_data)
print(data.head())

# Independent And Dependent Features
X = data.drop('Category', axis=1)
y = data['Category']
print(X.head())

print(y.value_counts())
print(y.value_counts(normalize=True))

# Train and test data split
X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(X, y, problem_type='binary')
print(type(X_train))
print(X_train.head())

algorithms = evalml.problem_types.ProblemTypes.all_problem_types
print('Algorithms:', algorithms)

automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='binary', max_batches=1, optimize_thresholds=True)
automl.search()
ranks = automl.rankings
print('Rankings:\n', ranks)

# Getting The Best Pipeline
best_pipeline = automl.best_pipeline
print('Best Pipeline:', best_pipeline)

# Let's Check the detailed description
print(automl.describe_pipeline(ranks.iloc[0]['id']))

# Evaluate on the test data
scores = best_pipeline.score(X_test, y_test, objectives=evalml.objectives.get_core_objectives('binary'))
print(f'Accuracy Binary: {scores["Accuracy Binary"]}')

# Evaluate on hold out data
best_pipeline_score = best_pipeline.score(X_test, y_test, objectives=['auc', 'f1', 'precision', 'recall'])
print(best_pipeline_score)
