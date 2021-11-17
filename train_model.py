import os
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

cv_search_prefix = 'model_cv_search__'
config = {
    'data_path': './data/pos_neg_reviews.csv',
    'artifacts_save_path': './artifacts/',
    'outcome': 'outcome',
    f'{cv_search_prefix}preprocess__text__stop_words': 'english'
}
config.update(os.environ)

data = pd.read_csv(config['data_path'])
data_field = 'text_data'
outcome = config['outcome']


X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=outcome), data[outcome],
    test_size=0.2, random_state=2021)

preprocess = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), data_field)
    ]
)

pipeline = Pipeline(
    [
        ('preprocess', preprocess),
        ('svc', LinearSVC())
    ]
)

# generate param grid search
grid_params = dict()
# search in config
for k, v in config.items():
    # find appropriate parameter
    if k.startswith(cv_search_prefix):
        k_sub = k[len(cv_search_prefix):]
        try:
            ev = eval(v)
            if isinstance(ev, tuple) or isinstance(ev, list):
                grid_params[k_sub] = ev
            else:
                grid_params[k_sub] = (ev, )
        except:
            grid_params[k_sub] = (v,)

grid_search = GridSearchCV(pipeline, grid_params,
                           scoring='f1_macro',
                           n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

cv_results = pd.DataFrame(grid_search.cv_results_)
test_metrics = classification_report(y_test, grid_search.predict(X_test),
                                output_dict=True)


model_artifact_path = f"{config['artifacts_save_path']}model.joblib"
Path(model_artifact_path).parent.mkdir(parents=True, exist_ok=True)
with open(model_artifact_path, 'wb') as of:
    joblib.dump(grid_search, of)

metrics = []
metrics.append(dict(name='test-f1_macro', numberValue=grid_search.best_score_))
metrics.append(dict(name='train-f1_macro', numberValue=test_metrics['macro avg']['f1-score']))

with open(f"{config['artifacts_save_path']}gradient-model-metadata.json", 'w') as of:
    json.dump(dict(metrics=metrics), of)
