import pandas as pd
import numpy as np
from scipy.stats import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

def main():
    param_grid = {
        'classifier__criterion': ["gini", "entropy", "log_loss"],
        'classifier__n_estimators': np.arange(100, 200, 30),
        'classifier__min_samples_split': [2, 3],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__min_impurity_decrease': np.arange(0, 0.01, 0.005),
        'classifier__min_weight_fraction_leaf': np.arange(0, 0.01, 0.02),
        'classifier__max_features': ["sqrt", "log2", None],
        'classifier__max_leaf_nodes': np.arange(2, 4, 1),
        #'classifier__oob_score': [True, False],
        'classifier__ccp_alpha': np.arange(0, 0.02, 0.01),
        'classifier__max_samples': np.arange(0, 0.4, 0.2)
    }

    df = pd.read_csv("project_train.csv")
    RANDOM_STATE = 8

    X = df.drop('Label', axis=1)
    y = df.Label

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, y_train = shuffle(X, y, random_state=RANDOM_STATE)

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        # ('pca', PCA(n_components=9)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, df.select_dtypes(include=['float64']).columns)
    ])
    # 847
    classifier = RandomForestClassifier(criterion='entropy',
                                        random_state=RANDOM_STATE,
                                        n_estimators=100,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        min_impurity_decrease=0.005,
                                        max_leaf_nodes=50, max_features=None,
                                        max_depth=12,
                                        bootstrap=True)

    final_classifier = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', classifier)])
    final_classifier.fit(X_train, y_train)

    # Create a GridSearchCV object
    grid_search = GridSearchCV(final_classifier, param_grid,
                               cv=StratifiedKFold(shuffle=True, random_state=RANDOM_STATE, n_splits=10), verbose=2)

    # Fit the GridSearchCV object to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Print the best parameters
    print("Best parameters: ", best_params)

    cval_acc = np.average(cross_val_score(Pipeline(
        steps=[('preprocessor', preprocessor), ('classifier', grid_search.best_estimator_
)]),
        X_train, y_train, cv=StratifiedKFold(shuffle=True, random_state=RANDOM_STATE, n_splits=10)))

    print("Forest accuracy during cross validation: " + str(cval_acc))

main()