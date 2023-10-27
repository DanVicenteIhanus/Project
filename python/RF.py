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

def main():

    df = pd.read_csv("project_train.csv")
    RANDOM_STATE = 8

    X = df.drop('Label', axis=1)
    y = df.Label

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, y_train = shuffle(X, y, random_state=RANDOM_STATE)


    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        #('pca', PCA(n_components=9)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, df.select_dtypes(include=['float64']).columns)
    ])

    classifier = RandomForestClassifier(criterion='entropy',
                                        random_state=RANDOM_STATE,
                                 n_estimators=100,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 min_impurity_decrease=0.005,
                                 max_leaf_nodes=50,
                                 max_depth=12,
                                 bootstrap=True)

    cval_acc = np.average(cross_val_score(Pipeline(
        steps=[('preprocessor', preprocessor),('classifier', classifier)]),
        X_train, y_train, cv=StratifiedKFold(shuffle=True, random_state=RANDOM_STATE, n_splits=10)))


    final_classifier = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', classifier)])
    final_classifier.fit(X_train, y_train)

    df_test = pd.read_csv("project_test.csv")
    X_test = pd.DataFrame(scaler.fit_transform(df_test), columns=df_test.columns)
    predictions = final_classifier.predict(X_test)
    predictions_train = final_classifier.predict(X)

    print("Forest accuracy during cross validation: " + str(cval_acc))

main()
