# imports
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer

from utils import compute_rmse
from encoders import TimeFeaturesEncoder
from encoders import DistanceTransformer
from data import get_data, clean_data
from sklearn.model_selection import train_test_split



class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'),StandardScaler())
        pipe_distance = make_pipeline(DistanceTransformer(), RobustScaler())

        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude','dropoff_longitude']
        time_cols = ['pickup_datetime']
        feat_eng_bloc = ColumnTransformer([('time', pipe_time, time_cols),
                                     ('distance', pipe_distance, dist_cols)])

        self.pipeline = Pipeline(steps=[('feat_eng_bloc', feat_eng_bloc),
                               ('regressor', RandomForestRegressor())])


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    X = df.drop(columns='fare_amount')
    y = df['fare_amount']
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()
    # evaluate
    print(trainer.evaluate(X_test, y_test))
