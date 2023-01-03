"""
1. load train,test data
2. train algorithm
3. save metrics and params
"""
import os 
import pandas as pd
import argparse
import json
import joblib
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from get_data import read_params


def eval_quantities(actual,pred):

    rmse = mean_squared_error(actual,pred)
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual,pred)

    return rmse,mae,r2

def train_and_evaluate(config_path):
    config  = read_params(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    alpha = config["estimators"]["ElasticNet"]['params']['alpha']
    li_ratio = config["estimators"]["ElasticNet"]['params']['li_ratio']
    target = [config['base']['target_col'] ]
    scores_file = config['reports']['scores']
    params_file = config['reports']['params']

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)
    
    train_y = train[target]
    test_y = test[target]

    train_x =  train.drop(target,axis=1)
    test_x = test.drop(target,axis=1)

    lr = ElasticNet(alpha=alpha,random_state=random_state)
    lr.fit(train_x,train_y)

    predict_quantities = lr.predict(test_x)
    print(predict_quantities.shape,test_y.shape)
    (rmse,mae,r2) = eval_quantities(test_y,predict_quantities)
    print(rmse,mae,r2)
    with open(scores_file,'w') as f:
        scores = {
            'rmse':rmse,
            'mae':mae,
            'r2':r2
        }
    with open(params_file,'w') as f:
        scores = {
            'rmse':rmse,
            'mae':mae,
            'r2':r2
        }
    joblib.dump(lr, os.path.join(model_dir,'model.pkl'))

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--config",default='params.yaml')
    parsed_args = args.parse_args()

    train_and_evaluate(parsed_args.config)

