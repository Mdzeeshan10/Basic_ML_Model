import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime

class Zee_Auto_LR_Model:
    
    
    def __init__(self):
        self.time = 0
        
    def fit(self, x, y, testsize, scale):
        
        self.time = datetime.now().strftime("D%Y-%m-%dT%H-%M-%S")
        
        # Spliting the data in training and testing format
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, random_state=42)
        
        # Scale the train data as per requirement standard ya minmax format
        
        if "standard" == scale:
            scaler = StandardScaler()
            x_train=scaler.fit_transform(x_train)
            x_test=scaler.transform(x_test)
            
            
        elif "minmax" == scale:
            scaler = MinMaxScaler()
            x_train=scaler.fit_transform(x_train)
            x_test=scaler.transform(x_test)
            
        
            
        else:
            x_train = x_train
            x_test = x_test
            
        # Applying the Linear Regression Model to to fit provid training x and y    
        regression=LinearRegression()
        regression.fit(x_train,y_train)
        
        ##prediction to x test data set
        reg_pred=regression.predict(x_test)
        
        test_df = pd.DataFrame(x_test, columns= x.columns.tolist())
        test_df = test_df.reset_index(drop=True)
        pred_df = pd.DataFrame(reg_pred)
        pred_df = pred_df.reset_index(drop=True)
        
        test_df['pred_test'] = pred_df[0]
        test_df['y_test'] = y_test.reset_index(drop=True)
        test_df.to_csv(f"pred_test_{self.time}.csv")
        
        r2 = r2_score(reg_pred,y_test)
        mae = mean_absolute_error(reg_pred,y_test)
        mse = mean_squared_error(reg_pred,y_test)
        rmse = np.sqrt(mean_squared_error(reg_pred,y_test))
        
        # list of Matrics, Value
        nme = ["R2_Sore", "MAE", "MSE", "RMSE"]
        value = [r2, mae, mse, rmse]
         
        # dictionary of lists
        dict = {'Matrics': nme, 'Value': value}
             
        matrics_df = pd.DataFrame(dict)
        
        matrics_df.to_csv(f"Matrics_{self.time}.csv")
        
        

        with open(f"Zee_Auto_model_LR_{self.time}.pkl", "wb") as f:
            pickle.dump(regression, f)
         
        return test_df, matrics_df
        
        
        
        
    