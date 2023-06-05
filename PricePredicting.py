
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import threading
class BostonLinearRegression:
    def __init__(self):
        """
        Constructor will initialize boston housing data to df pandas dataframe

        """
        data_url="https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        self.df=pd.read_csv(data_url)
        self.column_names=self.df.columns
        pf=ProfileReport(self.df)
        t1=threading.Thread(target=self.making_file,args=(pf,))
        t1.start()
        # pf.to_file('static/results/report.html')
    def making_file(self,pf):
        pf.to_file('static/results/report.html')


    def correlationmatrix(self):
        """
        This function saves the correlation matrix as file
        :return:
        """
        try:
            res=self.df.corr()
            res.to_html('static/results/correlation.html')
            # print(res)
        except Exception as e:
            raise Exception(f"(correlationmatrix) something went wrong : {str(e)}")
    def split_df(self,x,y,test_size=0.3,random_state=np.random.randint(1,10000)):
        """
        This function split the dataframes for training and testing
        """
        try:
            self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(x,y,test_size=test_size,random_state=random_state)
            return self.x_train,self.x_test,self.y_train,self.y_test
        except Exception as e:
            raise Exception(f"(split_df) something went wrong : {str(e)}")

    def createModel(self):
        """
        This function creates a Linear Regression model
        :return:
        """
        try:
            self.model=LinearRegression()
            return self.model
        except Exception as e:
            raise Exception(f"(createModel) something went wrong : {str(e)}")

    def normalization(self,x):
        """
        This function normalizes the data passed by using zscore normalization
        :param x:pandas data frame
        :return:
        """
        try:
            self.x=x
            self.norm_model=StandardScaler()
            self.x_norm=self.norm_model.fit_transform(self.x)
            self.x_norm_df = pd.DataFrame(self.x_norm, columns=self.x.columns)
            return self.x_norm_df
        except Exception as e:
            raise Exception(f"(normalization) something went wrong : {str(e)}")


    def trainModel(self):
        """
        This function trains the model created before
        :return:
        """
        try:
            self.model.fit(self.x_norm_df,self.y_train)
            return True
        except Exception as e:
            raise Exception(f"(trainModel) something went wrong : {str(e)}")
    def predict(self,record):
        """
        This function predicts the value
        :param record:
        :return:
        """
        try:
            self.input_normalized=self.norm_model.fit_transform(record)
            self.output=self.model.predict(self.input_normalized)
        except Exception as e:
            raise Exception(f"(predict) something went wrong : {str(e)}")
    def accuracy(self,x_test,y_test):
        """
        This function finds the accuracy unsing rsquare
        :param x_test:
        :param y_test:
        :return:
        """
        try:
            accuracy=self.model.score(x_test,y_test)
            return accuracy
        except Exception as e:
            raise Exception(f"(accuracy) something went wrong : {str(e)}")




