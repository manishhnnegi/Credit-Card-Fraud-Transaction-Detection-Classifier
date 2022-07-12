import pandas
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation
import pandas as pd

class prediction:

    def __init__(self,path):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()

        if path is not None:
            self.pred_data_val = Prediction_Data_validation(path)

    def predictionFromModel(self):

        try:
            self.pred_data_val.deletePredictionFile()  # deletes the existing prediction file from last run!
            self.log_writer.log(self.file_object, 'Start of Prediction')
            data_getter=data_loader_prediction.Data_Getter_Pred(self.file_object,self.log_writer)
            data=data_getter.get_data()

            #code change
            TRANS_ID=data['TRANSACTION_ID']
            # data=data.drop(labels=['Wafer'],axis=1)

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            is_null_present=preprocessor.is_null_present(data)

            # if missing values are there, replace them appropriately.
            if (is_null_present):
                data = preprocessor.impute_missing_values(data)  # missing value imputation



            # creat new featyre from datetime isweekend and isnight transaction time
            data = preprocessor.DateTime_transformations(data)

            # customer id transformation rolling sum for each id for time window of 1,7,30 days
            data = preprocessor.CustomerID_transformations(data)

            # terminalid transform to get riskscore
            #data = preprocessor.TerminalID_transformations(data)

            # check further which columns do not contribute to predictions
            # if the standard deviation for a column is zero, it means that the column has constant values
            # and they are giving the same output both for good and bad sensors
            # prepare the list of such columns to drop
            #cols_to_drop = preprocessor.get_columns_with_zero_std_deviation(data)

            # drop the columns obtained above
            cols_to_drop = ['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_TIME_SECONDS',
                            'TX_TIME_DAYS']
            data = preprocessor.remove_columns(data, cols_to_drop)

            data = data[['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                              'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                              'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                              'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                              'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                              'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                              'TERMINAL_ID_RISK_30DAY_WINDOW']]
            # drop the columns obtained above
            #data = preprocessor.remove_columns(data, cols_to_drop)
            file_loader = file_methods.File_Operation(self.file_object,self.log_writer)
            model_name = file_loader.find_correct_model_file()
            model = file_loader.load_model(model_name)
            result = list(model.predict(data))
            result = pandas.DataFrame(list(zip(TRANS_ID, result)), columns=['TRANS_ID', 'Prediction'])
            path = "Prediction_Output_File/Predictions.csv"
            result.to_csv("Prediction_Output_File/Predictions.csv", header=True,index = None, mode='a+')
            self.log_writer.log(self.file_object, 'End of Prediction')

        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path, result.head().to_json(orient="records")




