



# Doing the necessary imports
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger
import pandas as pd



class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')


    def trainingModel(self, start_date_training,n_folds,delta_train,delta_delay,delta_assessment):

        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # Getting the data from the source
            data_getter=data_loader.Data_Getter(self.file_object,self.log_writer)
            data=data_getter.get_data()


            preprocessor=  preprocessing.Preprocessor(self.file_object,self.log_writer)



            # check if missing values are present in the dataset
            is_null_present = preprocessor.is_null_present(data)

            # if missing values are there, replace them appropriately.
            if (is_null_present):
                data = preprocessor.impute_missing_values(data)  # missing value imputation

            # creat new featyre from datetime isweekend and isnight transaction time
            data = preprocessor.DateTime_transformations(data)

            # customer id transformation rolling sum for each id for time window of 1,7,30 days
            data = preprocessor.CustomerID_transformations(data)

            # terminalid transform to get riskscore
            data = preprocessor.TerminalID_transformations(data)      #from start date to end date final transformed data 6months






           #modeling
            #data.to_pickle("derid_feature.pkl")
            #data = pd.read_pickle('derid_feature.pkl')
            model_finder=tuner.Model_Finder(self.file_object,self.log_writer) # object initialization       #tuner.

            output_feature = "TX_FRAUD"

            input_features = ['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                              'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                              'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                              'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                              'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                              'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                              'TERMINAL_ID_RISK_30DAY_WINDOW']

            x = data[input_features]
            y = data[output_feature]

            train_df, test_df =  model_finder.get_train_test_set(data, start_date_training, delta_train=7, delta_delay=7, delta_test=7, )

            X_train = train_df[input_features]
            y_train = train_df[output_feature]
            X_test = test_df[input_features]
            y_test = test_df[output_feature]


            # drop the columns obtained above
           # cols_to_drop = ['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_TIME_SECONDS','TX_TIME_DAYS']
            #train_df_n = preprocessor.remove_columns(train_df, cols_to_drop)
            #test_df_n = preprocessor.remove_columns(test_df, cols_to_drop)


           # X_train, y_train, X_test, y_test = model_finder.split_train_test(train_df_n, test_df_n,)


            #getting the best model for each of the clusters
            best_model_name,best_model=model_finder.get_best_model(X_train,y_train,X_test,y_test, x,y,data,start_date_training,n_folds,delta_train,delta_delay,delta_assessment)

            #saving the best model to the directory.
            file_op = file_methods.File_Operation(self.file_object,self.log_writer)                  #file_methods.
            save_model=file_op.save_model(best_model,best_model_name)

            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()


        except Exception as e:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training',e)
            self.file_object.close()
            raise Exception






















