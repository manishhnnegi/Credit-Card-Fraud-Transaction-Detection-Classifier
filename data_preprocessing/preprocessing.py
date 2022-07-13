import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import datetime

class Preprocessor:
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_columns(self, data, columns):
        """
                        Method Name: remove_columns
                        Description: This method removes the given columns from a pandas dataframe.
                        Output: A pandas DataFrame after removing the specified columns.
                        On Failure: Raise Exception



                """
        self.logger_object.log(self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        try:
            useful_data = data.drop(labels=columns, axis=1)
            self.logger_object.log(self.file_object, 'removed column successfully')

            return useful_data

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in remove_columns method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()

    def separate_label_feature(self, data, label_column_name):
        """
                                Method Name: separate_label_feature
                                Description: This method separates the features and a Label Coulmns.
                                Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                                On Failure: Raise Exception

                        """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            X = data.drop(labels=label_column_name, axis=1)
            #X.drop(labels=['TRANSACTION_ID','TX_DATETIME','CUSTOMER_ID','TERMINAL_ID','TX_TIME_SECONDS','TX_TIME_DAYS','TX_FRAUD'], axis=1, inplace =True)
            y = data[label_column_name]
            return X, y
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def is_null_present(self, data):
        """
                                               Method Name: is_null_present
                                               Description: This method checks whether there are null values present in the pandas Dataframe or not.
                                               Output: Returns a Boolean Value. True if null values are present in the DataFrame, False if they are not present.
                                               On Failure: Raise Exception


                                """
        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        try:
            null_present = False
            null_counts = data.isna().sum()
            for i in null_counts:
                if i > 0:
                    null_present = True
                    break
            if (null_present):  # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.isna().sum().index
                dataframe_with_null['missing values '] = data.isna().sum().values
                # dataframe_with_null.to_csv('preprocessing_data/null_values.csv')
            self.logger_object.log(self.file_object,
                                   'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return null_present
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def impute_missing_values(self, data):
        """
                                                Method Name: impute_missing_values
                                                Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
                                                Output: A Dataframe which has all the missing values imputed.
                                                On Failure: Raise Exception


                             """
        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        try:
            new_array = data.dropna()
            return new_data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()

    def DateTime_transformations(self, data):
        """
                                Method Name: DateTime_transformations
                                Description: This method transforms DateTime_transformations
                                Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                                On Failure: Raise Exception



                        """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:

            new_data = data.copy()
            # weekday (value 0) or a weekend (1)
            def is_weekend(tx_datetime):
                if tx_datetime.weekday() >= 5:
                    return 1
                else:
                    return 0

            new_data['TX_DURING_WEEKEND'] = new_data.TX_DATETIME.apply(is_weekend)

            # day (0) or during the night (1)
            def is_night(tx_datetime):
                if tx_datetime.hour <= 6:
                    return 1
                else:
                    return 0

            new_data['TX_DURING_NIGHT'] = new_data.TX_DATETIME.apply(is_night)

            return new_data

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def CustomerID_transformations(self, data):
        """
                                Method Name: CustomerID_transformations
                                Description: This method transforms CustomerID_transformations
                                Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                                On Failure: Raise Exception



                        """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            new_data = data.copy()

            def get_customer_spending_behaviour_features(dfx, windows_size_in_days=[1, 7, 30]):

                # Let us first order transactions chronologically
                dfx = dfx.sort_values('TX_DATETIME')

                # The transaction date and time is set as the index, which will allow the use of the rolling function
                dfx.index = dfx.TX_DATETIME

                # For each window size
                for i in windows_size_in_days:
                    # Compute the sum of the transaction amounts and the number of transactions for the given window size
                    tsum = dfx['TX_AMOUNT'].rolling(str(i) + 'd').sum()
                    tcount = dfx['TX_AMOUNT'].rolling(str(i) + 'd').count()

                    # Compute the average transaction amount for the given window size
                    # NB_TX_WINDOW is always >0 since current transaction is always included
                    avg = tsum / tcount

                    # Save feature values
                    dfx['CUSTOMER_ID_NB_TX_' + str(i) + 'DAY_WINDOW'] = list(tcount)
                    dfx['CUSTOMER_ID_AVG_AMOUNT_' + str(i) + 'DAY_WINDOW'] = list(avg)

                # Reindex according to transaction IDs
                dfx.index = dfx.TRANSACTION_ID

                # And return the dataframe with the new features
                return dfx

            new_data = new_data.groupby('CUSTOMER_ID').apply(
                lambda x: get_customer_spending_behaviour_features(x, windows_size_in_days=[1, 7, 30]))

            new_data = new_data.sort_values('TX_DATETIME').reset_index(drop=True)

            return new_data

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()


    def TerminalID_transformations(self, data):
        """
                                Method Name: TerminalID_transformations
                                Description: This method transforms terminalid
                                Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                                On Failure: Raise Exception


                        """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            new_data = data.copy()

            def get_count_risk_rolling_window(dfx, delay_period=7, windows_size_in_days=[1, 7, 30], feature="TERMINAL_ID"):

                dfx = dfx.sort_values('TX_DATETIME')

                dfx.index = dfx.TX_DATETIME

                NB_FRAUD_DELAY = dfx['TX_FRAUD'].rolling(str(delay_period) + 'd').sum()
                NB_TX_DELAY = dfx['TX_FRAUD'].rolling(str(delay_period) + 'd').count()

                for i in windows_size_in_days:
                    NB_FRAUD_DELAY_WINDOW = dfx['TX_FRAUD'].rolling(str(delay_period + i) + 'd').sum()
                    NB_TX_DELAY_WINDOW = dfx['TX_FRAUD'].rolling(str(delay_period + i) + 'd').count()

                    NB_FRAUD_WINDOW = NB_FRAUD_DELAY_WINDOW - NB_FRAUD_DELAY
                    NB_TX_WINDOW = NB_TX_DELAY_WINDOW - NB_TX_DELAY

                    RISK_WINDOW = NB_FRAUD_WINDOW / NB_TX_WINDOW

                    dfx[feature + '_NB_TX_' + str(i) + 'DAY_WINDOW'] = list(NB_TX_WINDOW)
                    dfx[feature + '_RISK_' + str(i) + 'DAY_WINDOW'] = list(RISK_WINDOW)

                dfx.index = dfx.TRANSACTION_ID

                # Replace NA values with 0 (all undefined risk scores where NB_TX_WINDOW is 0)
                dfx.fillna(0, inplace=True)
                return dfx

            new_data = new_data.groupby('TERMINAL_ID').apply(
                lambda x: get_count_risk_rolling_window(x, delay_period=7, windows_size_in_days=[1, 7, 30],
                                                        feature="TERMINAL_ID"))
            new_data = new_data.sort_values('TX_DATETIME').reset_index(drop=True)

            return new_data

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

