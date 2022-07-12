import shutil
import sqlite3
from datetime import datetime
from os import listdir
import os
import csv
import pandas as pd


# from application_logging.logger import App_Logger


class dBOperation:
    """
      This class shall be used for handling all the SQL operations.

      Written By: iNeuron Intelligence
      Version: 1.0
      Revisions: None

      """

    def __init__(self):

        self.path = 'Prediction_Database/'
        self.badFilePath = "Prediction_Raw_Files_Validated/Bad_Raw"
        self.goodFilePath = "Prediction_Raw_Files_Validated/Good_Raw"

    def dataBaseConnection(self, DatabaseName):

        """
                Method Name: dataBaseConnection
                Description: This method creates the database with the given name and if Database already exists then opens the connection to the DB.
                Output: Connection to the DB
                On Failure: Raise ConnectionError

                 Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """
        try:
            conn = sqlite3.connect(self.path + DatabaseName + '.db')


        except ConnectionError:

            raise ConnectionError
        return conn

    def insertIntoTableGoodData(self, Database, column_names):

        """
                               Method Name: insertIntoTableGoodData
                               Description: This method inserts the Good data files from the Good_Raw folder into the
                                            above created table.
                               Output: None
                               On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                               Version: 1.0
                               Revisions: None

        """

        conn = self.dataBaseConnection(Database)
        goodFilePath = self.goodFilePath
        badFilePath = self.badFilePath
        onlyfiles = [f for f in listdir(goodFilePath)]

        try:
            for file in onlyfiles:
                with open(goodFilePath + '/' + file, "r") as f:
                    df = pd.read_csv(f)
                    df['TX_DATETIME'] = df['TX_DATETIME'].astype('<M8[ns]')
                    if df.dtypes.to_dict() == column_names:
                        df.to_sql('Good_Raw_Data', conn, if_exists='append', index=False)
                        conn.commit()


        except Exception as e:
            raise e
            conn.rollback()
            conn.close()

        conn.close()

    def selectingDatafromtableintocsv(self, Database):

        """
                               Method Name: selectingDatafromtableintocsv
                               Description: This method exports the data in GoodData table as a CSV file. in a given location.
                                            above created .
                               Output: None
                               On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                               Version: 1.0
                               Revisions: None

        """
        fileFromDb = 'Prediction_FileFromDB/'
        fileName = 'InputFile.csv'

        try:

            # Make the CSV ouput directory
            if not os.path.isdir(fileFromDb):
                os.makedirs(fileFromDb)

            conn = self.dataBaseConnection(Database)
            c = conn.cursor()
            df = pd.read_sql_query("select * from Good_Raw_Data ", conn)

            df.to_csv(fileFromDb + fileName, index=None, header=True)
            c.execute('DELETE FROM Good_Raw_Data')


            conn.close()


        except Exception as e:
            raise e





