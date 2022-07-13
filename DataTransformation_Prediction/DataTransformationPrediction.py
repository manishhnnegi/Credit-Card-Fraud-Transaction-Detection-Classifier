from datetime import datetime
from os import listdir
import pandas


class dataTransformPredict:
    """
                 This class shall be used for transforming the Good Raw Training Data before loading it in Database!!.

                 Written By: iNeuron Intelligence
                 Version: 1.0
                 Revisions: None

                 """

    def __init__(self):
        self.goodDataPath = "Prediction_Raw_Files_Validated/Good_Raw"

    def replaceMissingWithNull(self):

        """
                                Method Name: replaceMissingWithNull
                                Description: This method replaces the missing values in columns with "NULL" to
                                             store in the table. We are using substring in the first column to
                                             keep only "Integer" data for ease up the loading.
                                             This column is anyways going to be removed during prediction.


                                        """

        try:

            onlyfiles = [f for f in listdir(self.goodDataPath)]
            for file in onlyfiles:
                csv = pandas.read_csv(self.goodDataPath + "/" + file)
                csv.fillna('NULL', inplace=True)
                # #csv.update("'"+ csv['Wafer'] +"'")
                # csv.update(csv['Wafer'].astype(str))
                #csv['Wafer'] = csv['Wafer'].str[6:]
                csv.to_csv(self.goodDataPath + "/" + file, index=None, header=True)


        except Exception as e:

            raise e

