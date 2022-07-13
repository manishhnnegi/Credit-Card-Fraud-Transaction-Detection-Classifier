import pickle
import os
import shutil


class File_Operation:
    """
                    This class shall be used to save the model after training
                    and load the saved model for prediction.


                    """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.model_directory = 'models/'

    def save_model(self, model, filename):
        """
                    Method Name: save_model
                    Description: Save the model file to directory
                    Outcome: File gets saved
                    On Failure: Raise Exception


        """
        self.logger_object.log(self.file_object, 'Entered the save_model method of the File_Operation class')
        try:
            path = os.path.join(self.model_directory, filename)  # create seperate directory for each cluster

            if os.path.isdir(path):  # remove previously existing models for each clusters
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)  #
            file = path + '/' + filename + '.sav'
            pickle.dump(model, open(file, "wb"))  # save the model to file
            self.logger_object.log(self.file_object,
                                   'Model File ' + filename + ' saved. Exited the save_model method of the Model_Finder class')
            return 'success'
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in save_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Model File '+filename+' could not be saved. Exited the save_model method of the Model_Finder class')
            raise Exception()


    def load_model(self, filename):
        """
                            Method Name: load_model
                            Description: load the model file to memory
                            Output: The Model file loaded in memory
                            On Failure: Raise Exception


                """
        self.logger_object.log(self.file_object, 'Entered the load_model method of the File_Operation class')
        try:
            path = os.path.join(self.model_directory, filename)
            file = path + '/' + filename + '.sav'
            self.logger_object.log(self.file_object,
                                   'Model File ' + filename + ' loaded. Exited the load_model method of the Model_Finder class')

            return pickle.load(open(file, "rb"))
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in load_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model File ' + filename + ' could not be saved. Exited the load_model method of the Model_Finder class')
            raise Exception()


    def find_correct_model_file(self):
        """
                                    Method Name: find_correct_model_file
                                    Description: Select the correct model based on cluster number
                                    Output: The Model file
                                    On Failure: Raise Exception

                        """
        self.logger_object.log(self.file_object,
                               'Entered the find_correct_model_file method of the File_Operation class')
        try:
            folder_name = self.model_directory
            list_of_files = os.listdir(folder_name)
            for file in list_of_files:

                    model_name = file
                    self.logger_object.log(self.file_object,
                                           'Exited the find_correct_model_file method of the Model_Finder class.')
                    return model_name

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in find_correct_model_file method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Exited the find_correct_model_file method of the Model_Finder class with Failure')
            raise Exception()






