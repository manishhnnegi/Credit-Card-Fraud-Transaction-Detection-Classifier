from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
import sklearn
from sklearn.pipeline import Pipeline
import datetime
import pandas as pd
import xgboost
import numpy as np

class Model_Finder:


    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object


    def get_best_params_for_random_forest(self, train_x, train_y,x,y ,data,start_date_training,n_folds,delta_train,delta_delay,delta_assessment):
        """
                                        Method Name: get_best_params_for_random_forest
                                        Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception


                                """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:




            #param_grid = {'clf__max_depth':[10,20], 'clf__n_estimators':[100,500],
                         #'clf__random_state':[0],'clf__class_weight':[{0: w} for w in [ 0.1, 0.5, 1]]
                          #}
            param_grid = {'clf__max_depth': [2, 3], 'clf__n_estimators': [5],
                          'clf__random_state': [0], 'clf__class_weight': [{0: w} for w in [0.1, 0.5]]
                          }

            scoring = {'roc_auc': 'roc_auc',
                       'average_precision': 'average_precision',
                       }

            prequential_split_indices = self.prequentialSplit(data, start_date_training, n_folds=4, delta_train=7,
                                                              delta_delay=7, delta_assessment=7)
            estimators = [('scaler', sklearn.preprocessing.StandardScaler()), ('clf', RandomForestClassifier())]
            pipe = sklearn.pipeline.Pipeline(estimators)

            grid_search = sklearn.model_selection.GridSearchCV(pipe, param_grid=param_grid, scoring=scoring,
                                                                cv=prequential_split_indices, refit=False,verbose=0)

            # finding the best parameters
            grid_search.fit(x, y)

            df = pd.DataFrame(grid_search.cv_results_)

            index_rf = df.index[np.argmax(df['mean_test_average_precision'].values)]
            best_parameters_rf = grid_search.cv_results_['params'][index_rf]


            max_depth = best_parameters_rf['clf__max_depth']
            n_estimators = best_parameters_rf['clf__n_estimators']
            random_state = best_parameters_rf['clf__random_state']
            class_weight = best_parameters_rf['clf__class_weight']

            rf_clf = RandomForestClassifier(max_depth=30, n_estimators=1000, random_state=random_state,
                                   class_weight={0:1},criterion = "entropy",min_samples_split=10, min_samples_leaf=1)

            # training the mew model
            rf_clf.fit(train_x, train_y)

            self.logger_object.log(self.file_object,
                                   'Random Forest best params: ' + str(best_parameters_rf) + '. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return rf_clf

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()


    def get_best_params_for_xgboost(self, train_x, train_y,x,y, data,start_date_training,n_folds,delta_train,delta_delay,delta_assessment):
        """
                                                Method Name: get_best_params_for_xgboost
                                                Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                             Use Hyper Parameter Tuning.
                                                Output: The model with the best parameters
                                                On Failure: Raise Exception



                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:







            #param_grid_xgboost = {
                                     # 'clf__max_depth':[3,6,9],
                                      #'clf__n_estimators':[25,50,100],
                                     # 'clf__learning_rate':[0.1, 0.3],
                                      #'clf__random_state':[0],

                                # }
            param_grid_xgboost = {
                'clf__max_depth': [2,3],
                'clf__n_estimators': [2],
                'clf__learning_rate': [0.1, 0.3],
                'clf__random_state': [0],

            }
            prequential_split_indices = self.prequentialSplit(data,start_date_training, n_folds=4, delta_train=7,
                                                              delta_delay=7, delta_assessment=7)

            scoring = {'roc_auc': 'roc_auc',
                       'average_precision': 'average_precision',
                       }


            estimators = [('scaler', sklearn.preprocessing.StandardScaler()), ('clf', xgboost.XGBClassifier())]
            pipe = sklearn.pipeline.Pipeline(estimators)

            grid_search = sklearn.model_selection.GridSearchCV(pipe, param_grid=param_grid_xgboost, scoring=scoring,
                                                               cv=prequential_split_indices, refit=False)
            # finding the best parameters
            grid_search.fit(x, y)

            df = pd.DataFrame(grid_search.cv_results_)

            index_xg = df.index[np.argmax(df['mean_test_average_precision'].values)]
            best_parameters_xg = grid_search.cv_results_['params'][index_xg]


            learning_rate = best_parameters_xg['clf__learning_rate']
            max_depth = best_parameters_xg['clf__max_depth']
            n_estimators = best_parameters_xg['clf__n_estimators']
            random_state = best_parameters_xg['clf__random_state']

            xgb = xgboost.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.05,
                                  random_state=random_state, min_child_weight=1, colsample_bytree=0.9, reg_alpha=1, objective='binary:logistic')

            # creating a new model with the best parameters

            # training the mew model
            xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(best_parameters_xg) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')

            return xgb

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()

    def prequentialSplit(self, transactions_df,
                         start_date_training,
                         n_folds=4,
                         delta_train=7,
                         delta_delay=7,
                         delta_assessment=7):
        """
                                                Method Name: prequentialSplit
                                                Description: prequential grid search which give the best accuracy.
                                                             Use Hyper Parameter Tuning.
                                                Output: The model with the best parameters
                                                On Failure: Raise Exception

                                        """
        #self.logger_object.log(self.file_object,
                               #'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            delta_valid=7
            start_date_training_for_valid = start_date_training + datetime.timedelta(days=-(delta_delay + delta_valid))
            prequential_split_indices = []

            # For each fold
            for fold in range(n_folds):
                # Shift back start date for training by the fold index times the assessment period (delta_assessment)
                # (See Fig. 5)
                start_date_training_fold = start_date_training_for_valid - datetime.timedelta(days=fold * delta_assessment)

                # Get the training and test (assessment) sets
                (train_df, test_df) = self.get_train_test_set(transactions_df,
                                                         start_date_training=start_date_training_fold,
                                                         delta_train=delta_train, delta_delay=delta_delay,
                                                         delta_test=delta_assessment)

                # Get the indices from the two sets, and add them to the list of prequential splits
                indices_train = list(train_df.index)
                indices_test = list(test_df.index)

                prequential_split_indices.append((indices_train, indices_test))

            return prequential_split_indices

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()




    def get_train_test_set(self, transactions_df,
                           start_date_training,
                           delta_train=7, delta_delay=7, delta_test=7, ):
        """
                                        Method Name: get_train_test_set
                                        Description: traintest split
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception


                                """
        self.logger_object.log(self.file_object,
                               'Entered the train test split method of the Model_Finder class')
        try:
            # Get the training set data
            train_df = transactions_df[(transactions_df.TX_DATETIME >= start_date_training) &
                                       (transactions_df.TX_DATETIME < start_date_training + datetime.timedelta(
                                           days=delta_train))]

            # Get the test set data
            test_df = []

            # Note: Cards known to be compromised after the delay period are removed from the test set
            # That is, for each test day, all frauds known at (test_day-delay_period) are removed

            # First, get known defrauded customers from the training set
            known_defrauded_customers = set(train_df[train_df.TX_FRAUD == 1].CUSTOMER_ID)

            # Get the relative starting day of training set (easier than TX_DATETIME to collect test data)
            start_tx_time_days_training = train_df.TX_TIME_DAYS.min()

            # Then, for each day of the test set
            for day in range(delta_test):
                # Get test data for that day
                test_df_day = transactions_df[transactions_df.TX_TIME_DAYS == start_tx_time_days_training +
                                              delta_train + delta_delay +
                                              day]

                # Compromised cards from that test day, minus the delay period, are added to the pool of known defrauded customers
                test_df_day_delay_period = transactions_df[transactions_df.TX_TIME_DAYS == start_tx_time_days_training +
                                                           delta_train +
                                                           day - 1]

                new_defrauded_customers = set(
                    test_df_day_delay_period[test_df_day_delay_period.TX_FRAUD == 1].CUSTOMER_ID)
                known_defrauded_customers = known_defrauded_customers.union(new_defrauded_customers)

                test_df_day = test_df_day[~test_df_day.CUSTOMER_ID.isin(known_defrauded_customers)]

                test_df.append(test_df_day)

            test_df = pd.concat(test_df)

            # Sort data sets by ascending order of transaction ID
            train_df = train_df.sort_values('TRANSACTION_ID')
            test_df = test_df.sort_values('TRANSACTION_ID')

            return (train_df, test_df)


        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception traing test split fail class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'traing test split fail  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()

    def split_train_test(self, train_df, test_df,):
        """
                                        Method Name: split_train_test
                                        Description: traintest split
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception



                                """
        self.logger_object.log(self.file_object,
                               'Entered the train test split method of the Model_Finder class')
        try:

            output_feature = "TX_FRAUD"

            input_features = ['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                              'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                              'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                              'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                              'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                              'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                              'TERMINAL_ID_RISK_30DAY_WINDOW']

            X_tr = train_df[input_features]
            y_tr = train_df[output_feature]
            X_ts = test_df[input_features]
            y_ts = test_df[output_feature]

            return X_tr, y_tr, X_ts, y_ts

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()

    def get_best_model(self, train_x, train_y, test_x, test_y,x,y, data,start_date_training,n_folds,delta_train,delta_delay,delta_assessment):
        """
                                                        Method Name: get_best_model
                                                        Description: Find out the Model which has the best AUC score.
                                                        Output: The best model name and the model object
                                                        On Failure: Raise Exception

                                                """

        # create best model for XGBoost
        try:
            # create best model for XGBoost
            #self.logger_object.log(self.file_object,'Entered the get_best_model method of the Model_Finder class')
            xgboost = self.get_best_params_for_xgboost( train_x, train_y,x,y ,data,start_date_training,n_folds,delta_train,delta_delay,delta_assessment)
            prediction_xgboost = xgboost.predict(test_x)  # Predictions using the XGBoost Model

            xgboost_prob = xgboost.predict_proba(test_x)

            try:
                xgboost_scores = xgboost_prob[:, 1]
            except:
                xgboost_scores = xgboost.decision_function(test_x)

            xgboost_score = average_precision_score(test_y, xgboost_scores)
            #auc = roc_auc_score(y, y_scores)
            #AP = metrics.average_precision_score(y, y_scores)



            # create best model for Random Forest
            random_forest = self.get_best_params_for_random_forest(train_x, train_y,x,y, data,start_date_training,n_folds,delta_train,delta_delay,delta_assessment)
            prediction_random_forest = random_forest.predict(test_x)  # prediction using the Random Forest Algorithm
            prediction_xgboost = xgboost.predict(test_x)  # Predictions using the XGBoost Model

            rf_prob = random_forest.predict_proba(test_x)

            try:
                rf_scores = rf_prob[:, 1]
            except:
                rf_scores = random_forest.decision_function(test_x)

            random_forest_score = average_precision_score(test_y, rf_scores)

            # comparing the two models
            if (random_forest_score < xgboost_score):
                return 'XGBoost', xgboost
            else:
                return 'RandomForest', random_forest

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()




