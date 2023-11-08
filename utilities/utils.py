import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import preprocessing

from sklearn.model_selection import learning_curve
from xgboost import XGBRegressor

def scale_data(dataset):
    """
    Scale the input dataset using Min-Max scaling.

    Parameters:
        dataset (pd.DataFrame): The dataset to be scaled.

    Returns:
        pd.DataFrame: Scaled dataset.
    """

    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(dataset.values)
    scaled_dataset = pd.DataFrame(scaled_array, columns=dataset.columns)
     
    return scaled_dataset

def split_features_target(dataset):
    """
    Split the input dataframe into features (X) and target (y).

    Parameters:
        dataset (pd.DataFrame): The input dataframe.

    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
    """
    X = np.array(dataset.iloc[:, :-1])  # All columns except the last one
    y = np.array(dataset.iloc[:, -1])   # The last column

    return X, y

def rescale_values(values, prediction_variable, dataset):
    """
    Rescale values based on the specified prediction_variable and dataset.

    Parameters:
        values (numpy.ndarray): Array to be rescaled.
        prediction_variable (str): The variable being predicted.
        dataset (pandas.DataFrame): The dataset containing the prediction variable.

    Returns:
        rescaled_values (numpy.ndarray): Rescaled values.
    """
    max_prediction_variable = np.max(dataset[prediction_variable])
    min_prediction_variable = np.min(dataset[prediction_variable])
    
    rescaled_values = min_prediction_variable + (max_prediction_variable - min_prediction_variable) * values
    
    return rescaled_values


def load_train_test_indices():
    """
    Load saved train and test indices.

    Returns:
        tuple: (train_indices, test_indices)
    """
    with open('train_indices', 'rb') as f:
        train_indices = pickle.load(f)
    with open('test_indices', 'rb') as f:
        test_indices = pickle.load(f)

    return train_indices, test_indices 


def add_random_noise(data, perc, features, input_selection,noise_mode):
    lower = (100 - perc) / 100
    upper = (100 + perc) / 100
    
    if noise_mode:
        if features == 'BP':
            selected_columns = ['brSBP','brDBP','heart_rate']

        if features == 'STI':
            if input_selection == 'M1' or input_selection == 'M3':
                selected_columns = ['PEP','ET','ted','tad','tes']

            if input_selection == 'M2':
                selected_columns = ['PEP','ET']

        r = np.random.uniform(lower, upper, size=(data[selected_columns].shape))
        data[selected_columns] = r * data[selected_columns].values
    
    return data

def select_columns_based_on_input(dataset, input_selection, prediction_variable):
    """
    Select columns from the dataset based on the input_selection.

    Parameters:
        dataset (pandas.DataFrame): The original dataset.
        input_selection (str): The selection mode (M1, M2, or M3).
        prediction_variable (str): The variable being predicted.

    Returns:
        pandas.DataFrame: Dataset with selected columns.
    """
    if input_selection == 'M1':
        return dataset[['brSBP', 'brDBP', 'heart_rate', 'PEP', 'ET', 'ted', 'tad', 'tes', prediction_variable]] 

    elif input_selection ==  'M2':
        return dataset[['brSBP', 'brDBP', 'heart_rate', 'PEP', 'ET', prediction_variable]] 

    elif input_selection == 'M3':
        return dataset[['brSBP', 'brDBP', 'heart_rate', 'PEP', 'ET', 'ted', 'tad', 'tes', 'stroke_volume', 'ejection_fraction', prediction_variable]]
    
    
def elastance_xgb_regressor(X_train,X_test,y_train,y_test,prediction_variable, input_selection):
    model = set_xgboost_params(prediction_variable, input_selection)
    y_pred = model.fit(X_train,y_train).predict(X_test)
    
    return model,y_pred


def print_results(y_test, y_pred, variable_unit):
    """
    Print various regression metrics and statistics.

    Parameters:
        y_test (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        None
    """
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    nrmse = 100 * rmse / (np.max(y_test) - np.min(y_test))

    print('Mean Absolute Error:', np.round(mae, 2), variable_unit)
    print('Mean Squared Error:', np.round(mse, 2), variable_unit)
    print('Root Mean Squared Error:', np.round(rmse, 2), variable_unit)
    print('Normalized Root Mean Squared Error:', np.round(nrmse, 2), '%\n')

    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
    print('Correlation:', round(r_value, 2))
    print('Slope:', round(slope, 2))
    print('Intercept:', round(intercept, 2), variable_unit)
    print('r_value:', round(r_value, 2))
    print('p_value:', round(p_value, 4))

    print('Distribution of the reference data:', round(np.mean(y_test), 1), '±', round(np.std(y_test), 1), variable_unit)
    print('Distribution of the predicted data:', round(np.mean(y_pred), 1), '±', round(np.std(y_pred), 1), variable_unit)
          

def plot_learning_curve(estimator, title, X, y, cv, train_sizes, ylim=None, scoring=None, n_jobs=None):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters:
        estimator (object): The learning algorithm to use.
        title (str): Title for the chart.
        X (array-like): Training vector.
        y (array-like): Target relative to X for classification.
        cv (int): Determines the cross-validation splitting strategy.
        train_sizes (array-like): Relative or absolute numbers of training examples.
        ylim (tuple): Defines minimum and maximum y-values plotted.
        scoring (str): Evaluation metric (e.g., 'neg_mean_squared_error').
        n_jobs (int): Number of jobs to run in parallel.

    Returns:
        plt.figure: Generated plot.
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha = 0.1,
                     color = 'red')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha = 0.1, color= 'g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='blue',
             label = "Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color= 'g',
             label = "Cross-validation score")

    plt.legend(loc = "best")
    return plt


def set_xgboost_params(prediction_variable, input_selection):
    """
    Set XGBoost parameters based on the prediction variable and input selection.

    Parameters:
        prediction_variable (str): The variable being predicted ('Ees' or 'dead_volume').
        input_selection (str): The input selection ('M1', 'M2', or 'M3').

    Returns:
        xgb_model (XGBRegressor): Configured XGBoost model.
    """
    if prediction_variable == 'Ees':
        if input_selection == 'M1':
            xgb_model = XGBRegressor(n_estimators=1750, learning_rate=0.05, max_depth=3)
        elif input_selection == 'M2':
            xgb_model = XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=3)
        elif input_selection == 'M3':
            xgb_model = XGBRegressor(n_estimators=1250, learning_rate=0.1, max_depth=3)
    elif prediction_variable == 'dead_volume':
        if input_selection == 'M1':
            xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=3)
        elif input_selection == 'M2':
            xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=3)
        elif input_selection == 'M3':
            xgb_model = XGBRegressor(n_estimators=1750, learning_rate=0.1, max_depth=3)
    else:
        raise ValueError("Invalid prediction variable. Choose either 'Ees' or 'dead_volume'.")
    
    return xgb_model



def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
                       model, param_grid, cv = 10, scoring_fit = 'neg_mean_squared_error',
                       do_probabilities = False):

    gs = GridSearchCV(estimator = model,
                      param_grid = param_grid, 
                      cv = cv, 
                      scoring = scoring_fit,
                      verbose = 2
                     )
    fitted_model = gs.fit(X_train_data, y_train_data)
    
    if do_probabilities:
        pred = fitted_model.predict_proba(X_test_data)
    else:
        pred = fitted_model.predict(X_test_data)
    
    return fitted_model, pred



def hyperparameter_tuning(X_train, X_test, y_train, y_test, regressor):
    if regressor == 'XGB':
        model = xgboost.XGBRegressor()
        param_grid = {
            'n_estimators': [500, 750, 1000, 1250, 1500, 1750],
            'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.15],
            'max_depth': [3, 5, 10]
        }

    model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model, param_grid, cv=10)

    # Root Mean Squared Error
    rmse = np.sqrt(-model.best_score_)
    print(rmse)
    
    best_parameters = model.best_params_

    return model, pred, best_parameters