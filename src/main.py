import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, make_scorer, roc_curve, auc


def loadData():
    awards_players = pd.read_csv("../data/awards_players.csv")
    coaches = pd.read_csv("../data/coaches.csv")
    players_teams = pd.read_csv("../data/players_teams.csv")
    players = pd.read_csv("../data/players.csv")
    teams_post = pd.read_csv("../data/teams_post.csv")
    teams = pd.read_csv("../data/teams.csv")
    series_post = pd.read_csv("../data/series_post.csv")
    return awards_players, coaches, players_teams, players, teams_post, teams, series_post

def weighted_average_attributes(players_per_team_prev_years_original, players_per_team_year, current_year, columns, groupby_attribute):
    players_per_team_prev_years = players_per_team_prev_years_original.copy()
    k = 0.5
    players_per_team_prev_years['Weight'] = np.exp(-k * (current_year - players_per_team_prev_years['year']))
    for attribute in columns:
        players_per_team_prev_years[attribute] = players_per_team_prev_years[attribute] * players_per_team_prev_years['Weight']


    attribute_averages = players_per_team_prev_years.groupby(groupby_attribute)[columns].sum() 



    summed_weights = players_per_team_prev_years['Weight'].unique().sum()
    for attribute in columns:
        attribute_averages[attribute] = attribute_averages[attribute] / summed_weights
    if groupby_attribute == "playerID":
        attribute_averages['Total'] = attribute_averages.sum(axis=1)

    attribute_averages.reset_index(inplace=True)


    all_players_prev_years = attribute_averages[groupby_attribute].unique()


    current_year_players = players_per_team_year[groupby_attribute]

    for player in current_year_players:
        if player not in all_players_prev_years:
            new_row = pd.DataFrame(
                {col: 0 for col in attribute_averages.columns}, index=[0]
            )
            new_row[groupby_attribute] = player
            attribute_averages = pd.concat([attribute_averages, new_row])


    return attribute_averages



def players_per_team_averages(players_per_team, coaches, year, team_info, player_info, coach_info):
    """
    This function calculates the average of the players in each team for a given year, based on the past years.
    If year is 7, it will calculate the averages for the years 1 to 6.
    """
    players_per_team_year = players_per_team[players_per_team["year"] == year]
    players_per_team_year = players_per_team_year[players_per_team_year['stint'] < 2]


    coaches_year = coaches[coaches["year"] == year]
    coaches_year = coaches_year[coaches_year['stint'] < 2]

    players_per_team_prev_years = players_per_team[players_per_team["year"] < year].copy()
    coaches_prev_years = coaches[coaches["year"] < year].copy()

    players_per_team_prev_years = players_per_team_prev_years.drop(columns=['stint'])
    coaches_prev_years = coaches_prev_years.drop(columns=['stint'])
    players_per_team_year = players_per_team_year.drop(columns=['stint'])
    coaches_year = coaches_year.drop(columns=['stint'])


    player_averages = weighted_average_attributes(players_per_team_prev_years, players_per_team_year, year, player_info, "playerID")



    players_per_team_year = players_per_team_year[['playerID', 'year', 'tmID', 'playoff', 'winRatio']]





    players_per_team_year = pd.merge(players_per_team_year, player_averages, how="inner", on="playerID")

    players_per_team_year = players_per_team_year.loc[players_per_team_year.groupby("tmID")["Total"].idxmax()]

    coach_info_2 = ['won', 'lost', 'post_wins', 'post_losses']


    coach_averages = weighted_average_attributes(coaches_prev_years, coaches_year, year, coach_info_2, "coachID")


    
    coaches_year = coaches_year[['coachID', 'tmID']]
    coaches_year = pd.merge(coaches_year, coach_averages, how="inner", on="coachID")

    

    coaches_year['coachWinRatio'] = coaches_year['won'] / coaches_year['lost']
    coaches_year['coachPostWinRatio'] = coaches_year['post_wins'] / coaches_year['post_losses']
    coaches_year.replace([np.inf, -np.inf], np.nan, inplace=True)
    coaches_year['coachPostWinRatio'] = coaches_year['coachPostWinRatio'].fillna(coaches_year['post_wins'])
    coaches_year['coachWinRatio'] = coaches_year['coachWinRatio'].fillna(coaches_year['won'])

    columns = ['tmID', 'coachID']
    columns.extend(coach_info)
    coaches_year = coaches_year[columns]





    teams_prev_years = players_per_team_prev_years.drop_duplicates(subset=['tmID', 'year'])
    teams_current_year = players_per_team_year.drop_duplicates(subset=['tmID', 'year'])

    team_averages = weighted_average_attributes(teams_prev_years, teams_current_year, year, team_info, "tmID")



    players_per_team_year = pd.merge(players_per_team_year, team_averages, how="inner", on=["tmID"])



    players_per_team_year = pd.merge(players_per_team_year, coaches_year, how="inner", on=["tmID"])



    columns = ['tmID', 'year', 'playoff', 'winRatio']
    columns.extend(team_info)
    columns.extend(player_info)
    columns.extend(coach_info)
    players_per_team_year = players_per_team_year[columns]



    return players_per_team_year

def balance_data(data, label):
    most_common_value = data[label].value_counts().idxmax()
    minority_class = data[data[label] != most_common_value]
    majority_class = data[data[label] == most_common_value]



    minority_oversampled = resample(
        minority_class,
        replace=True,
        n_samples=len(majority_class),
        random_state=42
    )

    balanced_data = pd.concat([majority_class, minority_oversampled])

    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_data
    

def rename_attributes(data, player_info):
    columns = {}
    counter = 0
    for player_stat in player_info:
        columns[player_stat] = "BP" + player_stat
        player_info[counter] = "BP" + player_stat
        counter += 1
    # Rename the columns to Best Player (BP)
    return data.rename(columns=columns)
    


def getData():
    awards_players, coaches, players_teams, players, teams_post, teams, series_post = loadData()

    teams['winRatio'] = teams['won'] / (teams['won'] + teams['lost'])
    teams['playoff'] = 'N'
    for year in range(1, 12):
        teams_in_year = teams[teams['year'] == year]
        if not teams_in_year.empty:
            top_8_teams = teams_in_year.nlargest(8, 'winRatio').index
            teams.loc[top_8_teams, 'playoff'] = 'Y'


    team_info = ['rank', 'o_3pa', 'o_asts', 'o_pf', 'o_stl', 'o_to', 'o_blk', 'o_pts', 'd_fgm', 'd_ftm', 'd_fta', 'd_3pm', 'd_3pa', 'd_oreb', 'd_dreb', 'd_reb', 'd_asts', 'd_pf', 'd_stl', 'd_to', 'd_blk', 'd_pts', 'won', 'lost', 'confW', 'confL', 'min', 'attend']



    player_info = ['assists', 'steals', 'rebounds']

    coach_info = ['coachWinRatio', 'coachPostWinRatio']

    columns = ['tmID', 'year', 'playoff', 'winRatio']
    columns.extend(team_info)
    columns.extend(player_info)
    columns.extend(coach_info)

    data = pd.DataFrame(columns=columns)


    columns = ['tmID', 'year', 'playoff', 'winRatio']
    columns.extend(team_info)
    teams = teams[columns]
    players_team_columns = ['playerID', 'year', 'tmID', 'stint']
    players_team_columns.extend(player_info)
    players_teams = players_teams[players_team_columns]
    players_per_team = pd.merge(teams, players_teams, how="inner", left_on=["year", "tmID"], right_on=["year", "tmID"])

    # Calculate the best player stats for each team for each year
    for year in range(2, 11):
        data = pd.concat([data, players_per_team_averages(players_per_team, coaches, year, team_info, player_info, coach_info)])


    data = rename_attributes(data, player_info)

    
    data.to_csv("data.csv", index=False)

    total_ints = team_info.copy()

    total_ints.extend(player_info)
    total_ints.extend(coach_info)

    return data, total_ints


def preprocess(data, total_ints, current_year, label, balanceData = False):
    train_data = data[data['year'] < current_year]
    if label == "playoff":
        train_data[label] = train_data[label].map({'Y': 1, 'N': 0})
    if balanceData:
        train_data = balance_data(train_data, label)

    train_data_labels = train_data[label]
    

    test_data = data[data['year'] == current_year]
    if label == "playoff":
        test_data[label] = test_data[label].map({'Y': 1, 'N': 0})
    test_data_labels = test_data[label]
    
    train_data = train_data.drop(columns=['winRatio', 'year', 'tmID', 'playoff'])
    test_data_tmID = test_data['tmID']
    test_data_playoff = test_data['playoff']

    test_data = test_data.drop(columns=['year', 'winRatio', 'tmID', 'playoff'])
    


    scaler = StandardScaler()
    scaler.fit(train_data[total_ints])


    train_data.loc[:, total_ints] = scaler.transform(train_data[total_ints])
    test_data.loc[:, total_ints] = scaler.transform(test_data[total_ints])

    return train_data, train_data_labels, test_data, test_data_labels, test_data_tmID, test_data_playoff

def calculate_error(probabilities, test_data_labels):
    error = []
    for [_, win_prob] in probabilities:
        error.append(win_prob*100.0)

    error_sum = test_data_labels[test_data_labels == 1].count()
    for index in range(0, len(error)):
        error[index] = abs((error[index] * (error_sum/100.0)/error_sum) - test_data_labels[index])

    error = sum(error)
    return error

def train_model(model, x, y):
    model.fit(x, y)

def train_predict_regression(model, x_train, y_train, x_test, y_test):
    def predict_labels(model, features):
        predictions = model.predict(features)
        predictions = np.clip(predictions, 0, 1)
        return predictions
    

    train_model(model, x_train, y_train)

    train_predictions = predict_labels(model, x_train)


    predictions = predict_labels(model, x_test)

    return predictions, train_predictions

def train_predict_classifier(model, x_train, y_train, x_test, y_test):
    def predict_labels(model, features):
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        return predictions, probabilities

    train_model(model, x_train, y_train)

    train_predictions, train_probabilities, = predict_labels(model, x_train)

    predictions, probabilities = predict_labels(model, x_test)

    return predictions, probabilities, train_predictions, train_probabilities

def evaluate_model(metric, labels, prediction, probability = None):
    if metric == "f1":
        return f1_score(labels, prediction, pos_label=1) * 100.0
    elif metric == "acc":
        return (sum(labels == prediction) / len(prediction)) * 100.0
    elif metric == "auc":
        fpr, tpr, _ = roc_curve(labels, probability[:, 1])
        return auc(fpr, tpr) * 100.0
    elif metric == "error":
        return calculate_error(probability, labels)
    elif metric == "mae":
        return mean_absolute_error(labels, prediction)
    elif metric == "mse":
        return mean_squared_error(labels, prediction)
    elif metric == "r2":
        return r2_score(labels, prediction) * 100.0


def forward_selection(model, original_train_data, original_test_data, train_data_labels, test_data_labels, problem_type, metric):
    rfe_model = None
    if problem_type == "Classification":
        rfe_model = LogisticRegression(random_state=42)
    elif problem_type == "Regression":
        rfe_model = LinearRegression()
    best_stat = 0
    if metric == "error" or metric == "mae":
        best_stat = 100
    best_columns = None
    metric_per_features = []
    for rank_num in range(1, len(original_train_data.columns)):
        train_data = original_train_data.copy()
        test_data = original_test_data.copy()
        rfe = RFE(estimator=rfe_model, n_features_to_select=1)
        fit = rfe.fit(train_data, train_data_labels)
        new_columns = []
        index_array = []
        for index in range(0, len(fit.ranking_)):
            if fit.ranking_[index] <= rank_num:
                index_array.append(index)
        for index in range(0, len(train_data.columns)):
            if(index in index_array):
                new_columns.append(train_data.columns[index])
        train_data = train_data[new_columns]
        test_data = test_data[new_columns]
        stat = None
        if problem_type == "Classification":
            predictions, probabilities, _, _ = train_predict_classifier(model, train_data, train_data_labels, test_data, test_data_labels)
            stat = evaluate_model(metric, test_data_labels, predictions, probabilities)
        elif problem_type == "Regression":
            predictions, _ = train_predict_regression(model, train_data, train_data_labels, test_data, test_data_labels)
            stat = evaluate_model(metric, test_data_labels, predictions)
        
        metric_per_features.append((stat, len(new_columns)))
        if (metric == "error" or metric == "mae") and stat < best_stat:
            best_stat = stat
            best_columns = new_columns
        elif metric != "error" and metric != "mae" and stat > best_stat:
            best_stat = stat
            best_columns = new_columns

    return original_train_data[best_columns], original_test_data[best_columns], metric_per_features

def getModels(problem_type):
    if problem_type == "Regression":
        return [LinearRegression(), DecisionTreeRegressor(random_state=42), RandomForestRegressor(random_state=42), SVR(), xgb.XGBRegressor(), MLPRegressor(random_state=42)]
    elif problem_type == "Classification":
        return [LogisticRegression(random_state=42), SVC(random_state=912, kernel='linear', probability=True), xgb.XGBClassifier(seed=82), MLPClassifier(random_state=42)]

def getBestModel(models, train_data, train_data_labels, test_data, test_data_labels, problem_type, metric = "error"):
    best_model = None

    best_stat = 0

    if metric == "mae" or metric == "error":
        best_stat = 100


    for model in models:
        stat = None
        if problem_type == "Classification":
            predictions, probabilities, _, _ = train_predict_classifier(model, train_data, train_data_labels, test_data, test_data_labels)
            stat = evaluate_model(metric, test_data_labels, predictions, probabilities)
        elif problem_type == "Regression":
            predictions, _ = train_predict_regression(model, train_data, train_data_labels, test_data, test_data_labels)
            stat = evaluate_model(metric, test_data_labels, predictions)

        if (metric == "mae" or metric == "error") and stat < best_stat:
            best_stat = stat
            best_model = model
        elif metric != "mae" and metric != "error" and stat > best_stat:
            best_stat = stat
            best_model = model

    return best_model

def predict_win_ratio(model, train_data, train_data_labels, test_data, test_data_labels, test_data_tmID, test_data_playoff, metrics):


    def fine_tune_model(model, train_data, train_data_labels):
        match(model.__class__.__name__):
            case "RandomForestRegressor":
                return RandomForestRegressor(random_state=42)
            case "SVR":
                #return SVR(C=0.1, gamma='auto', kernel='poly')
                return SVR(C=0.1, kernel="linear")
            case "MLPClassifier":
                return 
            

        parameter_rf = {
        'n_estimators': [50, 100, 200, 300],      # Number of trees in the forest
        'max_depth': [None, 10, 20, 30, 50],      # Maximum depth of each tree
        'min_samples_split': [2, 5, 10],          # Minimum number of samples to split an internal node
        'min_samples_leaf': [1, 2, 4],            # Minimum number of samples required to be at a leaf node
        'max_features': ['auto', 'sqrt', 'log2'], # Number of features to consider when looking for the best split
        'bootstrap': [True, False],               # Whether to use bootstrapping when building trees
        'random_state': [42],                     # Fixed seed for reproducibility
        }

        parameter_svr = {
            'C': [0.1, 1, 10, 100, 1000],              # Regularization parameter
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Type of kernel
            'epsilon': [0.01, 0.1, 0.5, 1, 1.5],        # Epsilon
            'gamma': ['scale', 'auto', 0.1, 1, 10],     # Kernel coefficient (relevant for 'rbf', 'poly', and 'sigmoid')
            'degree': [3, 4, 5]                        # Degree of the polynomial kernel (only for 'poly' kernel)
        }

        parameter_dict = {"RandomForestRegressor": parameter_rf, "SVR": parameter_svr}

        if parameter_dict[model.__class__.__name__] == None:
            print("parameters for " + model.__class__.__name__ + "do not exist")
            exit(1)

        best_model_params = parameter_dict[model.__class__.__name__]



        scorer = make_scorer(mean_absolute_error, greater_is_better=False)

        grid_obj = GridSearchCV(
            estimator=model,
            scoring=scorer,
            param_grid=best_model_params,
            cv=5,
            n_jobs=-1
        )
        
        
        grid_obj = grid_obj.fit(train_data, train_data_labels)

        model = grid_obj.best_estimator_

        return model

    
    
    #best_model = fine_tune_model(best_model, train_data, train_data_labels)


    predictions, _ = train_predict_regression(model, train_data, train_data_labels, test_data, test_data_labels)

    
    test_data['tmID'] = test_data_tmID
    test_data['winRatio'] = test_data_labels
    test_data['predictedWinRatio'] = predictions
    test_data['playoff'] = test_data_playoff

    test_data['predictedPlayoff'] = 'N'
    top_8_teams = test_data.nlargest(8, 'predictedWinRatio').index
    test_data.loc[top_8_teams, 'predictedPlayoff'] = 'Y'

    test_data = test_data[['tmID', 'winRatio', 'predictedWinRatio', 'playoff', 'predictedPlayoff']]
    print("The " + model.__class__.__name__ + " Predicted:")
    print(test_data)
    for metric in metrics:
        labels = test_data_labels
        pred = predictions
        if metric == "acc" or metric == "f1":
            labels = test_data_playoff.map({"Y": 1, "N": 0})
            pred = test_data['predictedPlayoff'].map({"Y": 1, "N": 0})
        stat = str(evaluate_model(metric, labels, pred))


        print("The " + metric + " is: " + stat)



def predict_playoff(model, train_data, train_data_labels, test_data, test_data_labels, test_data_tmID, metrics):

    
    predictions, probabilities, _, _, = train_predict_classifier(model, train_data, train_data_labels, test_data, test_data_labels)

    probabilities_per_team = []
    final_playoff = []
    for [_, win_prob] in probabilities:
        probabilities_per_team.append(win_prob)
        final_playoff.append('N')

    test_data['tmID'] = test_data_tmID
    test_data['playoff'] = test_data_labels.map({1: "Y", 0: "N"})
    test_data['predictedPlayoff'] = predictions
    test_data['predictedPlayoff'] = test_data['predictedPlayoff'].map({1: "Y", 0: "N"})
    test_data['finalPlayoff'] = final_playoff
    test_data['probabilities'] = probabilities_per_team

    top_8_teams = test_data.nlargest(8, 'probabilities').index
    test_data.loc[top_8_teams, 'finalPlayoff'] = 'Y'

    #test_data = test_data.drop(columns=["probabilities"])
    test_data = test_data[['tmID', 'playoff', 'predictedPlayoff', 'finalPlayoff', 'probabilities']]

    print("The " + model.__class__.__name__ + " Predicted:")
    print(test_data)

    for metric in metrics:
        pred = predictions
        if metric == "acc" or metric == "f1":
            pred = test_data['finalPlayoff'].map({"Y": 1, "N": 0})
        stat = str(evaluate_model(metric, test_data_labels, pred, probabilities))


        print("The " + metric + " is: " + stat)

def main():
    pd.set_option('display.max_rows', None)


    data, total_ints = getData()

    problem_type = "Classification"


    if problem_type == "Regression":
        train_data, train_data_labels, test_data, test_data_labels, test_data_tmID, test_data_playoff = preprocess(data, total_ints, 10, "winRatio", False)
        models = getModels(problem_type)

        best_model = getBestModel(models, train_data, train_data_labels, test_data, test_data_labels, problem_type, "r2")

        train_data, test_data, metric_per_feature = forward_selection(best_model, train_data, test_data, train_data_labels, test_data_labels, problem_type, "mae")

        predict_win_ratio(best_model, train_data, train_data_labels, test_data, test_data_labels, test_data_tmID, test_data_playoff, ["mae", "f1"])
    elif problem_type == "Classification":
        train_data, train_data_labels, test_data, test_data_labels, test_data_tmID, test_data_playoff = preprocess(data, total_ints, 10, "playoff", True)

        models = getModels(problem_type)


        best_model = getBestModel(models, train_data, train_data_labels, test_data, test_data_labels, problem_type, "error")

        train_data, test_data, metric_per_feature = forward_selection(best_model, train_data, test_data, train_data_labels, test_data_labels, problem_type, "error")

        predict_playoff(best_model, train_data, train_data_labels, test_data, test_data_labels, test_data_tmID, ["error", "f1", "acc", "auc"])





if __name__ == "__main__":
    main()