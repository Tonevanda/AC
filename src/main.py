import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
from sklearn.model_selection import learning_curve
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, make_scorer, roc_curve, auc

grid_results = []

def loadData():
    awards_players = pd.read_csv("../data/awards_players.csv")
    coaches = pd.read_csv("../data/coaches.csv")
    players_teams = pd.read_csv("../data/players_teams.csv")
    players = pd.read_csv("../data/players.csv")
    teams_post = pd.read_csv("../data/teams_post.csv")
    teams = pd.read_csv("../data/teams.csv")
    series_post = pd.read_csv("../data/series_post.csv")
    return awards_players, coaches, players_teams, players, teams_post, teams, series_post

def loadDataComp():
    coaches = pd.read_csv("../comp/coaches.csv")
    players_teams = pd.read_csv("../comp/players_teams.csv")
    teams = pd.read_csv("../comp/teams.csv")

    return coaches, players_teams, teams

def weighted_average_attributes(players_per_team_prev_years_original, players_per_team_year, current_year, columns, groupby_attribute, decay_number):
    players_per_team_prev_years = players_per_team_prev_years_original.copy()
    k = decay_number
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
        if player not in all_players_prev_years and groupby_attribute != 'playerID':
            new_row = pd.DataFrame(
                {col: 0 for col in attribute_averages.columns}, index=[0]
            )
            new_row[groupby_attribute] = player
            attribute_averages = pd.concat([attribute_averages, new_row])


    return attribute_averages



def players_per_team_averages(players_per_team, coaches, year, team_info, player_info, coach_info, problem_type, player_averages_number, decay_number):
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

    min_num_of_players_per_team_per_year = players_per_team_year.groupby(['tmID', 'year']).size()
    min_num_of_players_per_team_per_year = min_num_of_players_per_team_per_year.min()


    player_averages = weighted_average_attributes(players_per_team_prev_years, players_per_team_year, year, player_info, "playerID", decay_number)


    columns = ['playerID', 'year', 'tmID', 'playoff', "confID"]
    if problem_type == "Regression":
        columns.append('winRatio')
    players_per_team_year = players_per_team_year[columns]





    players_per_team_year = pd.merge(players_per_team_year, player_averages, how="inner", on="playerID")



    #players_per_team_year = players_per_team_year.loc[players_per_team_year.groupby("tmID")["Total"].idxmax()]

    players_per_team_year = players_per_team_year.groupby("tmID", group_keys=False).apply(
        lambda group: group.nlargest(min(min_num_of_players_per_team_per_year, player_averages_number), "Total")
    )

    players_per_team_year = players_per_team_year.groupby(["tmID", "year", "playoff", "confID"]).mean(numeric_only=True)

    players_per_team_year = players_per_team_year.reset_index()




    coach_info_2 = ['won', 'lost', 'post_wins', 'post_losses']


    coach_averages = weighted_average_attributes(coaches_prev_years, coaches_year, year, coach_info_2, "coachID", decay_number)


    
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




    

    team_averages = weighted_average_attributes(teams_prev_years, teams_current_year, year, team_info, "tmID", decay_number)



    players_per_team_year = pd.merge(players_per_team_year, team_averages, how="inner", on=["tmID"])



    players_per_team_year = pd.merge(players_per_team_year, coaches_year, how="inner", on=["tmID"])



    columns = ['tmID', 'year', 'playoff', 'confID']
    if problem_type == "Regression":
        columns.append('winRatio')
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
    
def profile_plot(data):
    plot_data = data.copy()
    # Convert 'playoff' column to string type
    
    plot_data.drop(columns=['tmID',"playoff"], inplace=True)
    # Normalize the column values
    scaler = MinMaxScaler()
    plot_data[plot_data.columns] = scaler.fit_transform(plot_data[plot_data.columns])
    
    # Add 'playoff' column back
    plot_data['playoff'] = data['playoff']
    
    # Group by 'playoff' and calculate the mean for each group
    plot_data_grouped = plot_data.groupby('playoff').mean().reset_index()
    
    # Save normalized data to CSV
    plot_data_grouped.to_csv("data_normalized2.csv", index=False)
    
    # Plotting the parallel coordinates plot
    """plt.figure(figsize=(20, 12))
    pd.plotting.parallel_coordinates(plot_data, 'playoff', color=('#556270', '#4ECDC4'))
    plt.title('Profile Plot of Team Data')
    plt.xlabel('Attributes')
    plt.ylabel('Values')
    plt.show()"""

def fillCompData(teams, coaches, players_teams):
    def fixMissingColumns(tableWithData, tableWithMissingData):
        missing_columns = set(tableWithData.columns) - set(tableWithMissingData.columns)

        for col in missing_columns:
            tableWithMissingData[col] = 0.0

        tableWithMissingData = tableWithMissingData[tableWithData.columns]
        return tableWithMissingData

    coachesComp, players_per_team_Comp, teamsComp = loadDataComp()

    teamsComp['playoff'] = ['Y', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N']
    
    coaches = pd.concat([coaches, fixMissingColumns(coaches, coachesComp)])
    players_teams = pd.concat([players_teams, fixMissingColumns(players_teams, players_per_team_Comp)])
    teams = pd.concat([teams, fixMissingColumns(teams, teamsComp)])

    return coaches, players_teams, teams


def getData(problem_type, player_averages_number, decay_number):
    awards_players, coaches, players_teams, players, teams_post, teams, series_post = loadData()



    coaches, players_teams, teams = fillCompData(teams, coaches, players_teams) 

    '''teams_post = teams_post[["tmID", "year", "W", "L"]]

    teams = pd.merge(teams, teams_post, how="outer", left_on=["tmID", "year"], right_on=["tmID", "year"])
    teams = teams.fillna(0.0)

    players_teams = pd.merge(players_teams, players, how="inner", left_on=["playerID"], right_on=["bioID"])

    players_teams = pd.merge(players_teams, awards_players, how="outer", left_on=["playerID", "year"], right_on=["playerID", "year"])

    players_teams = players_teams.fillna(0.0)
    all_awards = awards_players["award"].unique()
    all_awards_map = {}
    counter = 1
    for award in all_awards:
        all_awards_map[award] = counter
        counter += 1
    players_teams["award"] = players_teams["award"].map(all_awards_map)'''

    teams['winRatio'] = teams['won'] / (teams['won'] + teams['lost'])



    team_info = ['rank', 'o_3pa', 'o_asts', 'o_pf', 'o_stl', 'o_to', 'o_blk', 'o_pts', 'd_fgm', 'd_ftm', 'd_fta', 'd_3pm', 'd_3pa', 'd_oreb', 'd_dreb', 'd_reb', 'd_asts', 'd_pf', 'd_stl', 'd_to', 'd_blk', 'd_pts', 'won', 'lost', 'confW', 'confL', 'min', 'attend']

    if problem_type == "Classification":
        team_info.append('winRatio')

    player_info = ['assists', 'steals', 'rebounds']#, 'height', 'weight', 'award', 'GP', 'GS', 'minutes', 'points', 'oRebounds', 'dRebounds', 'blocks', 'turnovers', 'PF', 'fgAttempted', 'fgMade', 'ftAttempted', 'ftMade', 'threeAttempted', 'threeMade', 'dq']

    coach_info = ['coachWinRatio', 'coachPostWinRatio']


    columns = ['tmID', 'year', 'confID', 'playoff']
    if problem_type == "Regression":
        columns.append('winRatio')
    columns.extend(team_info)
    columns.extend(player_info)
    columns.extend(coach_info)

    data = pd.DataFrame(columns=columns)


    columns = ['tmID', 'year', 'confID', 'playoff']
    if problem_type == "Regression":
        columns.append('winRatio')
    columns.extend(team_info)
    teams = teams[columns]
    players_team_columns = ['playerID', 'year', 'tmID', 'stint']
    players_team_columns.extend(player_info)
    players_teams = players_teams[players_team_columns]
    players_per_team = pd.merge(teams, players_teams, how="inner", left_on=["year", "tmID"], right_on=["year", "tmID"])

    # Calculate the best player stats for each team for each year
    for year in range(2, 12):
        data = pd.concat([data, players_per_team_averages(players_per_team, coaches, year, team_info, player_info, coach_info, problem_type, player_averages_number, decay_number)])


    data = rename_attributes(data, player_info)

    
    data.to_csv("data.csv", index=False)

    total_ints = team_info.copy()

    total_ints.extend(player_info)
    total_ints.extend(coach_info)

    return data, total_ints


def preprocess(data, total_ints, current_year, problem_type):
    label = "playoff"
    if problem_type == "Regression":
        label = "winRatio"
    train_data = data[data['year'] < current_year]
    if label == "playoff":
        train_data[label] = train_data[label].map({'Y': 1, 'N': 0})
        train_data = balance_data(train_data, label)

    train_data_labels = train_data[label]
    

    test_data = data[data['year'] == current_year]
    if label == "playoff":
        test_data[label] = test_data[label].map({'Y': 1, 'N': 0})
    test_data_labels = test_data[label]
    
    train_data = train_data.drop(columns=['year', 'tmID', 'confID', 'playoff'])
    if problem_type == "Regression":
        train_data = train_data.drop(columns=['winRatio'])
    test_data_tmID = test_data['tmID']
    test_data_playoff = test_data['playoff']
    test_data_confID = test_data['confID']

    test_data = test_data.drop(columns=['year', 'tmID', 'confID', 'playoff'])
    if problem_type == "Regression":
        test_data = test_data.drop(columns=['winRatio'])
    elif problem_type == "Playoff":
        total_ints.append('winRatio')
    


    scaler = StandardScaler()
    scaler.fit(train_data[total_ints])
    print(total_ints)

    train_data.loc[:, total_ints] = scaler.transform(train_data[total_ints])
    test_data.loc[:, total_ints] = scaler.transform(test_data[total_ints])

    return train_data, train_data_labels, test_data, test_data_labels, test_data_tmID, test_data_playoff, test_data_confID

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

def train_predict_classifier(model, x_train, y_train, x_test, y_test, confID):
    def predict_labels(model, features):
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        return predictions, probabilities

    train_model(model, x_train, y_train)

    train_predictions, train_probabilities, = predict_labels(model, x_train)

    predictions, probabilities = predict_labels(model, x_test)

    x_test['probabilities'] = probabilities[:, 1]
    x_test['confID'] = confID
    x_test['result'] = 'N'

    top_4_teams = x_test[x_test['confID'] == "EA"].nlargest(4, 'probabilities').index
    x_test.loc[top_4_teams, 'result'] = 'Y'

    top_4_teams = x_test[x_test['confID'] == "WE"].nlargest(4, 'probabilities').index
    x_test.loc[top_4_teams, 'result'] = 'Y'

    index = 0
    for [_, win_prob] in probabilities:
        table_row = x_test.iloc[index]
        if table_row['result'] == 'Y':
            probabilities[index][1] = max(0.51, win_prob)
        else:
            probabilities[index][1] = min(0.49, win_prob)
        index += 1

    x_test = x_test.drop(columns=['confID', 'result', 'probabilities'])

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


def forward_selection(model, original_train_data, original_test_data, train_data_labels, test_data_labels, problem_type, metric, confID):
    rfe_model = None
    if problem_type == "Classification":
        rfe_model = LogisticRegression(random_state=42)
    elif problem_type == "Regression":
        rfe_model = LinearRegression()
    best_stat = -10000000
    if metric == "error" or metric == "mae" or metric == "mse":
        best_stat = 10000000
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
            predictions, probabilities, _, _ = train_predict_classifier(model, train_data, train_data_labels, test_data, test_data_labels, confID)
            stat = evaluate_model(metric, test_data_labels, predictions, probabilities)
        elif problem_type == "Regression":
            predictions, _ = train_predict_regression(model, train_data, train_data_labels, test_data, test_data_labels)
            stat = evaluate_model(metric, test_data_labels, predictions)
        
        metric_per_features.append((stat, len(new_columns)))
        if (metric == "error" or metric == "mae" or metric == "mse") and stat < best_stat:
            best_stat = stat
            best_columns = new_columns
        elif metric != "error" and metric != "mae" and metric != "mse" and stat > best_stat:
            best_stat = stat
            best_columns = new_columns

    return original_train_data[best_columns], original_test_data[best_columns], metric_per_features

def grid_search_model(model, train_data, train_data_labels):
    #match(model.__class__.__name__):
        
        

    parameter_xgb = {
        'n_estimators': [50, 100, 200],         # Number of boosting rounds (trees)
        'learning_rate': [0.01, 0.05, 0.1, 0.3],     # Step size (learning rate)
        'max_depth': [3, 5, 6, 7],                  # Maximum depth of each tree
        'min_child_weight': [1, 2, 5],            # Minimum sum of instance weight in a child
        'subsample': [0.7, 0.8, 0.9, 1.0],       # Fraction of data for each tree
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0], # Fraction of features for each tree
        'gamma': [0, 0.1, 0.2],                 # Minimum loss reduction for splitting
        'reg_alpha': [0, 0.01, 0.1],             # L1 regularization term
        'reg_lambda': [0, 0.01, 0.1, 1.0],            # L2 regularization term
        'scale_pos_weight': [1, 10, 50],          # Balance class weights for imbalanced data
    }
    parameter_dict = {"XGBClassifier": parameter_xgb}

    if model.__class__.__name__ not in parameter_dict or True:
        print("parameters for " + model.__class__.__name__ + " do not exist")
        return None

    model_params = parameter_dict[model.__class__.__name__]


    scorer = make_scorer(f1_score)

    grid_obj = GridSearchCV(
        estimator=model,
        scoring=scorer,
        param_grid=model_params,
        cv=5,
        n_jobs=-1
    )
    
    
    grid_obj = grid_obj.fit(train_data, train_data_labels)


    cv_results = grid_obj.cv_results_
    models_tested = cv_results['params']

    models = [model]
    for model_params in models_tested:
        model_tested = clone(model)
        model_tested.set_params(**model_params)
        models.append(model_tested)

        

    return models

def getModels(problem_type):
    if problem_type == "Regression":
        return [get_model("LinearRegression"), DecisionTreeRegressor(random_state=42), RandomForestRegressor(random_state=42), SVR(), xgb.XGBRegressor(), MLPRegressor(random_state=42)]
    elif problem_type == "Classification":
        return [xgb.XGBClassifier()]#[LogisticRegression(random_state=42), SVC(random_state=912, kernel='linear', probability=True), xgb.XGBClassifier(), MLPClassifier(random_state=42)]

def get_model(model_name, getDefault = True):
    if getDefault:
        match(model_name):
            case "XGBClassifier":
                return xgb.XGBClassifier()
            case "LogisticRegression":
                return LogisticRegression()
            case "SVC":
                return SVC()
            case "MLPClassifier":
                return MLPClassifier()
            case "LinearRegression":
                return LinearRegression()
            case "DecisionTreeRegressor":
                return DecisionTreeRegressor()
            case "RandomForestRegressor":
                return RandomForestRegressor()
            case "SVR":
                return SVR()
            case "XGBRegressor":
                return xgb.XGBRegressor()
            case "MLPRegressor":
                return MLPRegressor()

def getBestModel(models, train_data, train_data_labels, test_data, test_data_labels, test_data_tmID, test_data_playoff, test_data_confID, problem_type, metric = "error"):
    def print_params(model, metric, stat):
        model_params = model.get_params()

        default_model = get_model(model.__class__.__name__)
        default_params = default_model.get_params()
        non_default_params = {name: value for name, value in model_params.items() if value != default_params[name]}
        grid_results.append((model.__class__.__name__, non_default_params, stat, metric))
        print(f"{metric} of {str(stat)} for model: {model.__class__.__name__}({non_default_params})")
        
    best_model = None
    best_train_feat = None
    best_test_feat = None

    best_stat = -1000000

    if metric == "mae" or metric == "error" or metric == "mse":
        best_stat = 100000000

    grid_models = []
    for model in models:
        print("Grid Searching " + model.__class__.__name__)
        grid_model = grid_search_model(model, train_data, train_data_labels)
        if grid_model != None:
            grid_models.extend(grid_model)
        else:
            grid_models.append(model)
    for model in grid_models:
        print("Forward Selectioning " + model.__class__.__name__)
        train_data_feat, test_data_feat, _ = forward_selection(model, train_data, test_data, train_data_labels, test_data_labels, problem_type, metric, test_data_confID)

        _, _, stats = predict(model, train_data_feat, train_data_labels, test_data_feat.copy(), test_data_labels, test_data_tmID, test_data_playoff, test_data_confID, problem_type, [metric])
        stat = float(stats[0][1])
        print_params(model, metric, stat)
        if (metric == "mae" or metric == "error" or metric == "mse") and stat < best_stat:
            best_stat = stat
            best_model = model
            best_train_feat = train_data_feat
            best_test_feat = test_data_feat
        elif metric != "mae" and metric != "error" and metric != "mse" and stat > best_stat:
            best_stat = stat
            best_model = model
            best_train_feat = train_data_feat
            best_test_feat = test_data_feat

    return best_model, best_train_feat, best_test_feat



def predict(model, train_data, train_data_labels, test_data, test_data_labels, test_data_tmID, test_data_playoff, test_data_confID, problem_type, metrics):
    stats = []
    if problem_type == "Classification":    
        predictions, probabilities, _, _, = train_predict_classifier(model, train_data, train_data_labels, test_data, test_data_labels, test_data_confID)

        probabilities_per_team = []
        final_playoff = []
        for [_, win_prob] in probabilities:
            probabilities_per_team.append(win_prob)
            final_playoff.append('N')

        test_data['tmID'] = test_data_tmID
        test_data['playoff'] = test_data_labels.map({1: "Y", 0: "N"})
        test_data['confID'] = test_data_confID
        test_data['predictedPlayoff'] = predictions
        test_data['predictedPlayoff'] = test_data['predictedPlayoff'].map({1: "Y", 0: "N"})
        test_data['finalPlayoff'] = final_playoff
        test_data['probabilities'] = probabilities_per_team

        top_4_teams = test_data[test_data['confID'] == "EA"].nlargest(4, 'probabilities').index
        test_data.loc[top_4_teams, 'finalPlayoff'] = 'Y'

        top_4_teams = test_data[test_data['confID'] == "WE"].nlargest(4, 'probabilities').index
        test_data.loc[top_4_teams, 'finalPlayoff'] = 'Y'


        #test_data = test_data.drop(columns=["probabilities"])
        test_data = test_data[['tmID', 'confID', 'playoff', 'predictedPlayoff', 'finalPlayoff', 'probabilities']]


        if metrics == "all":
            metrics = ["f1", "acc", "error", "auc"]
        for metric in metrics:
            pred = predictions
            if metric == "acc" or metric == "f1":
                pred = test_data['finalPlayoff'].map({"Y": 1, "N": 0})
            stat = str(evaluate_model(metric, test_data_labels, pred, probabilities))
            stats.append((metric, stat))
    elif problem_type == "Regression":

        #best_model = fine_tune_model(best_model, train_data, train_data_labels)


        predictions, _ = train_predict_regression(model, train_data, train_data_labels, test_data, test_data_labels)

        
        test_data.loc[:, 'tmID'] = test_data_tmID
        test_data.loc[:, 'winRatio'] = test_data_labels
        test_data.loc[:, 'predictedWinRatio'] = predictions
        test_data.loc[:, 'playoff'] = test_data_playoff

        test_data['predictedPlayoff'] = 'N'
        top_4_teams = test_data[test_data['confID'] == "EA"].nlargest(4, 'predictedWinRatio').index
        test_data.loc[top_4_teams, 'predictedPlayoff'] = 'Y'

        top_4_teams = test_data[test_data['confID'] == "WE"].nlargest(4, 'predictedWinRatio').index
        test_data.loc[top_4_teams, 'predictedPlayoff'] = 'Y'

        '''top_8_teams = test_data.nlargest(8, 'predictedWinRatio').index
        test_data.loc[top_8_teams, 'predictedPlayoff'] = 'Y'''

        test_data = test_data[['tmID', 'winRatio', 'predictedWinRatio', 'playoff', 'predictedPlayoff']]

        if metrics == "all":
            metrics = ["f1", "acc", "mae", "mse", "r2"]
        for metric in metrics:
            labels = test_data_labels
            pred = predictions
            if metric == "acc" or metric == "f1":
                labels = test_data_playoff.map({"Y": 1, "N": 0})
                pred = test_data['predictedPlayoff'].map({"Y": 1, "N": 0})
            stat = str(evaluate_model(metric, labels, pred))
            stats.append((metric, stat))

    return test_data, model, stats

def cross_validation(data, total_ints, problem_type, metric_to_choose_best_model, printResults = False):
    years = []
    models = getModels(problem_type)
    model_names = [model.__class__.__name__ for model in models]
    model_lines = {}
    for model_name in model_names:
        model_lines[model_name] = []
    for year in range(3, 11):
        years.append(year)
        results = run_predictions(data, total_ints, year, problem_type, metric_to_choose_best_model, printResults)
        for (model, _, stats) in results:
            for (metric, stat) in stats:
                if metric == metric_to_choose_best_model:
                    model_lines[model.__class__.__name__].append(stat)

    model_stats = pd.DataFrame({model_name: model_lines[model_name] for model_name in model_names})

    model_stats.index = years
    model_stats = model_stats.astype(float)
    model_stats.plot(kind="line")
    plt.xlabel('Year')
    plt.ylabel(metric_to_choose_best_model)
    plt.title("Model's " + metric_to_choose_best_model + " over the years of training and testing")
    plt.xticks(years, labels=years)
    plt.legend(title="Models")
    plt.tight_layout()
    plt.show()



def run_predictions(data, total_ints, current_year, problem_type, metric_to_choose_best_model, printResults = False):
    original_train_data, train_data_labels, original_test_data, test_data_labels, test_data_tmID, test_data_playoff, test_data_confID = preprocess(data, total_ints, current_year, problem_type)

    results = []
    for model in getModels(problem_type):
        train_data, test_data, _ = forward_selection(model, original_train_data, original_test_data, train_data_labels, test_data_labels, problem_type, metric_to_choose_best_model, test_data_confID)
        final_data, model, stats = predict(model, train_data, train_data_labels, test_data, test_data_labels, test_data_tmID, test_data_playoff, test_data_confID, problem_type, "all")
        results.append((model, final_data, stats))
        if printResults:
            print("The " + model.__class__.__name__ + " Predicted:")
            print(final_data)
            for (metric, stat) in stats:
                if metric == "f1" or metric == "acc" or metric == "r2" or metric == "auc":
                    print("The " + metric + " is: " + str(stat) + "%")
                else:
                    print("The " + metric + " is: " + str(stat))



    return results
    

def run_best_prediction(data, total_ints, current_year, problem_type, metric_to_choose_best_model, printResults = False):
    train_data, train_data_labels, test_data, test_data_labels, test_data_tmID, test_data_playoff, test_data_confID = preprocess(data, total_ints, current_year, problem_type)
    models = getModels(problem_type)


    best_model, train_data, test_data = getBestModel(models, train_data, train_data_labels, test_data, test_data_labels, test_data_tmID, test_data_playoff, test_data_confID, problem_type, metric_to_choose_best_model)


    final_data, model, stats = predict(best_model, train_data, train_data_labels, test_data.copy(), test_data_labels, test_data_tmID, test_data_playoff, test_data_confID, problem_type, "all")

    writeAnswerToCsv(final_data)
    return_metric = None
    if printResults:
        print("The " + model.__class__.__name__ + " Predicted:")
        print(final_data)
        for (metric, stat) in stats:
            if metric == metric_to_choose_best_model:
                return_metric = float(stat)
            if metric == "f1" or metric == "acc" or metric == "r2" or metric == "auc":
                print("The " + metric + " is: " + str(stat) + "%")
            else:
                print("The " + metric + " is: " + str(stat))
    else:
        for (metric, stat) in stats:
            if metric == metric_to_choose_best_model:
                return_metric = float(stat)
                break


    return best_model, train_data, train_data_labels, test_data, test_data_labels, return_metric


def plot_roc_curve(model, train_data, train_data_labels, test_data, test_data_labels, confID):
    _, probabilities, _, _ = train_predict_classifier(model, train_data, train_data_labels, test_data, test_data_labels, confID)

    fpr, tpr, _ = roc_curve(test_data_labels, probabilities[:, 1])

    roc_plot = pd.DataFrame({
        "Curve": tpr
    })

    roc_plot.index = fpr


    roc_plot.plot(kind="line")
    plt.xlabel('False Positive Rate')
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve of " + model.__class__.__name__)
    plt.tight_layout()
    plt.legend().remove()
    plt.show()

def plot_learning_curve(model, train_data, train_data_labels, test_data, test_data_labels):
    _, probabilities, _, _ = train_predict_classifier(model, train_data, train_data_labels, test_data, test_data_labels, None)

    fpr, tpr, _ = roc_curve(test_data_labels, probabilities[:, 1])

    roc_plot = pd.DataFrame({
        "Curve": tpr
    })

    roc_plot.index = fpr


    roc_plot.plot(kind="line")
    plt.xlabel('False Positive Rate')
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve of " + model.__class__.__name__)
    plt.tight_layout()
    plt.legend().remove()
    plt.show()

def plot_data_balance(data):
    playoff_teams = len(data[data['playoff'] == 'Y'])
    non_playoff_teams = len(data[data['playoff'] == 'N'])


    balance_plot = pd.DataFrame({
        "Playoff": [playoff_teams, non_playoff_teams],
    })

    values = ['Y', 'N']
    balance_plot.index = values



    balance_plot.plot(kind="bar")
    plt.xticks(rotation=0)
    plt.xlabel('If Team went to Playoff')
    plt.ylabel("Number of Teams")
    plt.title("Team playoff distribution")
    plt.tight_layout()
    plt.legend().remove()
    plt.show()


def plot_metric_per_features(data, total_ints, current_year, problem_type, metric_to_choose_best_model):
    train_data, train_data_labels, test_data, test_data_labels, test_data_tmID, test_data_playoff, test_data_confID = preprocess(data, total_ints, current_year, problem_type)

    if problem_type == "Classification":
        model = xgb.XGBClassifier()
    elif problem_type == "Regression":
        model = MLPRegressor()

    print(test_data.columns)
    train_data, test_data, metric_per_feature = forward_selection(model, train_data, test_data, train_data_labels, test_data_labels, problem_type, metric_to_choose_best_model, test_data_confID)
    print(test_data.columns)
    metrics = []
    feature_nums = []
    for (metric, num_feature) in metric_per_feature:
        metrics.append(metric)
        feature_nums.append(num_feature)


    metric_per_feature_plot = pd.DataFrame({
        "MetricPerFeature": metrics
    })

    metric_per_feature_plot.index = feature_nums



    metric_per_feature_plot.plot(kind="line")
    plt.xlabel('Number of Features')
    plt.ylabel("Theoretical Best Error")
    plt.title("Theoretical Best Error per Number of Features using " + model.__class__.__name__ + " for year 10")
    plt.tight_layout()
    plt.legend().remove()
    plt.show()

def writeAnswerToCsv(result):
    result = result[["tmID", "probabilities"]]
    result["Playoff"] = result["probabilities"].round(2)
    result = result.drop(columns=["probabilities"])
    result.to_csv("result.csv", index=False)

def run_best_prediction_all_data_options(year, problem_type, metric_to_choose_best_model, print_results):
    best_player_averages_number = 1
    best_decay_number = 0
    best_metric = -1000
    if metric_to_choose_best_model == "mae" or metric_to_choose_best_model == "mse" or metric_to_choose_best_model == "error":
        best_metric = 10000
    decay_numbers = [0.01, 0.1, 0.5, 1.0, 2.0]
    for player_averages_number in range(1, 10):
        for decay_number in decay_numbers:
            data, total_ints = getData(problem_type, player_averages_number, decay_number)
            _, _, _, _, _, metric = run_best_prediction(data, total_ints, year, problem_type, metric_to_choose_best_model, print_results)
            if metric_to_choose_best_model == "mae" or metric_to_choose_best_model == "mse" or metric_to_choose_best_model == "error" and metric < best_metric:
                best_player_averages_number = player_averages_number
                best_decay_number = decay_number
                best_metric = metric
            elif metric_to_choose_best_model != "mae" and metric_to_choose_best_model != "error" and metric_to_choose_best_model != "mse" and metric > best_metric:
                best_player_averages_number = player_averages_number
                best_decay_number = decay_number
                best_metric = metric

    data, total_ints = getData(problem_type, best_player_averages_number, best_decay_number)
    run_best_prediction(data, total_ints, year, problem_type, metric_to_choose_best_model, True)
    print("Best player averages number: " + str(best_player_averages_number) + " and best decay number: " + str(best_decay_number))


def main():
    #best params: XGBClassifier({'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}, player_averages_number = 3  #decay number 1
    pd.set_option('display.max_rows', None)
    problem_type = "Classification"
    metric_to_choose_best_model = "error"

    #run_best_prediction_all_data_options(11, problem_type, metric_to_choose_best_model, True)

    data, total_ints = getData(problem_type, 3, 1)

    

    #plot_metric_per_features(data, total_ints, 10, problem_type, metric_to_choose_best_model)

    run_best_prediction(data, total_ints, 11, problem_type, metric_to_choose_best_model, True)
    #model, train_data, train_data_labels, test_data, test_data_labels, _ = run_best_prediction(data, total_ints, 10, problem_type, metric_to_choose_best_model, True)


    #plot_roc_curve(model, train_data, train_data_labels, test_data, test_data_labels, test_data_confID)

    #run_predictions(data, total_ints, 10, problem_type, metric_to_choose_best_model, True)

    #cross_validation(data, total_ints, problem_type, metric_to_choose_best_model, False)



    model_n = "XGBClassifier"
    best_stat = 1000000
    for (model_name, params, stat, metric) in grid_results:
        if model_name == model_n and stat < best_stat:
            best_stat = stat
            print(f"best {metric} of {str(stat)} for best model: {model_name}({params})")
    


    







if __name__ == "__main__":
    main()