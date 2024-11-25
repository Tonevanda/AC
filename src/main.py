import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, make_scorer


def loadData():
    awards_players = pd.read_csv("../data/awards_players.csv")
    coaches = pd.read_csv("../data/coaches.csv")
    players_teams = pd.read_csv("../data/players_teams.csv")
    players = pd.read_csv("../data/players.csv")
    teams_post = pd.read_csv("../data/teams_post.csv")
    teams = pd.read_csv("../data/teams.csv")
    series_post = pd.read_csv("../data/series_post.csv")
    return awards_players, coaches, players_teams, players, teams_post, teams, series_post

def weighted_average_attributes(players_per_team_prev_years_original, current_year, columns, groupby_attribute):
    players_per_team_prev_years = players_per_team_prev_years_original.copy()
    k = 0.5
    players_per_team_prev_years['Weight'] = np.exp(-k * (current_year - players_per_team_prev_years['year']))
    for attribute in columns:
        players_per_team_prev_years[attribute] = players_per_team_prev_years[attribute] * players_per_team_prev_years['Weight']


    attribute_averages = players_per_team_prev_years.groupby(groupby_attribute)[columns].sum() 



    summed_weights = players_per_team_prev_years['Weight'].unique().sum()
    for attribute in columns:
        attribute_averages[attribute] = attribute_averages[attribute] / summed_weights

    return attribute_averages



def players_per_team_averages(players_per_team, coaches, year, team_info, player_info, coach_info):
    """
    This function calculates the average of the players in each team for a given year, based on the past years.
    If year is 7, it will calculate the averages for the years 1 to 6.
    """
    players_per_team_year = players_per_team[players_per_team["year"] == year]
    players_per_team_year = players_per_team_year[players_per_team_year['stint'] == 0]


    coaches_year = coaches[coaches["year"] == year]
    coaches_year = coaches_year[coaches_year['stint'] == 0]

    players_per_team_prev_years = players_per_team[players_per_team["year"] < year].copy()
    coaches_prev_years = coaches[coaches["year"] < year].copy()

    players_per_team_prev_years = players_per_team_prev_years.drop(columns=['stint'])
    coaches_prev_years = coaches_prev_years.drop(columns=['stint'])
    players_per_team_year = players_per_team_year.drop(columns=['stint'])
    coaches_year = coaches_year.drop(columns=['stint'])


    player_averages = weighted_average_attributes(players_per_team_prev_years, year, player_info, "playerID")

    player_averages['Total'] = player_averages.sum(axis=1)

    player_averages.reset_index()


    players_per_team_year = players_per_team_year[['playerID', 'year', 'tmID', 'won', 'lost']]

    players_per_team_year['winRatio'] = players_per_team_year['won'] / (players_per_team_year['won'] + players_per_team_year['lost'])
    players_per_team_year = players_per_team_year.drop(columns=['won', 'lost'])


    players_per_team_year = pd.merge(players_per_team_year, player_averages, how="inner", on="playerID")

    players_per_team_year = players_per_team_year.loc[players_per_team_year.groupby("tmID")["Total"].idxmax()]

    coach_info_2 = ['won', 'lost', 'post_wins', 'post_losses']


    coach_averages = weighted_average_attributes(coaches_prev_years, year, coach_info_2, "coachID")

    coach_averages.reset_index()

    
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

    team_averages = weighted_average_attributes(teams_prev_years, year, team_info, "tmID")



    team_averages.reset_index()




    players_per_team_year = pd.merge(players_per_team_year, team_averages, how="inner", on=["tmID"])

    players_per_team_year = pd.merge(players_per_team_year, coaches_year, how="inner", on=["tmID"])


    columns = ['tmID', 'year', 'winRatio']
    columns.extend(team_info)
    columns.extend(player_info)
    columns.extend(coach_info)
    players_per_team_year = players_per_team_year[columns]



    return players_per_team_year

def balance_data(data):
    most_common_value = data['winRatio'].value_counts().idxmax()
    minority_class = data[data['winRatio'] != most_common_value]
    majority_class = data[data['winRatio'] == most_common_value]



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
    

def main():
    pd.set_option('display.max_rows', None)
    awards_players, coaches, players_teams, players, teams_post, teams, series_post = loadData()


    team_info = ['rank', 'o_fgm', 'o_fga', 'o_ftm', 'o_fta', 'o_3pm', 'o_3pa', 'o_oreb', 'o_dreb', 'o_reb', 'o_asts', 'o_pf', 'o_stl', 'o_to', 'o_blk', 'o_pts', 'd_fgm', 'd_ftm', 'd_fta', 'd_3pm', 'd_3pa', 'd_oreb', 'd_dreb', 'd_reb', 'd_asts', 'd_pf', 'd_stl', 'd_to', 'd_blk', 'd_pts', 'won', 'lost', 'confW', 'confL', 'min', 'attend']



    player_info = ['points', 'assists', 'steals', 'blocks', 'rebounds', 'GP', 'GS']

    coach_info = ['coachWinRatio', 'coachPostWinRatio']

    columns = ['tmID', 'year', 'winRatio']
    columns.extend(team_info)
    columns.extend(player_info)
    columns.extend(coach_info)

    data = pd.DataFrame(columns=columns)


    columns = ['tmID', 'year']
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

    print('\n\n')

    current_year = 10

    train_data = data[data['year'] < current_year]
    train_data = balance_data(train_data)
    train_data_labels = train_data['winRatio']
    

    test_data = data[data['year'] == current_year]
    test_data_labels = test_data['winRatio']
    
    train_data = train_data.drop(columns=['winRatio', 'year', 'tmID'])
    test_data_tmID = test_data['tmID']
    test_data_winRatio = test_data['winRatio']
    test_data = test_data.drop(columns=['year', 'winRatio', 'tmID'])
    

    total_ints = team_info.copy()

    total_ints.extend(player_info)




    scaler = StandardScaler()
    scaler.fit(train_data[total_ints])


    train_data.loc[:, total_ints] = scaler.transform(train_data[total_ints])
    test_data.loc[:, total_ints] = scaler.transform(test_data[total_ints])



    
    def train_model(classifier, x, y):
        classifier.fit(x, y)


    
    def train_predict(model, x_train, y_train, x_test, y_test, printResults = False):
        def predict_labels(model, features, labels):
            predictions = model.predict(features)
            return predictions, mean_absolute_error(labels, predictions), mean_squared_error(labels, predictions), r2_score(labels, predictions)
        
        if(printResults):
            print("Training using " + model.__class__.__name__)

        train_model(model, x_train, y_train)

        _, train_mae, train_mse, train_r2 = predict_labels(model, x_train, y_train)
        if(printResults):
            print("Mean Absolute Error for the train set: " + str(train_mae) + '%')
            print("Mean Squared Error for the train set: " + str(train_mse) + '%')
            print("R2 Score for the train set: " + str(train_r2) + '%\n\n')

        predictions, mae, mse, r2 = predict_labels(model, x_test, y_test)
        if(printResults):
            print("Mean Absolute Error for the test set: " + str(mae) + '%')
            print("Mean Squared Error for the test set: " + str(mse) + '%')
            print("R2 Score for the test set: " + str(r2) + '%\n\n')

        return predictions, mae, mse, r2, train_mae, train_mse, train_r2
    
    def fine_tune_classifier(classifier, train_data, train_data_labels, train_years, train_tmID):
        parameter_svc = {
            'C': [0.1, 1, 10, 100],               # Regularization strength
            'kernel': ['linear', 'rbf', 'poly'],  # Kernel type
            'gamma': ['scale', 'auto', 0.1, 1],   # Kernel coefficient
            'degree': [2, 3, 4],                  # Degree for 'poly' kernel
            'class_weight': [None, 'balanced']    # Class weights
        }

        parameter_mlp = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  # Number of neurons in hidden layers
            'activation': ['relu', 'tanh', 'logistic'],                 # Activation function for hidden layers
            'solver': ['adam', 'sgd'],                                  # Optimization algorithm
            'alpha': [0.0001, 0.001, 0.01],                             # L2 regularization term
            'learning_rate': ['constant', 'adaptive'],                  # Learning rate schedule
            'learning_rate_init': [0.001, 0.01],                        # Initial learning rate
            'max_iter': [50],                                # Maximum number of iterations
            'batch_size': [32, 64, 'auto'],                             # Batch size for training
        }

        parameter_dict = {"MLPClassifier": parameter_mlp, "SVC": parameter_svc}

        best_classifier_params = parameter_dict[classifier.__class__.__name__]



        f1_scorer = make_scorer(f1_score, pos_label=1)

        grid_obj = GridSearchCV(
            classifier,
            scoring=f1_scorer,
            param_grid=best_classifier_params,
            cv=5)
        
        
        grid_obj = grid_obj.fit(train_data, train_data_labels)

        classifier = grid_obj.best_estimator_

        return classifier
    
    def get_best_classifier(classifier_name):
        match(classifier_name):
            case "MLPClassifier":
                return MLPClassifier(batch_size=32, hidden_layer_sizes=(50,), learning_rate_init=0.01,
                    max_iter=50, random_state=42)
        return None


    models = [LinearRegression(), DecisionTreeRegressor(random_state=42), RandomForestRegressor(random_state=42), SVR()]
    #xgboost = xgb.XGBClassifier(seed=82)

    best_model = None



    for model in models:
        test_predictions, mae, mse, r2, train_mae, train_mse, train_r2 = train_predict(model, train_data, train_data_labels, test_data, test_data_labels, True)
        '''if (test_f1 > best_f1) or (test_f1 == best_f1 and test_acc > best_acc) or (test_f1 == best_f1 and test_acc == best_acc and train_f1 > best_train_f1) or (test_f1 == best_f1 and test_acc == best_acc and train_f1 == best_train_f1 and train_acc > best_train_acc):
            best_model = model
            best_f1 = test_f1
            best_acc = test_acc
            best_train_f1 = train_f1
            best_train_acc = train_acc'''


    print("The best Model is " + model.__class__.__name__)

    test_data['tmID'] = test_data_tmID
    test_data['winRatio'] = test_data_winRatio
    test_data['predictions'] = test_predictions

    print(test_data)

    '''
    #predictions,  _, _ = train_predict(best_classifier, train_data, train_data_labels, test_data, test_data_labels, True)

    #probabilities, predictions, _, _, _, = train_predict(best_classifier, train_data, train_data_labels, test_data, test_data_labels, train_year, train_tmID, test_year, test_tmID, True)


    #print(fine_tune_classifier(best_classifier, train_data, train_data_labels, train_year, train_tmID))
    #best_classifier = get_best_classifier(best_classifier.__class__.__name__)

    if best_classifier == None:
        print("Best classifier is not available")
        return
    
    probabilities, predictions, _, _, _ = train_predict(best_classifier, train_data, train_data_labels, test_data, test_data_labels, train_year, train_tmID, test_year, test_tmID, True)




    predictions_probs = []

    for [_, win_prob] in probabilities:
        predictions_probs.append(win_prob)


    test_data['tmID'] = test_tmID
    test_data['hasWon'] = predictions
    test_data['probs'] = predictions_probs

    print(test_data)'''



    #for player_sta

if __name__ == "__main__":
    main()