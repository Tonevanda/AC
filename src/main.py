import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from functools import partial


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


    players_per_team_year = players_per_team_year[['playerID', 'year', 'tmID', 'finals']]


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


    columns = ['tmID', 'year', 'finals']
    columns.extend(team_info)
    columns.extend(player_info)
    columns.extend(coach_info)
    players_per_team_year = players_per_team_year[columns]


    players_per_team_year['finals'] = players_per_team_year['finals'].fillna('L')

    players_per_team_year = players_per_team_year.rename(columns={'finals': 'hasWon'})

    players_per_team_year['hasWon'] = players_per_team_year['hasWon'].map({'L': 0, 'W': 1})

    return players_per_team_year

def balance_data(data):
    minority_class = data[data['hasWon'] == 1]
    majority_class = data[data['hasWon'] == 0]

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

    columns = ['tmID', 'year', 'hasWon']
    columns.extend(team_info)
    columns.extend(player_info)
    columns.extend(coach_info)

    data = pd.DataFrame(columns=columns)
    #test_table = pd.DataFrame(columns=columns)


    columns = ['tmID', 'year', 'finals']
    columns.extend(team_info)
    teams = teams[columns]
    players_team_columns = ['playerID', 'year', 'tmID', 'stint']
    players_team_columns.extend(player_info)
    players_teams = players_teams[players_team_columns]
    players_per_team = pd.merge(teams, players_teams, how="inner", left_on=["year", "tmID"], right_on=["year", "tmID"])

    # Calculate the best player stats for each team for each year
    for year in range(2, 11):
        data = pd.concat([data, players_per_team_averages(players_per_team, coaches, year, team_info, player_info, coach_info)])

    #test_table = pd.concat([test_table, players_per_team_averages(players_per_team, 10, team_info, player_info)])

    data = rename_attributes(data, player_info)

    data['hasWon'] = data['hasWon'].astype(int)

    #test_table = rename_attributes(test_table, player_info)


    #training_table = balance_data(training_table)
    
    data.to_csv("data.csv", index=False)
    #test_table.to_csv("test.csv", index=False)

    print('\n\n')

    current_year = 10

    train_data = data[data['year'] < current_year]
    train_data = balance_data(train_data)
    train_data_labels = train_data['hasWon']
    

    test_data = data[data['year'] == current_year]
    test_data_labels = test_data['hasWon']
    
    train_year = train_data['year']
    train_tmID = train_data['tmID']
    train_data = train_data.drop(columns=['hasWon', 'year', 'tmID'])
    test_tmID = test_data['tmID']
    test_year = test_data['year']
    test_data = test_data.drop(columns=['year', 'hasWon', 'tmID'])
    

    total_ints = team_info.copy()

    total_ints.extend(player_info)




    scaler = StandardScaler()
    scaler.fit(train_data[total_ints])


    train_data.loc[:, total_ints] = scaler.transform(train_data[total_ints])
    test_data.loc[:, total_ints] = scaler.transform(test_data[total_ints])


    def train_model(classifier, x, y):
        classifier.fit(x, y)

    def get_predictions(classifier, features, years, tmID):
        label_probabilities = classifier.predict_proba(features)
        features_copy = features.copy()
        features_copy['year'] = years
        features_copy['tmID'] = tmID
        win_probabilities = []
        for prob in label_probabilities:
            win_probabilities.append(prob[1])
        features_copy['probability'] = win_probabilities


        
        for i in range(1, 12):
            features_year = features_copy[features_copy['year'] == i]
            if not features_year.empty:
                best_winner_probability = 0
                best_winner_tmID = ""
                for w_prob, tmID in zip(features_year['probability'], features_year['tmID']):
                    if(w_prob > best_winner_probability):
                        best_winner_probability = w_prob
                        best_winner_tmID = tmID

                predictions = []


                for tmID in features_year['tmID']:
                    if(tmID == best_winner_tmID):
                        predictions.append(1)
                    else:
                        predictions.append(0)

                features_year.loc[:, ['probability']] = predictions
                features_copy.loc[features_copy['year'] == i, :] = features_year.values

        
        return features_copy["probability"]
    def teste():
        return None
    def f1(features, labels, classifier, years=train_year, tmID=train_tmID):
        predictions = get_predictions(classifier, features, years, tmID)
        return f1_score(labels, labels, pos_label=1) * 100.0
    
    def accuracy(features, labels, classifier, years=None, tmID=None):
        predictions = get_predictions(classifier, features, years, tmID)
        return (sum(labels == predictions) / float(len(predictions))) * 100.0



    def predict_labels(classifier, features, labels, years=None, tmID=None):
        
        return classifier.predict_proba(features), get_predictions(classifier, features, years, tmID), f1(features, labels, classifier, years, tmID), accuracy(features, labels, classifier, years, tmID)
    
    def train_predict(classifier, x_train, y_train, x_test, y_test, train_years=None, train_tmID=None, test_years = None, test_tmID = None, printResults = False):
        if(printResults):
            print("Training using " + classifier.__class__.__name__)

        train_model(classifier, x_train, y_train)

        _, predictions, train_f1, acc = predict_labels(classifier, x_train, y_train, train_years, train_tmID)
        if(printResults):
            print("F1 score for the training set: " + str(train_f1) + '%')
            print("Accuracy score for the training set: " + str(acc) + '%\n\n')

        probabilities, predictions, f1, acc = predict_labels(classifier, x_test, y_test, test_years, test_tmID)
        if(printResults):
            print("F1 score for the test set: " + str(f1) + '%')
            print("Accuracy score for the test set: " + str(acc) + '%\n\n')

        return probabilities, predictions, f1, acc, train_f1
    
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


    classifiers = [LogisticRegression(random_state=42), SVC(random_state=912, kernel="rbf", probability=True), MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
                    DecisionTreeClassifier(random_state=42), KNeighborsClassifier(n_neighbors=5)]


    #xgboost = xgb.XGBClassifier(seed=82)

    best_classifier = None
    best_f1 = 0
    best_train_f1 = 0

    for classifier in classifiers:
        _, _, _f1, acc, train_f1 = train_predict(classifier, train_data, train_data_labels, test_data, test_data_labels, train_year, train_tmID, test_year, test_tmID, False)
        if (_f1 > best_f1) or (_f1 == best_f1 and train_f1 > best_train_f1):
            best_classifier = classifier
            best_f1 = _f1
            best_train_f1 = train_f1


    print("The best Classifier is " + best_classifier.__class__.__name__)


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

    print(test_data)



    #for player_sta

if __name__ == "__main__":
    main()