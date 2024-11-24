import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


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



def players_per_team_averages(players_per_team, year, team_info, player_info):
    """
    This function calculates the average of the players in each team for a given year, based on the past years.
    If year is 7, it will calculate the averages for the years 1 to 6.
    """
    players_per_team_year = players_per_team[players_per_team["year"] == year]

    players_per_team_prev_years = players_per_team[players_per_team["year"] < year].copy()



    





    


    player_averages = weighted_average_attributes(players_per_team_prev_years, year, player_info, "playerID")

    player_averages['Total'] = player_averages.sum(axis=1)

    player_averages.reset_index()


    players_per_team_year = players_per_team_year[['playerID', 'year', 'tmID', 'finals']]


    players_per_team_year = pd.merge(players_per_team_year, player_averages, how="inner", on="playerID")

    players_per_team_year = players_per_team_year.loc[players_per_team_year.groupby("tmID")["Total"].idxmax()]

    teams_prev_years = players_per_team_prev_years.drop_duplicates(subset=['tmID', 'year'])

    team_averages = weighted_average_attributes(teams_prev_years, year, team_info, "tmID")



    team_averages.reset_index()




    players_per_team_year = pd.merge(players_per_team_year, team_averages, how="inner", on=["tmID"])

    columns = ['tmID', 'year', 'finals']
    columns.extend(team_info)
    columns.extend(player_info)
    players_per_team_year = players_per_team_year[columns]


    players_per_team_year['finals'] = players_per_team_year['finals'].fillna('L')

    players_per_team_year = players_per_team_year.rename(columns={'finals': 'hasWon'})

    #players_per_team_year['hasWon'] = players_per_team_year['hasWon'].replace({'L': 0, 'W': 1})

    return players_per_team_year

def balance_data(data):
    minority_class = data[data['hasWon'] == 'W']
    majority_class = data[data['hasWon'] == 'L']

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



    player_info = ['points', 'assists', 'steals', 'blocks', 'rebounds', 'stint', 'GP', 'GS']

    columns = ['tmID', 'year', 'hasWon']
    columns.extend(team_info)
    columns.extend(player_info)

    data = pd.DataFrame(columns=columns)
    #test_table = pd.DataFrame(columns=columns)


    columns = ['tmID', 'year', 'finals']
    columns.extend(team_info)
    teams = teams[columns]
    players_team_columns = ['playerID', 'year', 'tmID']
    players_team_columns.extend(player_info)
    players_teams = players_teams[players_team_columns]
    players_per_team = pd.merge(teams, players_teams, how="inner", left_on=["year", "tmID"], right_on=["year", "tmID"])

    # Calculate the best player stats for each team for each year
    for year in range(2, 11):
        data = pd.concat([data, players_per_team_averages(players_per_team, year, team_info, player_info)])

    #test_table = pd.concat([test_table, players_per_team_averages(players_per_team, 10, team_info, player_info)])

    data = rename_attributes(data, player_info)

    #test_table = rename_attributes(test_table, player_info)


    #training_table = balance_data(training_table)
    
    data.to_csv("data.csv", index=False)
    #test_table.to_csv("test.csv", index=False)

    print('\n\n')

    data = data.drop(columns=['tmID'])


    train_data = data[data['year'] < 10]
    train_data = balance_data(train_data)
    train_data_labels = train_data['hasWon']
    

    test_data = data[data['year'] == 10]
    test_data_labels = test_data['hasWon']
    

    train_data = train_data.drop(columns=['hasWon', 'year'])
    test_data = test_data.drop(columns=['year', 'hasWon'])

    total_ints = team_info.copy()

    total_ints.extend(player_info)




    scaler = StandardScaler()
    scaler.fit(train_data[total_ints])


    train_data.loc[:, total_ints] = scaler.transform(train_data[total_ints])
    test_data.loc[:, total_ints] = scaler.transform(test_data[total_ints])


    def train_model(classifier, x, y):
        classifier.fit(x, y)

    def predict_labels(classifier, features, labels):

        label_predict = classifier.predict(features)

        return f1_score(labels, label_predict, pos_label='W') * 100.0, (sum(labels == label_predict) / float(len(label_predict))) * 100.0
    
    def train_predict(classifier, x_train, y_train, x_test, y_test):

        print("Training using " + classifier.__class__.__name__)

        train_model(classifier, x_train, y_train)

        f1, acc = predict_labels(classifier, x_train, y_train)
        print("F1 score for the training set: " + str(f1) + '%')
        print("Accuracy score for the training set: " + str(acc) + '%\n\n')

        f1, acc = predict_labels(classifier, x_test, y_test)
        print("F1 score for the test set: " + str(f1) + '%')
        print("Accuracy score for the test set: " + str(acc) + '%\n\n')

    '''print(train_data.head())
    print(train_data_labels.head())
    print(test_data)
    print(test_data_labels)'''

    logistic_regression = LogisticRegression(random_state=42)
    svc = SVC(random_state=912, kernel="rbf")
    neural_network = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)  # K-Nearest Neighbors
    #xgboost = xgb.XGBClassifier(seed=82)


    train_predict(logistic_regression, train_data, train_data_labels, test_data, test_data_labels)
    train_predict(svc, train_data, train_data_labels, test_data, test_data_labels)
    train_predict(neural_network, train_data, train_data_labels, test_data, test_data_labels)
    train_predict(decision_tree, train_data, train_data_labels, test_data, test_data_labels)
    train_predict(knn, train_data, train_data_labels, test_data, test_data_labels)
    #train_predict(xgboost, train_data, train_data_labels, test_data, test_data_labels)


    

    #for player_sta

if __name__ == "__main__":
    main()