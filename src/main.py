import pandas as pd

def loadData():
    awards_players = pd.read_csv("../data/awards_players.csv")
    coaches = pd.read_csv("../data/coaches.csv")
    players_teams = pd.read_csv("../data/players_teams.csv")
    players = pd.read_csv("../data/players.csv")
    series_post = pd.read_csv("../data/teams_post.csv")
    teams = pd.read_csv("../data/teams.csv")
    series_post = pd.read_csv("../data/series_post.csv")
    return awards_players, coaches, players_teams, players, series_post, teams, series_post

def players_per_team_averages(players_per_team, year, team_info, player_info):
    """
    This function calculates the average of the players in each team for a given year, based on the past years.
    If year is 7, it will calculate the averages for the years 1 to 6.
    """
    players_per_team_year = players_per_team[players_per_team["year"] == year]

    players_per_team_prev_years = players_per_team[players_per_team["year"] < year]

    player_averages = players_per_team_prev_years.groupby("playerID")[player_info].mean()

    player_averages['Total'] = player_averages.sum(axis=1)

    player_averages.reset_index()


    players_per_team_year = players_per_team_year[['playerID', 'year', 'tmID', 'finals']]


    players_per_team_year = pd.merge(players_per_team_year, player_averages, how="inner", on="playerID")

    players_per_team_year = players_per_team_year.loc[players_per_team_year.groupby("tmID")["Total"].idxmax()]


    

    team_averages = players_per_team_prev_years.groupby("tmID")[team_info].mean()


    team_averages.reset_index()




    players_per_team_year = pd.merge(players_per_team_year, team_averages, how="inner", on=["tmID"])

    columns = ['tmID', 'year', 'finals']
    columns.extend(team_info)
    columns.extend(player_info)
    players_per_team_year = players_per_team_year[columns]

    players_per_team_year['finals'] = players_per_team_year['finals'].fillna('L')

    players_per_team_year = players_per_team_year.rename(columns={'finals': 'hasWon'})

    players_per_team_year['hasWon'] = players_per_team_year['hasWon'].replace({'L': 0, 'W': 1})

    return players_per_team_year

def main():
    pd.set_option('display.max_rows', None)
    awards_players, coaches, players_teams, players, series_post, teams, series_post = loadData()

    team_info = ['rank', 'o_fgm', 'o_fga', 'o_ftm', 'o_fta']



    player_info = ['points', 'assists', 'steals', 'blocks', 'rebounds', 'stint', 'GP', 'GS']

    columns = ['tmID', 'year', 'hasWon']
    columns.extend(team_info)
    columns.extend(player_info)

    final_table = pd.DataFrame(columns=columns)


    columns = ['tmID', 'year', 'finals']
    columns.extend(team_info)
    teams = teams[columns]
    players_team_columns = ['playerID', 'year', 'tmID']
    players_team_columns.extend(player_info)
    players_teams = players_teams[players_team_columns]
    players_per_team = pd.merge(teams, players_teams, how="inner", left_on=["year", "tmID"], right_on=["year", "tmID"])

    # Calculate the best player stats for each team for each year
    for year in range(2, 11):
        final_table = pd.concat([final_table, players_per_team_averages(players_per_team, year, team_info, player_info)])

    columns = {}
    for player_stat in player_info:
        columns[player_stat] = "BP" + player_stat
    # Rename the columns to Best Player (BP)
    final_table = final_table.rename(columns=columns)

    final_table.to_csv("res.csv", index=False)

if __name__ == "__main__":
    main()