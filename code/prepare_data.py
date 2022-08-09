import pandas as pd
from datetime import datetime
import numpy as np

def get_data_from_csv(columns=None, countries=None, start=None, end=None):
    """Creates pandas dataframe from .csv file.

    Data will be filtered based on data column name, list of countries to be plotted and
    time frame chosen.

    Args:
        columns (list(string)): a list of data columns you want to include
        countries ((list(string), optional): List of countries you want to include.
            If none is passed, dataframe should be filtered for the 6 countries with the highest
            number of cases per million at the last current date available in the timeframe chosen.
        start (string, optional): The first date to include in the returned dataframe.
            If specified, records earlier than this will be excluded.
            Default: include earliest date
            Example format: "2021-10-10"
        end (string, optional): The latest date to include in the returned data frame.
            If specified, records later than this will be excluded.
            Example format: "2021-10-10"
            
    Returns:
        cases_df (dataframe): returns dataframe for the timeframe, columns, and countries chosen
    """

    # specify path, columns and read the csv file
    path = "data/owid-covid-data.csv"

    if columns == None:
        columns=[]
    df = pd.read_csv(
        path,
        sep=",",
        usecols=["location"] + ["date"] + columns,
        parse_dates=["date"],
        date_parser=lambda col: pd.to_datetime(col, format="%Y-%m-%d"),
    )

    # if no countries specified select continents
    if countries is None:

        # set end date, if none specified pick latest date available
        if end is None:
            end_date = df.date.iloc[-1]
        else:
            end_date = datetime.strptime(end, "%Y-%m-%d")

        countries = df.location.unique()


    # now filter to include only the selected countries
    cases_df = df[df.location.isin(countries)]

    # apply date filters
    if start is not None:
        start_date = datetime.strptime(start, "%Y-%m-%d")
        cases_df = cases_df.loc[cases_df["date"] >= start_date]

    if end is not None:
        end_date = datetime.strptime(end, "%Y-%m-%d")
        if start_date is not None and start_date >= end_date:
            raise ValueError("The start date must be earlier than the end date.")
        cases_df = cases_df.loc[cases_df["date"] <= end_date]

    return cases_df


def get_dataframe():

	columns = ["total_cases", "new_cases", "total_deaths", "new_deaths", "total_cases_per_million", "new_cases_per_million", 
				"total_deaths_per_million", "new_deaths_per_million", "new_tests", "total_tests", "total_tests_per_thousand", 
				"new_tests_per_thousand", "positive_rate", "tests_per_case", "stringency_index", "population_density", "total_vaccinations",
				"people_vaccinated", "people_fully_vaccinated", "total_boosters", "total_vaccinations_per_hundred", 
				"people_vaccinated_per_hundred", "people_fully_vaccinated_per_hundred", "total_boosters_per_hundred", "reproduction_rate"]

	# fetch relevant columns
	df = get_data_from_csv(columns=columns)

	# Find list of countries
	countries = df.location.unique()

	# Group and pre-process by locatiton
	dfs = dict(tuple(df.groupby(df['location'])))

	cnt = 0
	data = pd.DataFrame(columns=columns)
	for i in range(0, len(countries)):
		#print("processing {}".format(countries[i]))

		if countries[i]=="Norway":
			df_part = process_country(dfs[countries[i]])
			df_part = df_part.sort_values(by='date')
			df_part.drop(["location", "date"], axis=1, inplace=True)
			last = df_part.pop("reproduction_rate")
			df_part.insert(24, "reproduction_rate", last)
			df_part.to_csv('data/processed_covid_data_norway.csv', index=False)

		if countries[i]=="Sweden":
			df_part = process_country(dfs[countries[i]])
			df_part = df_part.sort_values(by='date')
			df_part.drop(["location", "date"], axis=1, inplace=True)
			last = df_part.pop("reproduction_rate")
			df_part.insert(24, "reproduction_rate", last)
			df_part.to_csv('data/processed_covid_data_sweden.csv', index=False)

		if countries[i]=="United States":
			df_part = process_country(dfs[countries[i]])
			df_part = df_part.sort_values(by='date')
			df_part.drop(["location", "date"], axis=1, inplace=True)
			last = df_part.pop("reproduction_rate")
			df_part.insert(24, "reproduction_rate", last)
			df_part.to_csv('data/processed_covid_data_usa.csv', index=False)

		if dfs[countries[i]]["reproduction_rate"].isnull().all():
			#print("{} doesn't track reproduction rate".format(countries[i]))
			continue
		else:
			df_part = process_country(dfs[countries[i]])
			data = pd.concat([data, df_part])
			cnt=cnt+1
	

	data = data.sort_values(by='date')

	data.drop(["location", "date"], axis=1, inplace=True)

	data.to_csv('data/processed_covid_data.csv', index=False)

	return data


def process_country(partial_df):

	columns = ["location", "date", "total_cases", "new_cases", "total_deaths", "new_deaths", "total_cases_per_million", "new_cases_per_million", 
				"total_deaths_per_million", "new_deaths_per_million", "new_tests", "total_tests", "total_tests_per_thousand", 
				"new_tests_per_thousand", "positive_rate", "tests_per_case", "stringency_index", "population_density", "total_vaccinations",
				"people_vaccinated", "people_fully_vaccinated", "total_boosters", "total_vaccinations_per_hundred", 
				"people_vaccinated_per_hundred", "people_fully_vaccinated_per_hundred", "total_boosters_per_hundred", "reproduction_rate"]

	partial_df.to_csv('tmp.csv')

	# Drop rows where 'reproduction_rate' is none
	partial_df = partial_df.dropna(subset = ["reproduction_rate"])

	# Find empty columns and populate them with 0s
	for col in columns:
		if partial_df[col].isnull().all():
			partial_df.update(partial_df[col].fillna(value=0, inplace=True))

	# fill 'new' columns with 0s where empty
	partial_df.update(partial_df[["new_cases", 
									"new_deaths", 
									"new_cases_per_million", 
									"new_deaths_per_million", 
									"new_tests", 
									"new_tests_per_thousand",
									]].fillna(value=0, inplace=True))

	# fill leading values with 0s
	for col in columns:
		partial_df.update(partial_df[col].loc[:partial_df[col].first_valid_index()-1].fillna(0))


	# replace remainder with last known value
	partial_df = partial_df.fillna(method="ffill")

	#partial_df.to_csv('tmp.csv')

	return partial_df


if __name__ == "__main__":
    df = get_dataframe()
