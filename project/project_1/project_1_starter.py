'''
Project 1: Trading with Momentum

Instructions

Each problem consists of a function to implement and instructions on how to implement the function.
The parts of the function that need to be implemented are marked with a # TODO comment.
After implementing the function, run the cell to test it against the unit tests we've provided.
For each problem, we provide one or more unit tests from our project_tests package.
These unit tests won't tell you if your answer is correct, but will warn you of any major errors.
Your code will be checked for the correct solution when you submit it to Udacity.

Packages

When you implement the functions, you'll only need to you use the packages you've used in the classroom,
like Pandas and Numpy. These packages will be imported for you. We recommend you don't add any import statements,
otherwise the grader might not be able to run your code.

The other packages that we're importing are helper, project_helper, and project_tests.
These are custom packages built to help you solve the problems. The helper and project_helper module contains
utility functions and graph functions. The project_tests contains the unit tests for all the problems.'''

'''Install Packages'''

import sys
# !{sys.executable} -m pip install -r requirements.txt

'''Load Packages'''
import pandas as pd
import numpy as np
import helper
import project_helper
import project_tests

'''Market Data

Load Data

The data we use for most of the projects is end of day data. This contains data for many stocks,
but we'll be looking at stocks in the S&P 500. We also made things a little easier to run by narrowing down our
range of time period instead of using all of the data.'''

df = pd.read_csv('../../data/project_1/eod-quotemedia.csv', parse_dates=['date'], index_col=False)

close = df.reset_index().pivot(index='date', columns='ticker', values='adj_close')

print('Loaded Data')

'''View Data

Run the cell below to see what the data looks like for close.'''


project_helper.print_dataframe(close)

'''Stock Example

Let's see what a single stock looks like from the closing prices. For this example and future display examples in this
project, we'll use Apple's stock (AAPL). If we tried to graph all the stocks, it would be too much information.'''

apple_ticker = 'AAPL'
project_helper.plot_stock(close[apple_ticker], '{} Stock'.format(apple_ticker))

'''Resample Adjusted Prices

The trading signal you'll develop in this project does not need to be based on daily prices,
for instance, you can use month-end prices to perform trading once a month. To do this, you must first resample
the daily adjusted closing prices into monthly buckets, and select the last observation of each month.

Implement the resample_prices to resample close_prices at the sampling frequency of freq.'''


def resample_prices(close_prices, freq='M'):
    """
    Resample close prices for each ticker at specified frequency.

    Parameters
    ----------
    close_prices : DataFrame
        Close prices for each ticker and date
    freq : str
        What frequency to sample at
        For valid freq choices, see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    Returns
    -------
    prices_resampled : DataFrame
        Resampled prices for each ticker and date
    """
    # TODO: Implement Function

    return None


project_tests.test_resample_prices(resample_prices)