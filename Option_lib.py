# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:58:08 2024

@author: Nicola
"""

import numpy as np
import pandas as pd
import yfinance as yf
# from datetime import datetime
from scipy.stats import norm
from copy import copy
from math import pi
import scipy.special
from scipy.special import laguerre


def process_expiration(symbol, exp_td_str):
    """
    Download Yahoo Finance call and put option quotes 
    for a single expiration.
    
    Input:
    exp_td_str = expiration date string "%Y-%m-%d" 
        (a single item from yfinance.Ticker.options tuple)
    
    Return:
    pandas.DataFrame with merged calls and puts data.
    """
    tk = yf.Ticker(symbol)
    options = tk.option_chain(exp_td_str)
    
    calls = options.calls
    puts = options.puts
    
    # Add optionType column
    calls['optionType'] = 'C'
    puts['optionType'] = 'P'
    
    # Convert expiry_date to datetime and add as a new column
    expiry_date = pd.to_datetime(exp_td_str)
    calls['expiry_date'] = expiry_date
    puts['expiry_date'] = expiry_date
    
    # Merge calls and puts into a single dataframe
    exp_data = pd.concat(objs=[calls, puts], ignore_index=True)
    
    # Convert lastTradeDate to datetime
    exp_data['lastTradeDate'] = pd.to_datetime(exp_data['lastTradeDate'])
    
    # Ensure both dates are naive (remove timezone information)
    exp_data['lastTradeDate'] = exp_data['lastTradeDate'].dt.tz_localize(None)
    exp_data['expiry_date'] = exp_data['expiry_date'].dt.tz_localize(None)
    
    # Format lastTradeDate to 'YYYY-MM-DD' (for display purposes only)
    exp_data['lastTradeDate_display'] = exp_data['lastTradeDate'].dt.strftime('%Y-%m-%d')
    
    # Convert lastTradeDate back to datetime for calculations
    exp_data['lastTradeDate'] = pd.to_datetime(exp_data['lastTradeDate_display'])
    
    # Compute the difference between expiry_date and lastTradeDate in days
    exp_data['days_to_expiry'] = (exp_data['expiry_date'] - exp_data['lastTradeDate']).dt.days
    
    # Convert days to time-to-maturity in years (252 trading days in a year)
    exp_data['time_to_maturity'] = exp_data['days_to_expiry'] / 252
    
    return exp_data


def OptionsRetriever(symbol):
    
    # Initialize Ticker object and get all expiration dates
    tk = yf.Ticker(symbol)
    expirations = tk.options
    
    # Create an empty DataFrame, then add individual expiration data to it
    data = pd.DataFrame()

    for exp_td_str in expirations[:1]:  # Process only the first expiration for demonstration
        exp_data = process_expiration(symbol, exp_td_str)
        data = pd.concat(objs=[data, exp_data], ignore_index=True)
        
    # Add underlyingSymbol column
    data['underlyingSymbol'] = symbol
    
    return data
    
    # # Get today's date in the same format as 'lastTradeDate'
    # today = datetime.today().strftime('%Y-%m-%d')
    
    # # Filter the DataFrame to get only rows where 'lastTradeDate' matches today's date
    # today_data = data[data['lastTradeDate_display'] == today]
    
    # return today_data

############################################################

def gaussian_kernel(x):
    return np.sqrt(2*pi) * np.exp(-.5 * x**2)

def BS_call_price(K, T, r, sigma, S0, t=0):
    # The Black-Sholes formula for the preium of a European plain vanilla call option
    # K = strike price;
    # T = option maturity;
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T));
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T));
    
    if d1==0 or d2==0:
        return np.nan
    else:
        return (S0 * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0));
    

### Greeks

def Vega(S0, K, T, mu, sigma):
    
    d1 = (np.log(S0 / K) + (mu + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T));
    
    return S0 * gaussian_kernel(d1) * np.sqrt(T)

    
def implied_vol(K, T, C0, S0, sigma1, sigma2, r, t=0, tol=1e-5, maxiter=100):
        
    # C0 = observed market premium of the call option
    # S0 = observed value of the underlying at time zero
    # r = constant interest rate
    # t = initial time 
    # tol = tollerance
    
    # K = strike;
    # T = maturity;
    
    call_price1 = BS_call_price(K, T, r, sigma1, S0);
    call_price2 = BS_call_price(K, T, r, sigma2, S0);
    
    new_sigma1 = copy(sigma1);
    new_sigma2 = copy(sigma2);
    
    count = 0
    
    while call_price1>C0 and count<20:
        
        if count==20:
            return np.nan
        
        new_sigma1 = new_sigma1/2;
        call_price1 = BS_call_price(K, T, r, new_sigma1, S0);
        
        count += 1
        
    count = 0
    
    while call_price2<C0:
        
        if count==20:
            return np.nan
        
        new_sigma2 = new_sigma2 * 2;
        call_price2 = BS_call_price(K, T, r, new_sigma2, S0);
        
        count += 1
        
    if call_price1 == np.nan or call_price2==np.nan:
        return np.nan
        
    sigma_med = 0.5 * (new_sigma1 + new_sigma2);
    
    new_call_price = BS_call_price(K, T, r, sigma_med, S0);
    
    abs_diff = abs(new_call_price-C0);
    
    count = 0
    while abs_diff>=tol and count<maxiter:
        
        if new_call_price>C0:
            
            new_sigma2 = copy(sigma_med)
            sigma_med = 0.5 * (new_sigma1 + new_sigma2);
        
        elif new_call_price<C0:
            
            new_sigma1 = copy(sigma_med)
            sigma_med = 0.5 * (new_sigma1 + new_sigma2);
            
        new_call_price = BS_call_price(K, T, r, sigma_med, S0);
        
        abs_diff = abs(new_call_price-C0);
        
        count += 1 
    
    return sigma_med


class VanillaOption:
    
    def __init__(self, market_price=np.nan, strike_price=np.nan, maturity_date=None, 
                 time_to_maturity=np.nan, underlying_ticker='', trading_date=None,
                 contract_ticker='', typology='', style='', underlying_price=np.nan,
                 implied_vol=np.nan, rate=np.nan, **kwargs):
        
        default_inputs = {'market_price':market_price,
                          'strike_price':strike_price,
                          'maturity_date':maturity_date,
                          'trading_date' : trading_date,
                          'time_to_maturity':time_to_maturity,
                          'underlying_ticker':underlying_ticker,
                          'contract_ticker':contract_ticker,
                          'typology':typology,
                          'style':style,
                          'underlying_price':underlying_price,
                          'implied_vol':implied_vol,
                          'rate':rate}
        
        for key in kwargs.keys():

            if key in default_inputs.keys():
                default_inputs[key] = kwargs[key]
            else:
                raise Exception(key, 'is not a correct input')
                
        self.strike_price = default_inputs['strike_price']
        self.maturity_date = default_inputs['maturity_date']
        self.market_price = default_inputs['market_price']
        self.trading_date = default_inputs['trading_date']
        self.time_to_maturity = default_inputs['time_to_maturity']
        self.underlying_ticker = default_inputs['underlying_ticker']
        self.contract_ticker = default_inputs['contract_ticker']
        self.typology = default_inputs['typology']
        self.style = default_inputs['style']
        self.underlying_price = default_inputs['underlying_price']
        self.moneyness = self.underlying_price/self.strike_price 
        self.implied_vol = default_inputs['implied_vol']
        self.rate = default_inputs['rate']
        
    
    # def set_maturity(self, maturity_date):
    #     self.maturity_date = maturity_date
        
    # def set_strike(self, strike_price):
    #     self.strike_price = strike_price
    
    def __str__(self):
        
        layout = '{0:>20}{1:>20}\n{2:>20}{3:>20}\n{4:>20}{5:>20}\n{6:>20}{7:>20}\n{8:>20}{9:>20}\n{10:>20}{11:>20}\n{12:>20}{13:>20}\n{14:>20}{15:>20}\n{16:>20}{17:>20}'
        
        return layout.format('Typology :', self.typology,
                             'Style :', self.style,
                             'Price :', str(self.market_price),
                             'Strike price :', str(self.strike_price),
                             'Maturity date :', str(self.maturity_date.year)+'-'+str(self.maturity_date.month)+'-'+str(self.maturity_date.day),
                             'Contract Ticker :', self.contract_ticker,
                             'Underlying Ticker :', self.underlying_ticker,
                             'Underlying price :', str(np.round(self.underlying_price, 2)),
                             'Moneyness S/K:', str(np.round(self.moneyness, 4)))
        

class Time_series_derivatives:
    
    def __init__(self):
        
        self.data = {}
        self.days = []
        self._index = 0
    
    def add_contract(self, Contract, day):
        
        if day in self.days:
            self.data[day].append(Contract)
        else:
            self.data[day] = [Contract]
            self.days.append(day)
    
    # Let's make the class iterable
    
    def __iter__(self):
        self._index = 0  # Reset index for fresh iteration
        return self
    
    def __next__(self):
        if self._index < len(self.days):
            item = self.data[self.days[self._index]]
            self._index += 1 
            return item
        else:
            raise StopIteration
    




################### Monte-Carlo furmulas for option pricing ###################


###### UTILS ###

def OLS(X,Y):
    return np.linalg.inv(X.T@X)@X.T@Y 


def LaguerreBase(degree, x):
    
    result = [] 
    
    for i in range(0, degree+1):
        result.append(laguerre(i)(x))
    
    return np.array(result)

def TaylorPoly(degree, x):
    
    result = 0
    
    for i in range(0, degree+1):
        if i==0 or i==1:
            result += x**i 
        else:
            result += x**i/scipy.special.gamma(i+1)
    
    return result 

def TaylorBase(degree, x):
    
    result = [] 
    
    for i in range(0, degree+1):
        result.append(TaylorPoly(i, x))
    
    return np.array(result)


def EuropeanOption(S_T, K, T, r, payout=1):
    
    # S_T = Underlying price at maturity (T)
    # K = strike price 
    # r = risk-free interest rate 
    # T = Maturity of the Option 
    # payout = option payoff (1 for Call, -1 for Put) 
    
    return np.mean(np.maximum(payout*(S_T - K), 0)) * np.exp(-r*T) 


################ American Options

def select_exercise_prices(St, exercise_dates, n_steps):
    
    # Extracts the asset prices from simulated paths (St) at specified exercise dates.
    # St : np.ndarray simulated asset prices trajectories (n_steps + 1, n_paths).
    # exercise_dates : list or numpy array of float with exercise dates as year fractions
    # n_steps : int, number of time steps used in simulation.
    
    # Convert exercise dates into indices on the time grid
    time_indices = [round(date * n_steps) for date in exercise_dates]

    # Handle out-of-bounds safely
    time_indices = [min(idx, St.shape[0] - 1) for idx in time_indices]

    # Select the corresponding time steps from St
    St_exercise = St[time_indices, :]

    return St_exercise



def AmericanOption(S, K, r, T, exercise_date, payout=-1, degree=3, basis='Laguerre'):
    
    # This function implements the Longstaff-Schwartz algorithm to price 
    # American options
    
    # S = Trajectories of the underlying ate each exercise step
    # K = strike price 
    # r = risk-free interest rate 
    # T = Maturity of the Option 
    # exercise_date = row numpy array or list with the exercise steps from 0 to maturity 
    # payout = option payoff (1 for Call, -1 for Put) 
    # degree = degree of the polynomial bases used for to compute the  
    #          conditional expectation function (F_t measurable) 
    #          for the discounted cash flow received at (t+1)
    # basis = Laguerre or Taylor
    
    if len(exercise_date)!=len(S):
        raise ValueError('length of exercise_date must be equal to length of S')
        
    N = len(exercise_date)
    # dt = T / N
    payoff = np.maximum( payout*(S-K), 0)  # payout = -1 --> Put option
                                           # payout = 1 --> Call option
    # Discounted future cashflows
    cashflow = payoff[-1, :]
    
    for t in range(N-1, 0, -1):
        
        dt = exercise_date[t] - exercise_date[t-1]
        # print(dt)
        ITM = payoff[t, :] > 0 # array where 
        X = S[t, ITM]
        Y = cashflow[ITM] * np.exp(-r * dt)
        
        if basis=='Laguerre':
        # Basis functions: Laguerre polynomials
            A = LaguerreBase(degree, X).T
        elif basis=='Taylor':
            A = TaylorBase(degree, X).T
        else:
            raise ValueError('Wrong basis. Only Laguerre or Taylor')
        
        beta = OLS(A, Y)
        C_t = A @ beta # E[Y|X]
        
        exercise = payoff[t, ITM] > C_t # where E[Y|X] > payoff at time t
        
        cashflow[ITM] = np.where(exercise, payoff[t, ITM], cashflow[ITM] * np.exp(-r * dt))
    
    dt = exercise_date[1] - exercise_date[0]
    
    return np.mean(cashflow * np.exp(-r * dt))


# if __name__=='__main__':
    
#     symbol = 'AAPL'
    
#     # Fetch data for the ticker
#     stock_data = yf.download(symbol, period='10y', group_by='ticker')
    
#     data = OptionsRetriever(symbol).dropna()
    
#     list_of_options = []
    
#     for i in range(1, len(data)):
        
#         try:
#             option = VanillaOption(market_price=data.loc[i,'lastPrice'],
#                                    strike_price=data.loc[i,'strike'],
#                                    contract_ticker=data.loc[i,'contractSymbol'],
#                                    maturity_date=data.loc[i,'expiry_date'],
#                                    time_to_maturity=data.loc[i,'time_to_maturity'],
#                                    underlying_ticker=data.loc[i,'underlyingSymbol'],
#                                    typology=data.loc[i,'optionType'],
#                                    style='E',
#                                    trading_date=data.loc[i,'lastTradeDate'],
#                                    underlying_price=stock_data[symbol].loc[data.loc[i,'lastTradeDate'], 'Close'])
            
#             list_of_options.append(option)
        
#         except:
#             pass
                
    
#     K = 1
#     T = .5
#     price = .1
    
#     option = VanillaOption(strike_price=K)
#     option2 = VanillaOption(T)
