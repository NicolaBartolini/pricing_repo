# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:54:24 2024

@author: Nicola
"""

import numpy as np
import pandas as pd
from scipy.stats import norm 
from scipy.optimize import minimize, Bounds
from scipy.integrate import quad
from math import pi
from copy import copy

# from Option_lib import implied_vol
from time import time


# Utils

def gaussian_kernel(x):
    return np.sqrt(2*pi) * np.exp(-.5 * x**2)

# @jit(nopython=True)
def exponential_compound_poisson(dt, lam, muj, sigmaj, size):
    
    laplace_transform = np.exp(muj + .5*(sigmaj**2));
    
    compensator = lam * dt * (1 - laplace_transform);
    
    J = np.random.poisson(lam * dt, size );
    
    gaussian_jumps = np.random.normal(0, 1, np.shape(J) ); #generating the Gaussian jumps
    
    jumps = J * muj + sigmaj * np.sqrt(J) * gaussian_jumps;
    
    return np.exp(compensator + jumps);

# @jit(nopython=True)
def gaussian_char_fun(mu, sigma, u):
    
    return np.exp(1j * mu * u - .5*(sigma*u)**2);


def cir_conditional_mean(x0, k, theta, sigma, dt):
    
    return x0 * np.exp(-k * dt) + theta * (1 - np.exp(-k*dt))

def cir_conditional_variance(x0, k, theta, sigma, dt):
    
    return x0 * (sigma**2)/k * (np.exp(-k*dt) - np.exp(-2*k*dt)) + (theta*sigma**2)/(2*k) * (1 - np.exp(-k*dt))**2;


###############################################################################

class FitResult:
    
    def __init__(self, fitted_params, fun):
        
        self.params = fitted_params
        self.fun = fun
        

############### COS functions for pricing

# Definition of cum(c,d) [Equation 22 Fang(2008)]
def CHI(k, a, b, c, d):
    # funzione Chi dell'articolo
    bma = b-a
    uu  = k * np.pi/bma
    cum = np.multiply(np.divide(1, (1 + np.power(uu,2))), (np.cos(uu * (d-a)) * np.exp(d) - np.cos(uu * (c-a)) * np.exp(c) + np.multiply(uu,np.sin(uu * (d-a))) * np.exp(d)-np.multiply(uu,np.sin(uu * (c-a))) * np.exp(c)))
    return cum

# Defintion of Psi (c,d) [Equation 23 Fang(2008)]
def PSI(k, a, b, c, d):
    bma    = b-a
    uu     = k * np.pi/bma
    uu[0]  = 1
    psi    = np.divide(1,uu) * ( np.sin(uu * (d-a)) - np.sin(uu * (c-a)) )
    psi[0] = d-c
    return psi

# Defintion of U_k [Equation 29 Fang(2008)]
def UK(k, a, b, option_type):
    bma = b-a
    if option_type == 'put': 
        Uk  = 2 / bma * (-CHI(k,a,b,a,0) + PSI(k,a,b,a,0) )
    elif option_type == 'call':
        Uk = 2 / bma * (CHI(k,a,b,0,b) - PSI(k,a,b,0,b))
    return Uk



###############################################################################

class GBM:
    
    def __init__(self, mu=np.nan, sigma=np.nan, delta=0, **kwargs):
        
        default_inputs = {'mu':mu,
                          'sigma':sigma,
                          'delta':delta}
        
        for key in kwargs.keys():

            if key in default_inputs.keys():
                default_inputs[key] = kwargs[key]
            else:
                raise Exception(key, 'is not a correct input')
                
        self.mu = default_inputs['mu'] # drift
        self.sigma = default_inputs['sigma'] # vol 
        self.delta = default_inputs['delta'] # continuous dividend yield
        
    def set_drif(self, mu):
        self.mu = mu
    
    def set_sigma(self, sigma):
        self.sigma = sigma
    
    def set_delta(self, delta):
        self.delta = delta
    
    def simulate(self, S0, T, n_steps, N=0):
        
        n = int(2**N)
        
        dt = T/n_steps
        
        St = np.empty((n_steps, n))
        St[0] = S0
        
        for i in range(1, n_steps):
            
            if n==1:
                noise = np.random.normal(0, 1, 1)
            else:
                noise = np.random.normal(0, 1, (1,int(n/2)))
                noise = np.hstack((noise, -noise))
            
            St[i] = St[i-1] * np.exp((self.mu -.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt) * noise)
        
        return St
    
    def char_fun(self, T, u):
        return np.exp(1j * u * (self.mu - 0.5 * self.sigma**2) * T - 0.5 * (u * self.sigma)**2 * T)

    def damping_integrand(self, S0, K, T, u, alpha=1.5):
        return (self.char_fun(T, -u - alpha * 1j) / ((1j * u - alpha) * (1j * u - alpha + 1)) * np.exp(-1j * u * np.log(S0 / K))).real

    def Fourier_call_pricing(self, S0, K, T, alpha=1.5):
        int_value = quad(lambda u: self.damping_integrand(S0, K, T, u, alpha), 0, np.inf, limit=200)[0]
        call_value = ((np.exp(-self.mu * T) * S0**alpha * K**(1 - alpha)) / pi) * int_value
        return call_value
    
    def Fourier_put_pricing(self, S0, K, T, alpha=1.5):
        
        return self.Fourier_call_pricing(S0, K, T, alpha=1.5) - S0 + K*np.exp(-self.mu*(T))
    
    def mc_call_pricing(self, S0,  n_steps, K, T, N=0):
        
        trj = self.simulate(S0, T, n_steps, N); # Generating the trajectories of the asset
        call_payoff = trj[-1] - K;
        call_payoff[call_payoff<0] = 0;
        call_payoff = call_payoff*np.exp(-self.mu * T)
        
        return np.mean(call_payoff);
    
    def mc_put_pricing(self, n_steps, K, T, X0, nu0, t=0, N=0):
        
        return self.mc_call_pricing(X0,  n_steps, K, T, N) - X0 + K*np.exp(-self.mu*(T-t))


    def BS_call_price(self, S0, K, T, t=0):
        # The Black-Sholes formula for the preium of a European plain vanilla call option
        # K = strike price;
        # T = option maturity;
        
        d1 = (np.log(S0 / K) + (self.mu - self.delta + 0.5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(T));
        d2 = (np.log(S0 / K) + (self.mu -self.delta - 0.5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(T));
        
        # try:
        #     if d1==0 or d2==0:
        #         return np.nan
        #     else:
        #         S0 *np.exp(-self.delta * (T-t)) * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-self.mu * T) * norm.cdf(d2, 0.0, 1.0)
        # except:
        #     return S0 *np.exp(-self.delta * (T-t)) * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-self.mu * T) * norm.cdf(d2, 0.0, 1.0);
        
        return S0 *np.exp(-self.delta * (T-t)) * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-self.mu * T) * norm.cdf(d2, 0.0, 1.0);
    
    def cos_pricing(self, S0, K, T, typology, N=32):
        
        cum1 = (self.mu - 0.5*self.sigma**2)*T
        cum2 = self.sigma**2 * T
        cum4 = 0
        
        L = 10
        a = cum1 - L * (np.sqrt(cum2 + np.sqrt(cum4)))
        b = cum1 + L * (np.sqrt(cum2 + np.sqrt(cum4)))
        
        bma = b-a
        k  = np.arange(N+1)
        u  = k * np.pi/(b-a)
        # V_COS = np.zeros((np.size(K))) 
        
        CF = self.char_fun(T, u) # computing the characteristic function 
        
        x  = np.log(S0/K)
        Term = np.exp(1j * k * np.pi * (x-a)/bma)
        Fk = np.real(np.multiply(CF, Term))
        Fk[0] = 0.5 * Fk[0] 
        
        r = self.mu
        
        V_COS = K * np.sum(np.multiply(Fk,UK(k, a, b, typology))) * np.exp(-r*T)
        
        return V_COS
         
    
    
    def vega(self, S0, K, T, t=0):
        
        d1 = (np.log(S0 / K) + (self.mu - self.delta + 0.5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(T));
        
        return S0 * np.exp(-self.delta * (T-t)) * gaussian_kernel(d1) * np.sqrt(T-t)
    
    def BS_put_price(self, S0, K, T, t=0):
        # The Black-Sholes formula for the preium of a European plain vanilla call option
        # K = strike price;
        # T = option maturity;
        
        d1 = (np.log(S0 / K) + (self.mu - self.delta + 0.5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(T));
        d2 = (np.log(S0 / K) + (self.mu -self.delta - 0.5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(T));
        
        return K * np.exp(-self.mu * T) * norm.cdf(-d2, 0.0, 1.0)  - S0 *np.exp(-self.delta * (T-t)) * norm.cdf(-d1, 0.0, 1.0)
        
        # return self.BS_call_price(S0, K, T, t) - S0 + K*np.exp(-self.mu*(T-t)) + S0 * (1 - np.exp(-self.delta*(T-t)))
    
    def obj_fun(self, params, data):
        
        sigma = params[0]
        
        self.set_sigma(sigma)
        
        model_prices = []
        mkt_prices = []
                
        for item in data:
            for option in item:
            
                S = option.underlying_price
                K = option.strike_price
                T = option.time_to_maturity
                option_type = option.typology
                
                if option_type=='C':
                    model_prices.append(self.BS_call_price(S, K, T))
                    mkt_prices.append(option.market_price)
                elif option_type=='P':
                    model_prices.append(self.BS_put_price(S, K, T))
                    mkt_prices.append(option.market_price)
        
        mkt_prices = np.array(mkt_prices)
        model_prices = np.array(model_prices)
        
        return np.sum((mkt_prices - model_prices)**2)
        
    
    def calibrate(self, params, data, method='L-BFGS-B'):
        
        bounds = Bounds(0.0001, 1000)
        
        res = minimize(self.obj_fun, params, bounds=bounds, method=method, 
                       args=(data), jac='2-points')
        
        index = ['sigma']
        columns = ['params','gradient']
        
        df = pd.DataFrame(np.vstack((res.x, res.jac)).T, index=index, columns=columns)
        
        self.set_sigma(res.x[0])
        
        return FitResult(df, res.fun)
        


class Merton:
    
    def __init__(self, mu=np.nan, sigma=np.nan, lambda_jump=np.nan, 
                 mu_jump=0, sigma_jump=np.nan, **kwargs):
        
        default_inputs = {'mu':mu,
                          'sigma':sigma,
                          'lambda_jump':lambda_jump,
                          'mu_jump':mu_jump,
                          'sigma_jump':sigma_jump}
        
        for key in kwargs.keys():

            if key in default_inputs.keys():
                default_inputs[key] = kwargs[key]
            else:
                raise Exception(key, 'is not a correct input')
                
        self.mu = default_inputs['mu']
        self.sigma_jump = default_inputs['sigma_jump']
        self.mu_jump = default_inputs['mu_jump']
        self.lambda_jump = default_inputs['lambda_jump']
        self.sigma = default_inputs['sigma']
        
    def set_drif(self, mu):
        self.mu = mu
    
    def set_sigma(self, sigma):
        self.sigma = sigma
    
    def set_lambda(self, lambda_jump):
        self.lambda_jump = lambda_jump
        
    def set_sigma_jump(self, sigma_jump):
        self.sigma_jump = sigma_jump
    
    def simulate(self, S0, T, n_steps, N=0):
        
        n = int(2**N)
        
        dt = T/n_steps
        
        St = np.empty((n_steps, n))
        St[0] = S0
        
        compensator = -self.lambda_jump * dt * (np.exp(self.mu_jump+.5*self.sigma_jump**2) - 1)
        
        for i in range(1, n_steps):
            
            if n==1:
                noise = np.random.normal(0, 1, 1)
                J = np.random.normal(0, 1, 1)
            else:
                noise = np.random.normal(0, 1, (1,int(n/2)))
                noise = np.hstack((noise, -noise))
                J = np.random.normal(0, 1, (1,int(n/2)))
                J = np.hstack((J,-J))
            
            n_jumps = np.random.poisson(self.lambda_jump*dt, n)
            jump_noise = n_jumps * self.mu_jump + np.sqrt(n_jumps) * self.sigma_jump * J
            
            St[i] = St[i-1] * np.exp((self.mu -.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt) * noise + compensator + jump_noise)
        
        return St
    
    def mc_call_pricing(self, S0,  n_steps, K, T, N=0):
        
        trj = self.simulate(S0, T, n_steps, N); # Generating the trajectories of the asset
        call_payoff = trj[-1] - K;
        call_payoff[call_payoff<0] = 0;
        call_payoff = call_payoff*np.exp(-self.mu * T)
        
        return np.mean(call_payoff);
    
    def mc_put_pricing(self, n_steps, K, T, X0, nu0, t=0, N=0):
        
        return self.mc_call_pricing(X0,  n_steps, K, T, N) - X0 + K*np.exp(-self.mu*(T-t))
    
        

class Heston93:
    
    def __init__(self, mu=np.nan, k=np.nan, theta=np.nan, sigma=np.nan, rho=0, **kwargs):
        
        default_inputs = {'mu':mu,
                          'k':k,
                          'theta':theta,
                          'sigma':sigma,
                          'rho':rho}
        
        for key in kwargs.keys():

            if key in default_inputs.keys():
                default_inputs[key] = kwargs[key]
            else:
                raise Exception(key, 'is not a correct input')
                
        self.mu = default_inputs['mu']
        self.k = default_inputs['k']
        self.theta = default_inputs['theta']
        self.sigma = default_inputs['sigma']
        self.rho = default_inputs['rho']
    
    def set_mu(self, mu):
        self.mu = mu
    
    def set_kappa(self, k):
        self.k = k
        
    def set_theta(self, theta):
        self.theta = theta 
        
    def set_sigma(self, sigma):
        self.sigma = sigma
    
    def set_rho(self, rho):
        self.rho = rho
    
    def simulate(self, X0, nu0, T, n_steps, N, gamma1=.5, gamma2=.5, method='euler'):
        
        mu = self.mu
        k = self.k 
        theta = self.theta 
        sigma = self.sigma 
        rho = self.rho 
        
        if method=='euler':
            
            n = 2**N;
            dt = T/n_steps;
            
            X_trj = np.empty((n_steps, n));
            vol_trj = np.empty((n_steps, n));
            
            X_trj[0] = X0;
            vol_trj[0] = nu0;
            
            if n==0:
                X_noise = np.random.normal(0,1, (n_steps-1, 1));
                vol_noise = np.random.normal(0,1, (n_steps-1, 1));
            else:
                vol_noise = np.random.normal(0,1, (n_steps-1, int(n/2)));
                vol_noise = np.hstack((vol_noise,-vol_noise))
                
                X_noise = np.random.normal(0,1, (n_steps-1, int(n/2)));
                X_noise = np.hstack((X_noise,-X_noise))
            
            for i in np.arange(1,n_steps):
                
                nu = vol_trj[i-1] + k * (theta - vol_trj[i-1]) * dt + sigma * np.sqrt(vol_trj[i-1] * dt) * vol_noise[i-1];
                nu[nu<0] = 0;
                vol_trj[i] = nu;
                
                X_trj[i] = X_trj[i-1] * np.exp( (rho/sigma)*(vol_trj[i]-vol_trj[i-1] -k*theta*dt) + (rho*k/sigma -0.5)*(vol_trj[i-1] * dt) + 
                                               np.sqrt((1-rho**2) * dt * vol_trj[i-1]) * X_noise[i-1]);
                
            return (X_trj * np.exp(mu*T), vol_trj);
        
        elif method=='andersen':
            
            n = 2**N;
            dt = T/n_steps;
            
            X_trj = np.empty((n_steps, n)); # array containing the simulations of the asset. Each column is a Monte-Carlo simulation
            vol_trj = np.empty((n_steps, n)); # array containing the simulations of the volatility. Each column is a Monte-Carlo simulation
            
            X_trj[0] = X0;
            vol_trj[0] = nu0;
            
            noise = np.random.uniform(0,1,(n_steps-1, 2*n)); # generating the noise from a uniform distribution 
            
            uniform_sampling = noise[:,0:n];
            
            # vol_noise = noise[:,0:n]; # taking the first uniform observations for generating the noise for the asset X_trj
            X_noise = noise[:,n:]; # taking the first uniform observations for generating the noise for the volatility vol_trj
            
            for i in np.arange(1,n_steps): # iterating for each step 
                
                m = cir_conditional_mean(vol_trj[i-1], k, theta, sigma, dt);
                s_square = cir_conditional_variance(vol_trj[i-1], k, theta, sigma, dt);
               
                psi = s_square/(m**2);
               
                psi_normal = copy(psi[psi<=1.5]);
                m_normal = m[psi<=1.5];
                
                b_square = 2/psi_normal - 1 + np.sqrt(2/psi_normal) * np.sqrt(2/psi_normal - 1); 
                a = m_normal/(1+b_square);
                gaussian_noise = norm.ppf(uniform_sampling[i-1]);
                noise = gaussian_noise[psi<=1.5];
                V_normal = a * (np.sqrt(b_square) + noise)**2;
               
                vol_trj[i][psi<=1.5] = V_normal;
               
                psi_u = copy(psi[psi>1.5]);
                m_u = m[psi>1.5];
                # s_u = s_square[psi>1.5];
               
                p = (psi_u-1)/(psi_u+1);
                beta = (1-p)/m_u;
               
                u = uniform_sampling[i-1];
                unif_noise = u[psi>1.5];
               
                V_unif = 1/beta * np.log((1-p)/(1-unif_noise));
                V_unif[V_unif<0] = 0;
               
                vol_trj[i][psi_u>1.5] = V_unif;
                
                # generating the asset at the i-th step
                K0 = -dt * (rho * k * theta)/sigma; 
                K1 = gamma1 * dt * (k*rho/sigma -0.5) - rho/sigma;
                K2 = gamma2 * dt * (k*rho/sigma -0.5) + rho/sigma;
                K3 = gamma1 * dt * (1 - rho**2);
                K4 = gamma2 * dt * (1 - rho**2);
                
                X_trj[i] = X_trj[i-1] * np.exp(mu*dt + K0 + K1*vol_trj[i-1] + K2*vol_trj[i] +np.sqrt(K3*vol_trj[i-1] + K4*vol_trj[i-1]) * norm.ppf(X_noise[i-1]));
                
            return (X_trj, vol_trj);
            
        else:
            pass
        
    def char_fun(self, u, T, t=0, X0=1, nu0=0):
      ''' Valuation of European call option in H93 (2001)
      Fourier-based approach: characteristic function.
      Parameter definitions see function BCC_call_value.'''
      
      mu = self.mu
      k = self.k 
      theta = self.theta 
      sigma = self.sigma 
      rho = self.rho 
      
      # Funzione presa dal libro di Pascucci
      r = mu;

      D = np.sqrt((k - 1j * rho * sigma * u)**2 + (u + 1j)*u*sigma**2);
      G = (k - 1j * rho * sigma * u - D) / (k - 1j * rho * sigma * u + D);

      A = 1j * u * r * (T-t) + nu0/sigma**2 * (1 - np.exp(-D * (T-t)))/(1 - G * np.exp(-D * (T-t))) * (k - 1j * rho * sigma * u - D);
      # A = 1j * u * r * (T-t) + nu0/sigma**2 * (1 - np.exp(-D * (T-t)))/(2 - G * np.exp(-D * (T-t))) * (k - 1j * rho * sigma * u - D);
      B = k*theta/sigma**2 * ((T-t) * (k - 1j * rho * sigma * u - D) - 2*np.log((1 - G * np.exp(-D * (T-t)))/(1-G)) );

      return np.exp(A + B); 
  
    def Damping_integrand(self, K, T, u, alpha, t=0, X0=1, nu0=0):
    		# This formula is the formula of the damping method for Fourier way for European call option pricing
        
        # mu = process.mu;
        # K = strike price;
        # T = option maturity;
        
        return (self.char_fun(-u - alpha*1j, T, t, 1, nu0) / ((1j*u - alpha) * (1j*u - alpha + 1)) * np.exp(-1j * u * np.log(X0/K))).real;
    
    def Fourier_call_pricing(self, K, T, X0, nu0, alpha=1.5, t=0, a=0, b=50, n_inter=100):
        
        # K = self.strike;
        # T = self.maturity;
        
        r = self.mu; # the risk free (drift of the process)
        
        int_value = quad(lambda u: self.Damping_integrand(K, T, u, alpha, t, X0, nu0), a, b, limit=n_inter)[0];

        call_value = ((np.exp(-r * (T-t)) * X0**alpha * K**(1-alpha)) / pi) * int_value;

        return call_value;
    
    def Fourier_put_pricing(self, K, T, X0, nu0, alpha=1.5, t=0, a=0, b=50, n_inter=100):
        
        return self.Fourier_call_pricing(K, T, X0, nu0, alpha, t, a, b, n_inter) - X0 + K*np.exp(-self.mu*(T-t))
    
    def mc_call_pricing(self, n_steps, K, T, X0, nu0, t=0, N=0):
        
        r = self.mu
        # k = self.k 
        # theta = self.theta 
        # sigma = self.sigma 
        # rho = self.rho 
        
        trj = self.simulate(X0, nu0, T, n_steps, N)[0]; # Generating the trajectories of the asset
        call_payoff = trj[-1] - K;
        call_payoff[call_payoff<0] = 0;
        call_payoff = call_payoff*np.exp(-r*T)
        
        return np.mean(call_payoff);
    
    def mc_put_pricing(self, n_steps, K, T, X0, nu0, t=0, N=0):
        
        return self.mc_call_pricing(n_steps, K, T, X0, nu0, t, N) - X0 + K*np.exp(-self.mu*(T-t))
    
    def cos_pricing(self, S0, nu0, K, T, typology, N=32):
        
        cum1 = (self.mu - 0.5*self.sigma**2)*T
        cum2 = self.sigma**2 * T
        cum4 = 0
        
        L = 10
        a = cum1 - L * (np.sqrt(cum2 + np.sqrt(cum4)))
        b = cum1 + L * (np.sqrt(cum2 + np.sqrt(cum4)))
        
        bma = b-a
        k  = np.arange(N+1)
        u  = k * np.pi/(b-a)
        # V_COS = np.zeros((np.size(K))) 
        
        CF = self.char_fun(u, T, 0, 1, nu0) # computing the characteristic function 
        
        x  = np.log(S0/K)
        Term = np.exp(1j * k * np.pi * (x-a)/bma)
        Fk = np.real(np.multiply(CF, Term))
        Fk[0] = 0.5 * Fk[0] 
        
        r = self.mu
        
        V_COS = K * np.sum(np.multiply(Fk,UK(k, a, b, typology))) * np.exp(-r*T)
        
        return V_COS
    
    def obj_fun(self, params, data, alpha=1.5):
        
        k = params[0]
        theta = params[1]
        sigma = params[2]
        rho = params[3]
        nu0 = params[4]
        
        self.set_kappa(k)
        self.set_theta(theta)
        self.set_rho(rho)
        self.set_sigma(sigma)
        
        model_prices = []
        mkt_prices = []
        
        for item in data:
            for option in item:
            
                X0 = option.underlying_price
                K = option.strike_price
                T = option.time_to_maturity
                option_type = option.typology
                
                if option_type=='C':
                    
                    model_prices.append(self.Fourier_call_pricing(K, T, X0, nu0, alpha, 0))
                    mkt_prices.append(option.market_price)
                    
                elif option_type=='P':
                    
                    model_prices.append(self.Fourier_put_pricing(K, T, X0, nu0, alpha, 0))
                    mkt_prices.append(option.market_price)
                
        
        model_prices = np.array(model_prices)
        mkt_prices = np.array([mkt_prices])
        
        result = np.sum((mkt_prices - model_prices)**2)
        
        return result
    
    def calibrate(self, params, data, method='L-BFGS-B'):
        
        bounds = Bounds([0, 0, 0, -1, 0], [np.inf, np.inf, np.inf, 1, np.inf])
        
        res = minimize(self.obj_fun, params, bounds=bounds, method=method, 
                       args=(data), jac='2-points')
        
        index = ['k', 'theta', 'sigma', 'rho', 'nu0']
        columns = ['params','gradient']
        
        df = pd.DataFrame(np.vstack((res.x, res.jac)).T, index=index, columns=columns)
        
        k = res.x[0]
        theta = res.x[1]
        sigma = res.x[2]
        rho = res.x[3]
        nu0 = res.x[4]
        
        self.set_kappa(k)
        self.set_theta(theta)
        self.set_rho(rho)
        self.set_sigma(sigma)
        
        return FitResult(df, res.fun)

class Bates:
    
    def __init__(self, mu=np.nan, k=np.nan, theta=np.nan, sigma=np.nan, rho=0, 
                 lam=np.nan, muj=np.nan, sigmaj=np.nan, **kwargs):
        
        default_inputs = {'mu':mu,
                          'k':k,
                          'theta':theta,
                          'sigma':sigma,
                          'rho':rho,
                          'lam':lam,
                          'muj':muj,
                          'sigmaj':sigmaj}
        
        for key in kwargs.keys():

            if key in default_inputs.keys():
                default_inputs[key] = kwargs[key]
            else:
                raise Exception(key, 'is not a correct input')
                
        self.mu = default_inputs['mu']
        self.k = default_inputs['k']
        self.theta = default_inputs['theta']
        self.sigma = default_inputs['sigma']
        self.rho = default_inputs['rho']
        self.lam = default_inputs['lam']
        self.muj = default_inputs['muj']
        self.sigmaj = default_inputs['sigmaj']
    
    def set_mu(self, mu):
        self.mu = mu
    
    def set_kappa(self, k):
        self.k = k
        
    def set_theta(self, theta):
        self.theta = theta 
        
    def set_sigma(self, sigma):
        self.sigma = sigma
    
    def set_rho(self, rho):
        self.rho = rho
    
    def set_muj(self, muj):
        self.muj = muj 
    
    def set_sigmaj(self, sigmaj):
        self.sigmaj = sigmaj
    
    def set_lambda(self, lam):
        self.lam = lam
    
    def simulate(self, X0, nu0, n_steps, T, N=0):
        
        mu = self.mu
        k = self.k
        theta = self.theta
        sigma = self.sigma
        rho = self.rho
        lam = self.lam
        muj = self.muj
        sigmaj = self.sigmaj
        
        # The simulation has been performed with the Euler scheme for both the dynamics 
        dt = float(T/n_steps); # time steps width of the MC simulations
        n = int(2**N); # total number of simulations
        
        S = np.nan;
        vol = np.nan;
        
        if n==1:
            S = np.empty((n_steps, 1));	# where will be stored the trajectory of the principal process 
            S[0] = X0;
            vol = np.empty((n_steps, 1));	# where will be stored the trajectory of the volatility process
            vol[0] = nu0; 
        else:
            S = np.empty((n_steps, n));	# where will be stored the trajectory of the principal process 
            S[0] = X0;
            
            vol = np.empty((n_steps, n));	# where will be stored the trajectory of the volatility process
            vol[0] = nu0;
        
        for j in np.arange(1, n_steps):
            
            if n!=1:
            
                vola_noise = np.random.normal(0,1, int(n/2)); #generating n-random noises of the volatility
                BM_vola = np.empty(n);  #array of the total noise
                # Total noises via antithetic variate
                BM_vola[0:int(n/2)] = vola_noise; 
                BM_vola[int(n/2):] = -vola_noise; #symmetric noise 
            
                nu = vol[j-1] + (k * (theta - vol[j-1]) * dt + sigma * np.sqrt(vol[j-1]*dt) * BM_vola);	# generating the vol
                nu[nu<0] = 0;
      
                vol[j,:] = nu;
            
                stock_noise = np.random.normal(0,1, int(n/2)); #generating n-random noises of the volatility
                BM_S = np.empty(n);  #array of the total noise
                # Total noises via antithetic variate
                BM_S[0:int(n/2)] = stock_noise; 
                BM_S[int(n/2):] = -stock_noise; #symmetric noise 
            
                S[j] = (S[j-1] * np.exp((mu - vol[j-1]/2) * dt + np.sqrt(vol[j-1] * dt) * 
                                        (rho * BM_vola + np.sqrt(1 - rho**2) * BM_S )) * 
                        exponential_compound_poisson(dt, lam, muj, sigmaj, n) )  ;	# generating the principal process
            
            else:
                
                BM_vola = np.random.normal(0,1, 1); #generating n-random noises of the volatility
            
                nu = (vol[j-1] + k * (theta - vol[j-1]) * dt + sigma * np.sqrt(vol[j-1]*dt) * BM_vola);	# generating the vol
                nu[nu<0] = 0;
      
                vol[j,:] = nu;
            
                BM_S = np.random.normal(0,1, 1); #generating n-random noises of the volatility
                
                S[j] = (S[j-1] * np.exp((mu - vol[j-1]/2) * dt + np.sqrt(vol[j-1] * dt) * 
                                        (rho * BM_vola + np.sqrt(1 - rho**2) * BM_S )) * 
                        exponential_compound_poisson(dt, lam, muj, sigmaj, 1) )  ;	# generating the principal process
        
        return (S,vol);
    
    def char_fun(self, u, T, t, X0, nu0):
        
        #This is the characteristic function of the Bates model and it is the formula from the book of Pascucci 
        
        # k; # The speed of mean reversion of the volatility dynamic
        # theta; # the long run memory of the volatility dynamic
        # sigma; # the diffusion parameter of the volatility dynamic
        # #stock
        # r = mu; # the drift of the stock dynamic
        # #correlation
        # rho; # the correlation of the two Brownian motions
        # lam; # intensity of the  Poisson
        # muj; # mean of the jumps normally distributed
        # sigmaj = self.sigmaj; # standard deviation of the jumps normally distributed
        
        mu = self.mu
        k = self.k
        theta = self.theta
        sigma = self.sigma
        rho = self.rho
        lam = self.lam
        muj = self.muj
        sigmaj = self.sigmaj
        
        r = mu;

        D = np.sqrt((k - 1j * rho * sigma * u)**2 + (u + 1j)*u*sigma**2);
        G = (k - 1j * rho * sigma * u - D) / (k - 1j * rho * sigma * u + D);

        A = 1j * u * r * (T-t) + nu0/sigma**2 * (1 - np.exp(-D * (T-t)))/(1 - G * np.exp(-D * (T-t))) * (k - 1j * rho * sigma * u - D);
        B = k*theta/sigma**2 * ((T-t) * (k - 1j * rho * sigma * u - D) - 2*np.log((1 - G * np.exp(-D * (T-t)))/(1-G)) );
        
        # now the transform of the jump part 
        
        laplace_transform = np.exp(muj + .5*(sigmaj**2)); #laplace transform of the jump width 
        
        compensator = lam * (T-t) * (1 - laplace_transform); # the compensator 
        
        exponent_char_function = lam * (T-t) * (gaussian_char_fun(muj, sigmaj, u) - 1);
        
        return np.exp(A + B + exponent_char_function + 1j * u * compensator);
    
    def Damping_integrand(self, K, T, u, alpha, t, X0, nu0):
    		# This formula is the formula of the damping method for Fourier way for European call option pricing
        
        # mu = process.mu;
        # K = strike price;
        # T = option maturity;
        
        return (self.char_fun(-u - alpha*1j, T, t, 1, nu0) / ((1j*u - alpha) * (1j*u - alpha + 1)) * np.exp(-1j * u * np.log(X0/K))).real;
    
    def Fourier_call_pricing(self, K, T, X0, nu0, alpha=1.5, t=0, a=0, b=50, n_inter=100):
        
        # K = self.strike;
        # T = self.maturity;
        
        r = self.mu; # the risk free (drift of the process) 
        
        int_value = quad(lambda u: self.Damping_integrand(K, T, u, alpha, t, X0, nu0), a, b, limit=n_inter)[0];

        call_value = ((np.exp(-r * (T-t)) * X0**alpha * K**(1-alpha)) / pi) * int_value;

        return call_value;
    
    def Fourier_put_pricing(self, K, T, X0, nu0, alpha=1.5, t=0, a=0, b=50, n_inter=100):
        
        return self.Fourier_call_pricing(K, T, X0, nu0, alpha, t, a, b, n_inter) - X0 + K*np.exp(-self.mu*(T-t))
    
    
    def mc_call_pricing(self, X0, nu0, T, K, n_steps, N=1):
        
        trj = self.simulate(X0, nu0, n_steps, T, N)[0]; # Generating the trajectories of the asset
        call_payoff = trj[-1] - K;
        call_payoff[call_payoff<0] = 0;
        call_payoff = call_payoff*np.exp(-self.mu*T)
        
        return np.mean(call_payoff);
    
    def mc_put_pricing(self, n_steps, K, T, X0, nu0, t=0, N=0):
        
        return self.mc_call_pricing(n_steps, K, T, X0, nu0, t, N) - X0 + K*np.exp(-self.mu*(T-t))
    
    def cos_pricing(self, S0, nu0, K, T, typology, N=32):
        
        cum1 = (self.mu - 0.5*self.sigma**2)*T
        cum2 = self.sigma**2 * T
        cum4 = 0
        
        L = 10
        a = cum1 - L * (np.sqrt(cum2 + np.sqrt(cum4)))
        b = cum1 + L * (np.sqrt(cum2 + np.sqrt(cum4)))
        
        bma = b-a
        k  = np.arange(N+1)
        u  = k * np.pi/(b-a)
        # V_COS = np.zeros((np.size(K))) 
        
        CF = self.char_fun(u, T, 0, 1, nu0) # computing the characteristic function 
        
        x  = np.log(S0/K)
        Term = np.exp(1j * k * np.pi * (x-a)/bma)
        Fk = np.real(np.multiply(CF, Term))
        Fk[0] = 0.5 * Fk[0] 
        
        r = self.mu
        
        V_COS = K * np.sum(np.multiply(Fk,UK(k, a, b, typology))) * np.exp(-r*T)
        
        return V_COS
    
    def obj_fun_price(self, params, data, alpha=1.5):
        
        k = params[0]
        theta = params[1]
        sigma = params[2]
        rho = params[3]
        lam = params[4]
        sigmaj = params[5]
        nu0 = params[6]
        
        self.set_kappa(k)
        self.set_theta(theta)
        self.set_rho(rho)
        self.set_sigma(sigma)
        self.set_lambda(lam)
        self.set_sigmaj(sigmaj)
        
        model_prices = []
        mkt_prices = []
        
        for item in data:
            for option in item:
            
                X0 = option.underlying_price
                K = option.strike_price
                T = option.time_to_maturity
                option_type = option.typology
                
                if option_type=='C':
                    model_prices.append(self.Fourier_call_pricing(K, T, X0, nu0, alpha, 0))
                    mkt_prices.append(option.market_price)
                elif option_type=='P':
                    model_prices.append(self.Fourier_put_pricing(K, T, X0, nu0, alpha, 0))
                    mkt_prices.append(option.market_price)
        
        model_prices = np.array(model_prices)
        mkt_prices = np.array(mkt_prices)
        
        result = np.sum((mkt_prices - model_prices)**2)
        # print(result)
        return result
    
    
    def obj_fun_implied_vol(self, params, data, alpha=1.5):
        
        k = params[0]
        theta = params[1]
        sigma = params[2]
        rho = params[3]
        lam = params[4]
        sigmaj = params[5]
        nu0 = params[6]
        
        self.set_kappa(k)
        self.set_theta(theta)
        self.set_rho(rho)
        self.set_sigma(sigma)
        self.set_lambda(lam)
        self.set_sigmaj(sigmaj)
        
        model_implied_vol = []
        mkt_implied_vol = []
        
        for item in data:
            for option in item:
            
                X0 = option.underlying_price
                K = option.strike_price
                T = option.time_to_maturity
                option_type = option.typology
                v = option.implied_vol
                
                if option_type=='C':
                    
                    model_price = self.Fourier_call_pricing(K, T, X0, nu0, alpha, 0)
                    
                    mkt_implied_vol = v
                    
                    model_implied_vol.append(implied_vol(K, T, model_price, X0, .0001, 1, self.mu))
                
                elif option_type=='P':
                    
                    model_price = self.Fourier_put_pricing(K, T, X0, nu0, alpha, 0)
                    
                    mkt_implied_vol = v
                    
                    model_implied_vol.append(implied_vol(K, T, model_price, X0, .0001, 1, self.mu))
        
        
        model_implied_vol = np.array(model_implied_vol)
        mkt_implied_vol = np.array(mkt_implied_vol)
        
        result = np.sum((mkt_implied_vol - model_implied_vol)**2)
        
        return result
    
    
    def calibrate(self, params, data, method='L-BFGS-B'):
        
        bounds = Bounds([0, 0, 0, -1, 0, 0, 0], [np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf])
        
        
        res = minimize(self.obj_fun_price, params, bounds=bounds, method=method, 
                       args=(data), jac='2-points')
        
        index = ['k', 'theta', 'sigma', 'rho', 'lambda', 'sigmaj', 'nu0']
        columns = ['params','gradient']
        
        df = pd.DataFrame(np.vstack((res.x, res.jac)).T, index=index, columns=columns)
        
        k = res.x[0]
        theta = res.x[1]
        sigma = res.x[2]
        rho = res.x[3]
        lam = res.x[4]
        sigmaj = res.x[5]
        nu0 = res.x[6]
        
        self.set_kappa(k)
        self.set_theta(theta)
        self.set_rho(rho)
        self.set_sigma(sigma)
        self.set_lambda(lam)
        self.set_sigmaj(sigmaj)
        
        return FitResult(df, res.fun)

class VarianceGamma:
    
    def __init__(self, r=np.nan, theta=np.nan, sigma=np.nan, nu=np.nan, **kwargs):
        
        default_inputs = {'r': r,
                          'sigma':sigma,
                          'nu':nu,
                          'theta':theta}
        
        for key in kwargs.keys():

            if key in default_inputs.keys():
                default_inputs[key] = kwargs[key]
            else:
                raise Exception(key, 'is not a correct input')
        
        self.theta = default_inputs['theta']
        self.r = default_inputs['r']
        self.sigma = default_inputs['sigma']
        self.nu = default_inputs['nu']
        
    
    def simulate(self, S0, n_steps, T, N=15):
        
        n = int(2**N)
        dt = T/n_steps
        
        compensator = dt/self.nu * np.log(1 - self.nu*self.theta -.5 * self.nu*self.sigma**2)
        
        Xt = np.empty((n_steps, n))
        Xt[0, :] = np.log(S0)
        
        for t in range(1, n_steps):
            
            gamma_time = np.random.gamma(dt/self.nu, self.nu, n)
            eps = np.random.normal(0, 1, n)
            Xt[t, :] = Xt[t-1, :] + self.r*dt + self.theta* gamma_time + self.sigma * np.sqrt(gamma_time) * eps + compensator 
        
        return np.exp(Xt)
         
    
    def char_fun(self, u, T):
        
        compensator = T/self.nu * np.log(1 - self.nu*self.theta -.5 * self.nu*self.sigma**2)
        A = 1j * u * self.r * T * compensator
                
        return np.exp(A) * (1 + .5*self.nu*(u*self.sigma)**2 -1j * u *self.sigma*self.theta)**(-T/self.nu)
    
        # # char_exp = (1j * u * T)/self.nu * np.log(1 - self.nu*self.theta -.5 * self.nu * self.sigma**2) * 1 / (1 + self.nu*(u*self.sigma)**2 - 1j*u*self.nu*self.sigma)**(T/self.nu)
        
        # return np.exp(char_exp)
        
        # drift = np.log(1 - self.nu*self.r - 0.5 * self.nu * self.sigma**2) / self.nu
        # denominator = (1 - 1j * u * self.nu * drift + 0.5 * self.nu * self.sigma**2 * u**2)
        # char_exp = (denominator)**(-T/self.nu)
        # return char_exp
    
    def Damping_integrand(self, K, T, u, alpha, X0):
        
        return (self.char_fun(-u - alpha*1j, T,) / ((1j*u - alpha) * (1j*u - alpha + 1)) * np.exp(-1j * u * np.log(X0/K))).real;
        # numerator = self.char_fun(-u - alpha * 1j, T) * np.exp(-1j * u * np.log(X0/K))
        # denominator = (1j * u + alpha) * (1j * u + alpha - 1)
        # return (numerator / denominator).real
    
    def Fourier_call_pricing(self, K, T, X0, alpha=1.5, t=0, a=0, b=50, n_inter=100):
        
        # r = self.r; # the risk free (drift of the process) 
        
        # int_value = quad(lambda u: self.Damping_integrand(K, T, u, alpha, X0), a, b, limit=n_inter)[0];

        # call_value = ((np.exp(-r * (T-t)) * X0**alpha * K**(1-alpha)) / pi) * int_value;

        # return call_value;
        
        r = self.r
        int_value, _ = quad(lambda u: self.Damping_integrand(K, T, u, alpha, X0), a, b, limit=n_inter)
        call_value = np.exp(-r * (T-t)) * (X0 ** alpha * K ** (1 - alpha) / np.pi) * int_value
        return max(call_value, 0)  # Ensuring non-negative price
    
    def Fourier_put_pricing(self, K, T, X0, alpha=1.5, t=0, a=0, b=50, n_inter=100):
        
        self.Fourier_call_pricing(K, T, X0, alpha, a, b, n_inter) - X0 + K*np.exp(-self.r*(T-t))
    
    def mc_call_pricing(self, X0, T, K, n_steps, N=0):
        
        trj = self.simulate(X0, n_steps, T, N); # Generating the trajectories of the asset
        call_payoff = trj[-1] - K;
        call_payoff[call_payoff<0] = 0;
        call_payoff = call_payoff*np.exp(-self.r*T)
        
        return np.mean(call_payoff);
    
    def mc_put_pricing(self, X0, T, K, n_steps, N=0):
        
        trj = self.simulate(X0, n_steps, T, N); # Generating the trajectories of the asset
        put_payoff = K - trj[-1];
        put_payoff[put_payoff<0] = 0;
        put_payoff = put_payoff*np.exp(-self.r*T)
        
        return np.mean(put_payoff);
    
    def cos_pricing(self, S0, K, T, typology, N=32):
        
        cum1 = (self.r - 0.5*self.sigma**2)*T
        cum2 = self.sigma**2 * T
        cum4 = 0
        
        L = 10
        a = cum1 - L * (np.sqrt(cum2 + np.sqrt(cum4)))
        b = cum1 + L * (np.sqrt(cum2 + np.sqrt(cum4)))
        
        bma = b-a
        k  = np.arange(N+1)
        u  = k * np.pi/(b-a)
                
        CF = self.char_fun(u, T) # computing the characteristic function 
        
        x  = np.log(S0/K)
        Term = np.exp(1j * k * np.pi * (x-a)/bma)
        Fk = np.real(np.multiply(CF, Term))
        Fk[0] = 0.5 * Fk[0] 
        
        r = self.r
        
        V_COS = K * np.sum(np.multiply(Fk,UK(k, a, b, typology))) * np.exp(-r*T)
        
        return V_COS
    
    def obj_fun_price(self, params, data, alpha=1.5):
        
        theta = params[0]
        sigma = params[1]
        nu = params[2]
        
        self.sigma = sigma
        self.nu = nu
        self.theta = theta
        
        model_prices = []
        mkt_prices = []
        
        for item in data:
            for option in item:
            
                X0 = option.underlying_price
                K = option.strike_price
                T = option.time_to_maturity
                option_type = option.typology
                
                if option_type=='C':
                    # model_prices.append(self.cos_pricing(X0, K, T, 'call'))
                    model_prices.append(self.Fourier_call_pricing(K, T, X0))
                    mkt_prices.append(option.market_price)
                    
                elif option_type=='P':
                    # model_prices.append(self.cos_pricing(X0, K, T, 'put'))
                    model_prices.append(self.Fourier_call_pricing(K, T, X0))
                    mkt_prices.append(option.market_price)
        
        model_prices = np.array(model_prices)
        mkt_prices = np.array(mkt_prices)
        
        result = np.sum((mkt_prices - model_prices)**2)
        # print(result)
        return  result
    
    def calibrate(self, params, data, method='L-BFGS-B'):
        
        bounds = Bounds([-np.inf, 0, 1], [np.inf, np.inf, np.inf])
        
        
        res = minimize(self.obj_fun_price, params, bounds=bounds, method=method, 
                       args=(data), jac='2-points')
        
        index = ['theta', 'sigma', 'nu']
        columns = ['params','gradient']
        
        df = pd.DataFrame(np.vstack((res.x, res.jac)).T, index=index, columns=columns)
        
        theta = res.x[0]
        sigma = res.x[1]
        nu = res.x[2]
        
        self.sigma = sigma
        self.nu = nu
        self.theta = theta
        
        return FitResult(df, res.fun)
    
    
    
    
        
        
if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    from datetime import datetime
    from Option_lib import VanillaOption, implied_vol, Time_series_derivatives
    
    np.random.seed(42)
    today = datetime.today()
    
    # # Testing GBM object
    
    # mu = 0 
    # sigma = .24
    # S0 = 3
    
    # n_steps = 10
    # N = 20
    
    # T = 1 
    
    # model = GBM(mu=mu, sigma=sigma)
    
    # St = model.simulate(S0, T, n_steps, N)
        
    # print(np.mean(St, axis=1))
    # print()
    
    # # pricing call option 
    
    # K = 3
    
    # mc_call_price = model.mc_call_pricing(S0, n_steps, K, T, N)
    # bs_call_price = model.BS_call_price(S0, K, T)
    # fr_call_price = model.Fourier_call_pricing(S0, K, T)
    
    # print(bs_call_price)
    # print(mc_call_price)
    # print(fr_call_price)
    # print()
    
    # maturities = np.array([1/252, 5/252, 21/252, 42/252, 63/252, 84/252, 126/252])
    # strikes = np.arange(2.5, 3.6, step=.1)
    
    # mkt_data = []
    # mkt_implied_vol = []
    
    # Options = Time_series_derivatives()
    
    # P = 0
    
    # today = datetime.today()
    
    # for T in maturities:
    #     for K in strikes:
            
    #         call_price = model.BS_call_price(S0, K, T)
    #         mkt_data.append([call_price, S0, K, T, 'C'])
            
    #         v = implied_vol(K, T, call_price, S0, .001, 1, 0)
    #         mkt_implied_vol.append([v, S0, K, T, 'C'])
            
    #         options = VanillaOption(market_price=call_price,
    #                                 strike_price=K,
    #                                 time_to_maturity=T,
    #                                 typology='C',
    #                                 style='E',
    #                                 trading_date=today,
    #                                 underlying_price=S0,
    #                                 implied_vol=v)
            
    #         Options.add_contract(options, today)
            
            # put = model.BS_put_price(S0, K, T)
            # mkt_data.append([put, S0, K, T, 'P'])
        
    # for item in Options:
    #     print('hello')
    
    # params = np.array([1])
    
    # fitted_gbm = GBM(mu=mu)
        
    # res = fitted_gbm.calibrate(params, Options)
    
    
    # Testing the Heston model
    
    # print('Heston model')
    
    # mu = 0.0; 			 # deterministic and fixed short rate
    # k = 1.5;             # speed of mean reversion of the vol process
    # theta = 0.04;        # long run memory of the vol process
    # nu0 = 0.04;          # initial value of the vol process
    # sigma = .2;          # diffusion parameter of the vol process
    # rho = -0.1;          # correlation of the Brownians (leverage effect)
    # S0 = 3
    
    # model2 = Heston93(mu, k, theta, sigma, rho)
    
    # # # n_steps = 100;
    # # # # N = 10;
    # # # N=15;
    # # # T = 0.25; # number of years
    
    # # # X0 = 11.75;
    # # # K = X0
    
    # # # Xt, nut = model2.simulate(X0, nu0, T, n_steps, N, .5, .5, 'andersen');
    
    # # # # print(np.mean(Xt, axis=1))
    # # # # print()
    
    # # # mc_call_price = model2.mc_call_pricing(n_steps, K, T, X0, nu0, 0, N)
    # # # fr_call_price = model2.Fourier_call_pricing(K, T, X0, nu0)
    
    # # # print(mc_call_price)
    # # # print(fr_call_price)
    # # # print()
    
    # maturities = np.array([1/252, 5/252, 21/252, 42/252, 63/252, 84/252, 126/252])
    # strikes = np.arange(2.5, 3.6, step=.1)
    # X0 = 3 
    
    # mkt_data = []
    
    # Options = Time_series_derivatives()
    # mkt_data = []
    # mkt_implied_vol = []
        
    # for T in maturities:
    #     for K in strikes:
            
    #         call_price = model2.Fourier_call_pricing(K, T, X0, nu0)
    #         mkt_data.append([call_price, S0, K, T, 'C'])
            
    #         v = implied_vol(K, T, call_price, S0, .001, 1, 0)
    #         mkt_implied_vol.append([v, S0, K, T, 'C'])
            
    #         options = VanillaOption(market_price=call_price,
    #                                 strike_price=K,
    #                                 time_to_maturity=T,
    #                                 typology='C',
    #                                 style='E',
    #                                 trading_date=today,
    #                                 underlying_price=S0,
    #                                 implied_vol=v)
            
    #         Options.add_contract(options, today)
            
    #         # put = model2.BS_put_price(S0, K, T)
    #         # mkt_data.append([put, S0, K, T, 'P'])
    
    # params = np.array([k, theta, sigma, 0, nu0]) * 1.5
    
    # fitted_heston = Heston93(0)
    
    # # fitted_gbm.obj_fun(params, mkt_data)
    
    # res = fitted_heston.calibrate(params, Options)
    
    
    # Testing Bates model 
    print('Bates')
    print()
    
    mu = 0.0; 			 # deterministic and fixed short rate
    k = 1.5;             # speed of mean reversion of the vol process
    theta = 0.04;       # long run memory of the vol process
    nu0 = 0.04;          # initial value of the vol process
    sigma = .2;        # diffusion parameter of the vol process
    rho = -0.1;         # correlation of the Brownians (leverage effect)
    muj = 0;
    sigmaj = 0.002;
    lam = 1;
    
    model3 = Bates(mu, k, theta, sigma, rho, lam, muj, sigmaj)
    
    # maturities = np.array([1/252, 5/252, 21/252, 42/252, 63/252, 84/252, 126/252])
    # strikes = np.arange(2.5, 3.6, step=.1)
    # X0 = 3 
    
    # mkt_data = []
    # mkt_implied_vol = []
    
    # sigma1 = 0.0001
    # sigma2 = 1
    
    start = time()
        
    # for T in maturities:
    #     for K in strikes:
            
    #         call = model3.Fourier_call_pricing(K, T, X0, nu0)
    #         mkt_data.append([call, X0, K, T, 'C'])
            
    #         vol = implied_vol(K, T, call, X0, sigma1, sigma2, mu)
    #         if vol<10**-4:
    #             continue
    #         mkt_implied_vol.append([vol, X0, K, T, 'C'])
    
    maturities = np.array([1/252, 5/252, 21/252, 42/252, 63/252, 84/252, 126/252])
    strikes = np.arange(2.5, 3.6, step=.1)
    X0 = 3 
    
    Options = Time_series_derivatives()
    mkt_data = []
    mkt_implied_vol = []
    
    D = np.empty((len(maturities), len(strikes)))
    
    i = 0
    j = 0 
    
    for T in maturities:
        for K in strikes:
            
            call_price = model3.Fourier_call_pricing(K, T, X0, nu0)
            mkt_data.append([call_price, X0, K, T, 'C'])
            
            v = implied_vol(K, T, call_price, X0, .001, 1, 0)
            mkt_implied_vol.append([v, X0, K, T, 'C'])
            
            D[i,j] = call_price
            
            options = VanillaOption(market_price=call_price,
                                    strike_price=K,
                                    time_to_maturity=T,
                                    typology='C',
                                    style='E',
                                    trading_date=today,
                                    underlying_price=X0,
                                    implied_vol=v)
            
            Options.add_contract(options, today)
            j += 1
        i+=1 
        j=0
    
    end = time()
    
    print('Elapsed time : '+str(end-start))
    
    # params = np.array([k, theta, sigma, 0, lam, sigmaj, nu0]) * 1.5
    
    # fitted_bates = Bates(mu=mu, muj=muj)
    
    # # fitted_bates.obj_fun_implied_vol(params, mkt_implied_vol)
    
    # start = time()
    # res = fitted_bates.calibrate(params, Options)
    # end = time()
    # print('Elapsed time : '+str(end-start))
    
    # # res = fitted_bates.calibrate(params, mkt_implied_vol, target='vol')
        
    # # test = model3.Damping_integrand(K, T, np.arange(0,10), 1.5)
    
    # # plt.plot(test,'-o')
    # # plt.show()
    
    # # X0 = 5.75;
    # # K = X0
    
    # # N = 15;
    # # n_steps = 100;
    # # T = 0.25;
    
    # # model3 = Bates(mu, k, theta, sigma, rho, lam, muj, sigmaj)
    
    # # Xt, nut = model3.simulate(X0, nu0, n_steps, T, N);
    
    # # # print(np.mean(Xt, axis=1))
    # # # print()
     
    # # mc_call_price = model3.mc_call_pricing(T, K, n_steps, N, X0, nu0)
    # # fr_call_price = model3.Fourier_call_pricing(K, T, X0, nu0)
    
    # # print(mc_call_price)
    # # print(fr_call_price)
    # # print()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        