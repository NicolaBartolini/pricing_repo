# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 17:33:39 2025

@author: Nicola
"""

import numpy as np 
import pandas as pd
from scipy.integrate import quad



def continuous_spot_rate(P, T, t=0):
    
    return -np.log(P)/(T-t);


def CIR_factor(x, theta, k, sigma, T, t=0):
    '''r = short rata at t  theta = long run mean\n k = speed of mean reversion
    sigma = volatility of r  delta_t = maturity of the bond (year fraction)
    mkt_prices = bond prices on the market'''

    delta_t = T-t;    

    th = theta

    s = sigma

    h = np.sqrt( k**2 + 2 * s**2)

    A = ((2 * h * np.exp((k + h) * delta_t * .5)) / (2 * h + (k + h) * (np.exp(delta_t*h) - 1))) ** ((2 * k * th)/ s**2)


    B = (2 * (np.exp(delta_t * h) - 1)) / (2 * h + (k + h) * (np.exp(delta_t*h) - 1))

    return A * np.exp(-B * x)

def Levy_OU_factor(alpha, lam, eta, x0, beta, T, t=0):
    
    # The Langeville eq is assumed to be driven by a pure jump Levy process
    # with exponential jumps
    
    B = 1/alpha * (1 - np.exp(-alpha*(T-t)));
    
    fun = lambda s : lam * ( eta/(eta + beta/alpha * (1 - np.exp(-alpha*(T-s)))) -1);
    
    A = quad(fun, t, T)[0];
    
    return np.exp(A + B * beta * x0);


def risky_zcb(k1, theta1, sigma1, gamma1, 
              x1, alpha1, lam1, eta1, beta1, 
              k2, theta2, sigma2, gamma2, beta2,
              k3, theta3, sigma3, gamma3, beta3, 
              T, DF, delta):
    # Pricing formula for tehe risky zero-coupon-bond where the hazard rate
    # follows a CIR dynamic and the paymenent in case of default happens at the 
    # maturity
    # k = speed of mean reversion
    # theta = long run mean of the process
    # sigma = vol of the process
    # delta = recovery rate
    # T = maturity of the zcb
    # DF = risk free discount factor with maturity T
    # alpha1 = speed of mean reversion of the first physical risky factor
    # lam1 = Poisson intensity of the first physical risky factor
    # eta1 = parameter of the exponential distribution  for the jump width 
    # beta1 = coefficient for the impact of the first factor
    # alpha2 = speed of mean reversion of the second physical risky factor
    # lam2 = Poisson intensity of the second physical risky factor
    # eta2 = parameter of the exponential distribution for the jump width 
    # beta2 = coefficient for the impact of the second factor
    
    survival_prob = CIR_factor(gamma1, theta1, k1, sigma1, T, 0) * CIR_factor(beta2*gamma2, beta2*theta2, k2, beta2*sigma2, T, 0) * CIR_factor(beta3*gamma3, beta3*theta3, k3, beta3*sigma3, T, 0) * Levy_OU_factor(alpha1, lam1, eta1, x1, beta1, T, 0); 

    return DF * survival_prob + delta*(1-survival_prob)*DF;



def risky_zcb0(k1, theta1, sigma1, gamma1, T, DF, delta):
    survival_prob = CIR_factor(gamma1, theta1, k1, sigma1, T, 0) 

    return DF * survival_prob + delta*(1-survival_prob)*DF;


def risky_zcb1(k1, theta1, sigma1, gamma1, #\weather CIR
              k3, theta3, sigma3, gamma3, beta3, 
              T, DF, delta):
    survival_prob = CIR_factor(gamma1, theta1, k1, sigma1, T, 0) * CIR_factor(beta3*gamma3, beta3*theta3, k3, beta3*sigma3, T, 0) ;

    return DF * survival_prob + delta*(1-survival_prob)*DF;

def risky_zcb13(k1, theta1, sigma1, gamma1, #1 weather Lévy
              x1, alpha1, lam1, eta1, beta1, 
              T, DF, delta):
    survival_prob = CIR_factor(gamma1, theta1, k1, sigma1, T, 0) * Levy_OU_factor(alpha1, lam1, eta1, x1, beta1, T, 0);

    return DF * survival_prob + delta*(1-survival_prob)*DF;


def risky_zcb2(k1, theta1, sigma1, gamma1, 
              k2, theta2, sigma2, gamma2, beta2,
              k3, theta3, sigma3, gamma3, beta3, 
              T, DF, delta):
    survival_prob = CIR_factor(gamma1, theta1, k1, sigma1, T, 0) * CIR_factor(beta2*gamma2, beta2*theta2, k2, np.sqrt(beta2)*sigma2, T, 0) * CIR_factor(beta3*gamma3, beta3*theta3, k3, np.sqrt(beta3)*sigma3, T, 0) ;

    return DF * survival_prob + delta*(1-survival_prob)*DF;

def risky_zcb21(k1, theta1, sigma1, gamma1, 
              x1, alpha1, lam1, eta1, beta1, 
              k3, theta3, sigma3, gamma3, beta3, 
              T, DF, delta):
    
    survival_prob = CIR_factor(gamma1, theta1, k1, sigma1, T, 0) * CIR_factor(beta3*gamma3, beta3*theta3, k3, np.sqrt(beta3)*sigma3, T, 0) * Levy_OU_factor(alpha1, lam1, eta1, x1, beta1, T, 0); 

    return DF * survival_prob + delta*(1-survival_prob)*DF;

def risky_zcb24(k1, theta1, sigma1, gamma1, #2 weather Lévys
              x1, alpha1, lam1, eta1, beta1, 
              x4, alpha4, lam4, eta4, beta4,
              T, DF, delta):
    survival_prob = CIR_factor(gamma1, theta1, k1, sigma1, T, 0) * Levy_OU_factor(alpha1, lam1, eta1, x1, beta1, T, 0)* Levy_OU_factor(alpha4, lam4, eta4, x4, beta4, T, 0);

    return DF * survival_prob + delta*(1-survival_prob)*DF;


def risky_zcb34(k1, theta1, sigma1, gamma1, #3 weather Lévys
              x1, alpha1, lam1, eta1, beta1, 
              x4, alpha4, lam4, eta4, beta4,
              x5, alpha5, lam5, eta5, beta5,
              T, DF, delta):
    survival_prob = CIR_factor(gamma1, theta1, k1, sigma1, T, 0) * Levy_OU_factor(alpha1, lam1, eta1, x1, beta1, T, 0)* Levy_OU_factor(alpha4, lam4, eta4, x4, beta4, T, 0) * Levy_OU_factor(alpha5, lam5, eta5, x5, beta5, T, 0);

    return DF * survival_prob + delta*(1-survival_prob)*DF;


def risky_zcb314(k1, theta1, sigma1, gamma1, # 2 weather Lévys, 1 CIR
              x1, alpha1, lam1, eta1, beta1, 
              k3, theta3, sigma3, gamma3, beta3, 
              x4, alpha4, lam4, eta4, beta4,
              T, DF, delta):
    
    survival_prob = CIR_factor(gamma1, theta1, k1, sigma1, T, 0) * CIR_factor(beta3*gamma3, beta3*theta3, k3, np.sqrt(beta3)*sigma3, T, 0) * Levy_OU_factor(alpha1, lam1, eta1, x1, beta1, T, 0) *  Levy_OU_factor(alpha4, lam4, eta4, x4, beta4, T, 0); 

    return DF * survival_prob + delta*(1-survival_prob)*DF;


def risky_zcb41(k1, theta1, sigma1, gamma1, # 3 weather Lévys, 1 CIR
              x1, alpha1, lam1, eta1, beta1, 
              k3, theta3, sigma3, gamma3, beta3, 
              x4, alpha4, lam4, eta4, beta4,
              x5, alpha5, lam5, eta5, beta5,
              T, DF, delta):
    
    survival_prob = CIR_factor(gamma1, theta1, k1, sigma1, T, 0) * CIR_factor(beta3*gamma3, beta3*theta3, k3, np.sqrt(beta3)*sigma3, T, 0) * Levy_OU_factor(alpha1, lam1, eta1, x1, beta1, T, 0) *  Levy_OU_factor(alpha4, lam4, eta4, x4, beta4, T, 0) * Levy_OU_factor(alpha5, lam5, eta5, x5, beta5, T, 0); 

    return DF * survival_prob + delta*(1-survival_prob)*DF;


def risky_cb(k1, theta1, sigma1, gamma1,
             x1, alpha1, lam1, eta1, beta1, 
             k2, theta2, sigma2, gamma2, beta2,
             k3, theta3, sigma3, gamma3, beta3,
             c, cT, T, DF, delta):
    
    # Pricing formula for tehe risky coupon-bond where the hazard rate
    # follows a CIR dynamic and the paymenent in case of default happens at the 
    # maturity
    # k = speed of mean reversion
    # theta = long run mean of the process
    # sigma = vol of the process
    # delta = recovery rate
    # T = maturity of the zcb
    # cT = is the array or list with the grid of coupon payments
    # DF = array of risk free discount factors with maturity for each
    # coupon payment and the last one is the discount factor with maurity T.
    # The recovery rate in case of default for the coupons is assumed to be 0.
    
    result = 0;
    
    for i in np.arange(0, len(cT)):

        result += c * risky_zcb(k1, theta1, sigma1, gamma1, x1, alpha1, lam1, eta1, beta1, k2, theta2, sigma2, gamma2, beta2, k3, theta3, sigma3, gamma3, beta3, cT[i], DF[i], 0); # computing the zcb associated to the coupon payment
        
    result += risky_zcb(k1, theta1, sigma1, gamma1, x1, alpha1, lam1, eta1, beta1, k2, theta2, sigma2, gamma2, beta2, k3, theta3, sigma3, gamma3, beta3, T, DF[-1], delta); # computing the risky zcb for the maturity payment
    
    return result;


def risky_cb21(k1, theta1, sigma1, gamma1,
             x1, alpha1, lam1, eta1, beta1, 
             k3, theta3, sigma3, gamma3, beta3,
             c, cT, T, DF, delta):

    result = 0;
    
    for i in np.arange(0, len(cT)):

        result += c * risky_zcb21(k1, theta1, sigma1, gamma1, x1, alpha1, lam1, eta1, beta1, k3, theta3, sigma3, gamma3, beta3, cT[i], DF[i], 0); # computing the zcb associated to the coupon payment
        
    result += risky_zcb21(k1, theta1, sigma1, gamma1, x1, alpha1, lam1, eta1, beta1, k3, theta3, sigma3, gamma3, beta3, T, DF[-1], delta); # computing the risky zcb for the maturity payment
    
    return result;


def risky_cb0(k1, theta1, sigma1, gamma1,
             c, cT, T, DF, delta):
    
    result = 0;
    
    for i in np.arange(0, len(cT)):

        result += c * risky_zcb0(k1, theta1, sigma1, gamma1, cT[i], DF[i], 0); # computing the zcb associated to the coupon payment
        
    result += risky_zcb0(k1, theta1, sigma1, gamma1, T, DF[-1], delta); # computing the risky zcb for the maturity payment
    
    return result;

def risky_cb1(k1, theta1, sigma1, gamma1,
             k3, theta3, sigma3, gamma3, beta3,
             c, cT, T, DF, delta):
    
    result = 0;
    
    for i in np.arange(0, len(cT)):

        result += c * risky_zcb1(k1, theta1, sigma1, gamma1, k3, theta3, sigma3, gamma3, beta3, cT[i], DF[i], 0); # computing the zcb associated to the coupon payment
 
    result += risky_zcb1(k1, theta1, sigma1, gamma1, k3, theta3, sigma3, gamma3, beta3, T, DF[-1], delta); # computing the risky zcb for the maturity payment
    
    return result;


def risky_cb13(k1, theta1, sigma1, gamma1,
             x1, alpha1, lam1, eta1, beta1,
             c, cT, T, DF, delta):
    
    result = 0;
    
    for i in np.arange(0, len(cT)):

        result += c * risky_zcb13(k1, theta1, sigma1, gamma1, x1, alpha1, lam1, eta1, beta1, cT[i], DF[i], 0); # computing the zcb associated to the coupon payment
        
    result += risky_zcb13(k1, theta1, sigma1, gamma1, x1, alpha1, lam1, eta1, beta1, T, DF[-1], delta); # computing the risky zcb for the maturity payment
    
    return result;


def risky_cb2(k1, theta1, sigma1, gamma1,
             k2, theta2, sigma2, gamma2, beta2,
             k3, theta3, sigma3, gamma3, beta3,
             c, cT, T, DF, delta):
    
    result = 0;
    
    for i in np.arange(0, len(cT)):

        result += c * risky_zcb2(k1, theta1, sigma1, gamma1, k2, theta2, sigma2, gamma2, beta2, k3, theta3, sigma3, gamma3, beta3, cT[i], DF[i], 0); # computing the zcb associated to the coupon payment
        
    result += risky_zcb2(k1, theta1, sigma1, gamma1, k2, theta2, sigma2, gamma2, beta2, k3, theta3, sigma3, gamma3, beta3, T, DF[-1], delta); # computing the risky zcb for the maturity payment
    
    return result;


def risky_cb24(k1, theta1, sigma1, gamma1,
             x1, alpha1, lam1, eta1, beta1,
             x4, alpha4, lam4, eta4, beta4,
             c, cT, T, DF, delta):
    
    result = 0;
    
    for i in np.arange(0, len(cT)):

        result += c * risky_zcb24(k1, theta1, sigma1, gamma1, x1, alpha1, lam1, eta1, beta1, x4, alpha4, lam4, eta4, beta4, cT[i], DF[i], 0); # computing the zcb associated to the coupon payment
        
    result += risky_zcb24(k1, theta1, sigma1, gamma1, x1, alpha1, lam1, eta1, beta1, x4, alpha4, lam4, eta4, beta4, T, DF[-1], delta); # computing the risky zcb for the maturity payment
    
    return result;


def risky_cb34(k1, theta1, sigma1, gamma1,
             x1, alpha1, lam1, eta1, beta1,
             x4, alpha4, lam4, eta4, beta4,
             x5, alpha5, lam5, eta5, beta5,
             c, cT, T, DF, delta):
    
    result = 0;
    
    for i in np.arange(0, len(cT)):

        result += c * risky_zcb34(k1, theta1, sigma1, gamma1, x1, alpha1, lam1, eta1, beta1, x4, alpha4, lam4, eta4, beta4, x5, alpha5, lam5, eta5, beta5, cT[i], DF[i], 0); # computing the zcb associated to the coupon payment
        
    result += risky_zcb34(k1, theta1, sigma1, gamma1, x1, alpha1, lam1, eta1, beta1, x4, alpha4, lam4, eta4, beta4, x5, alpha5, lam5, eta5, beta5, T, DF[-1], delta); # computing the risky zcb for the maturity payment
    
    return result;


def risky_cb314(k1, theta1, sigma1, gamma1,  # 2 weather Lévys, 1 CIR
             x1, alpha1, lam1, eta1, beta1, 
             k3, theta3, sigma3, gamma3, beta3,
             x4, alpha4, lam4, eta4, beta4,
             c, cT, T, DF, delta):

    result = 0;
    
    for i in np.arange(0, len(cT)):

        result += c * risky_zcb314(k1, theta1, sigma1, gamma1, x1, alpha1, lam1, eta1, beta1, k3, theta3, sigma3, gamma3, beta3, x4, alpha4, lam4, eta4, beta4, cT[i], DF[i], 0); # computing the zcb associated to the coupon payment
        
    result += risky_zcb314(k1, theta1, sigma1, gamma1, x1, alpha1, lam1, eta1, beta1, k3, theta3, sigma3, gamma3, beta3, x4, alpha4, lam4, eta4, beta4, T, DF[-1], delta); # computing the risky zcb for the maturity payment
    
    return result;

def risky_cb41(k1, theta1, sigma1, gamma1,  # 3 weather Lévys, 1 CIR
             x1, alpha1, lam1, eta1, beta1, 
             k3, theta3, sigma3, gamma3, beta3,
             x4, alpha4, lam4, eta4, beta4,
             x5, alpha5, lam5, eta5, beta5,
             c, cT, T, DF, delta):

    result = 0;
    
    for i in np.arange(0, len(cT)):

        result += c * risky_zcb41(k1, theta1, sigma1, gamma1, x1, alpha1, lam1, eta1, beta1, k3, theta3, sigma3, gamma3, beta3, x4, alpha4, lam4, eta4, beta4, x5, alpha5, lam5, eta5, beta5, cT[i], DF[i], 0); # computing the zcb associated to the coupon payment

    result += risky_zcb41(k1, theta1, sigma1, gamma1, x1, alpha1, lam1, eta1, beta1, k3, theta3, sigma3, gamma3, beta3, x4, alpha4, lam4, eta4, beta4, x5, alpha5, lam5, eta5, beta5, T, DF[-1], delta); # computing the risky zcb for the maturity payment
    
    return result;



class bond:
    # This class is the class for managing all the information of the coupon bonds 
    # needed for the calibration
    
    def __init__(self, maturity, coupon_grid_payments, list_of_discount_factors, 
                 coupon_rate, recovery_rate, market_price):
        
        self.maturity = maturity;
        self.coupon_grid_payments = coupon_grid_payments;
        self.list_of_discount_factors = list_of_discount_factors;
        self.recovery_rate = recovery_rate;
        self.market_price = market_price;
        self.coupon_rate = coupon_rate;

class term_structure:
    
    # The term structure object for recovering eventual missing discount factors 
    
    def __init__(self, list_time_to_mat, list_of_DF):
        # list_time_to_mat = list or numpy array with the time to maturity of each 
        # risk free discount factor
        # list_of_DF = list or numpy array of discount factor
        self.DF = pd.DataFrame(list_of_DF, index=list_time_to_mat, 
                               columns=['Discount_factor']);
        
    def get_all(self):
        # it returns all the term structure
        return self.DF;
    
    def get_discount_factors(self, maturities):
        # It returns the discount facor associeated to maturities of interest 
        # in case one or more maturities are in self.DF they revered by
        # linera interpolation
        # maturities = list or numpy array of maturities. 
        
        if type(maturities)==float or type(maturities)==int:
            
            maturities = np.array([maturities], dtype=float);
        
        result = np.empty(len(maturities))
        
        for i in range(0, len(maturities)):
            
            T = maturities[i];
            
            if T not in list(self.DF.index):
                
                self.DF.loc[T] = np.nan;
                #self.DF = self.DF.sort_index();
        
        self.DF = self.DF.sort_index();
        self.DF = self.DF.interpolate();
        
        for i in range(0, len(maturities)):
            
            T = maturities[i];
            result[i] = self.DF.loc[T];
            
        return pd.Series(result, index=maturities);
