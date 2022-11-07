import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

def loadData(path):
    '''
        Loading the Data from the Path 
    '''
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
    covid_data =  pd.read_csv(path ,parse_dates=['Date'], date_parser=dateparse)
    covid_data = covid_data[['Date' , 'Confirmed' , 'Tested' , 'First Dose Administered']]
    def noncumulative(colname):
        data = covid[colname].copy().values
        for i in range(len(data)-1,0,-1):
            data[i] -= data[i-1]
        return data
    covid = covid_data.copy()
    covid['Confirmed'] = noncumulative('Confirmed')
    covid['Tested'] = noncumulative('Tested')
    covid['First Dose Administered'] = noncumulative('First Dose Administered')
    return covid

def initR(N , s = 0.156 , e = .36 ):
    '''
        Initialise the R0 randomly
    '''
    return np.random.uniform(s,e) * N

def initCIR(s = 12.0 , e = 30.0):
    '''
        Initialise the R0 randomly
    '''
    return np.random.uniform(s,e)
    
def avgTest(start_date , end_date):
    days = (end_date - start_date).days + 1
    data = covid.loc[(covid.Date >= start_date) & (covid.Date <= end_date)][['Date','Tested']].copy()
    T = np.zeros(days)
    for t in range(days):
        cur_date = start_date + timedelta(days = t)
        seven_day_data = data.loc[(data.Date > cur_date - timedelta(days = 7))&(data.Date <= cur_date )]
        values = seven_day_data['Tested'].values
        if len(values) > 0:
            T[t] = np.mean(values)
        else:
            T[t] = T[t-1]
    del data
    return T

def avgConfirmed(start_date, end_date):
    days = (end_date - start_date).days + 1
    data = covid.loc[(covid.Date >= start_date) & (covid.Date <= end_date)][['Date','Confirmed']].copy()
    c = np.zeros(days)
    for t in range(days):
        cur_date = start_date + timedelta(days = t)
        seven_day_data = data.loc[(data.Date > cur_date - timedelta(days = 7))&(data.Date <= cur_date )]
        c[t] = np.mean(seven_day_data['Confirmed'].values)
    del data
    return c

def avgdelta_i(delta_i, days):
    avgdelta_i = np.zeros(days)
    for t in range(days):
        avgdelta_i[t] = np.mean(delta_i[max(t-7,0):t+1])
    
    return avgdelta_i

def SEIRV(beta , S0 , E0 , I0 , R0 , delta_V, s_date , e_date):
    days = (e_date - s_date).days + 1
    S = np.zeros(days) 
    E = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    S[0] , E[0] , I[0], R[0] = S0 , E0 , I0 , R0
    delta_W = R0 / 30 

    for t in range(1,days):        
        S[t] = S[t-1] - (beta * S[t-1] * I[t-1] / N) - (epsilon * delta_V[t])  + delta_W
        E[t] = E[t-1] + (beta * S[t-1] * I[t-1] / N)  - alpha * E[t-1]
        I[t] = I[t-1] + (alpha * E[t-1]) - gamma * I[t-1]
        R[t] = R[t-1] + gamma * I[t-1] + (epsilon * delta_V[t]) - delta_W

        cur_date = s_date + timedelta(days = t)
        if cur_date >= datetime(2021,4,15):
            delta_W = 0
        elif cur_date >= datetime(2021,9,11):
            delta_W = epsilon * delta_V[t-180] + (R[t-180] - R[t-181])
    
    return S , E , I , R

def findLoss(BETA , S0 , E0 , I0 , R0 ,CIR0 ,delta_V, T, s_date , e_date):
    days = (e_date - s_date).days + 1
    S , E, I, R = SEIRV(BETA , S0 , E0 , I0 , R0 ,delta_V ,  s_date , e_date)
    CIR  = CIR0 * T[0] / T
    delta_i = alpha * E / CIR
    avg_delta_i = avgdelta_i(delta_i,days)
    log_avg_delta_i = np.log(avg_delta_i)

    assert(len(log_avg_c_t) == len(avg_delta_i) and len(log_avg_c_t) == 42)
    loss = np.mean((log_avg_c_t - log_avg_delta_i)**2)
    return loss

def findGradients(BETA , S0 , E0 , I0 , R0 ,CIR0, delta_V, T, s_date , e_date , loss):
    loss_beta = findLoss(BETA + 0.01, S0 , E0 , I0, R0 ,CIR0 , delta_V, T, s_date , e_date )
    loss_e0 = findLoss(BETA, S0 , E0 + 1 , I0, R0 ,CIR0 , delta_V, T, s_date , e_date )
    loss_i0 = findLoss(BETA, S0 , E0 , I0 + 1, R0 ,CIR0 , delta_V, T, s_date , e_date )
    loss_r0 = findLoss(BETA, S0 , E0 , I0, R0 + 1 ,CIR0 , delta_V, T, s_date , e_date )
    loss_cir0 = findLoss(BETA, S0 , E0 , I0, R0 ,CIR0 + 0.1 ,delta_V, T, s_date , e_date )
    
    delta_beta  = loss_beta - loss
    delta_e0 = loss_e0 - loss
    delta_i0 = loss_i0 - loss
    delta_r0 = loss_r0 - loss
    delta_cir0 = loss_cir0 - loss

    return delta_beta , delta_e0 , delta_i0 , delta_r0,  delta_cir0

def optimize(BETA, S0 , E0 , I0 , R0 ,CIR0 ,delta_V , T,  s_date , e_date , epochs):
    loss = findLoss(BETA, S0 , E0 , I0 , R0 ,CIR0 ,delta_V, T, s_date , e_date)
    epoch = 0
    while loss >= 0.01 and epoch <= epochs:
        delta_beta , delta_e0 , delta_i0 , delta_r0,  delta_cir0 = findGradients(BETA , S0 , E0 , I0 , R0 ,CIR0, delta_V , T, s_date , e_date , loss)

        if epoch < 10:
            step = epoch
        BETA = BETA - delta_beta /(step + 1)
        E0 = E0 - delta_e0 / (step + 1)
        I0 = I0 - delta_i0 / (step + 1)
        R0 = R0 - delta_r0 / (step + 1)
        CIR0 = CIR0 - delta_cir0 / (step + 1)
        S0 = N-E0-I0-R0 
        loss = findLoss(BETA, S0 , E0 , I0 , R0 ,CIR0 ,delta_V, T, s_date , e_date)
        if epoch % 1000 == 0:
            print(f'Epochs {epoch} || Loss {loss}')
        epoch +=1
    print(f'Loss : {loss} || Optimized Parameters :: BETA {BETA} || S0 {S0} || E0 {E0} || I0 {I0} || R0 {R0} || CIR0 {CIR0}')

    return BETA ,S0 ,E0 , I0 , R0 , CIR0

def predictOpenLoopControl(BETA , S0 , E0 , I0 , R0 ,CIR0 ,delta_V, T, s_date , e_date):
    S , E, I, R = SEIRV(BETA , S0 , E0 , I0 , R0 ,delta_V , s_date , e_date)
    CIR  = CIR0 * T[0] / T
    delta_i = alpha * E / CIR
    return delta_i , S/N

def predictCloseLoopControl(BETA , S0 , E0 , I0 , R0 ,CIR0 ,delta_V, T, s_date , e_date):
    days = (e_date - s_date).days + 1
    S = np.zeros(days) 
    E = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    S[0] , E[0] , I[0], R[0] = S0 , E0 , I0 , R0
    delta_W = R0 / 30 
    delta_i = np.zeros(days)
    CIR = np.zeros(days)
    CIR[0] = CIR0
    delta_i[0] = alpha * E[0] / CIR[0]
    beta = BETA
    for t in range(1,days):        
        S[t] = S[t-1] - (beta * S[t-1] * I[t-1] / N) - (epsilon * delta_V[t])  + delta_W
        E[t] = E[t-1] + (beta * S[t-1] * I[t-1] / N)  - alpha * E[t-1]
        I[t] = I[t-1] + (alpha * E[t-1]) - gamma * I[t-1]
        R[t] = R[t-1] + gamma * I[t-1] + (epsilon * delta_V[t]) - delta_W

        cur_date = s_date + timedelta(days = t)
        if cur_date >= datetime(2021,4,15):
            delta_W = 0
        elif cur_date >= datetime(2021,9,11):
            delta_W = epsilon * delta_V[t-180] + (R[t-180] - R[t-181])
        CIR[t]  = CIR0 * T[0] / T[t]
        delta_i[t] = alpha * E[t] / CIR[t]
        last7days_avg_delta_i = np.mean(delta_i[max(t-7 , 0) : t+1])
        if last7days_avg_delta_i <= 10000:
            beta = BETA 
        elif last7days_avg_delta_i <= 25000:
            beta = BETA * 2/3
        elif last7days_avg_delta_i <= 1_00_000:
            beta = BETA * 1/2
        else:
            beta = BETA * 1/3
        
    return delta_i , S / N

if __name__ == "__main__":
    covid = loadData(path = '../COVID19_data.csv')
    start_date , end_date  = datetime(2021,3,16) , datetime(2021,4,26) 
    N , alpha , gamma , epsilon = 70_000_000, 1/5.8, 1/5, 0.66
    delta_V = covid.loc[(covid.Date >= start_date) & (covid.Date <= end_date)]['First Dose Administered'].values
    _R0 = 0.20 * N
    _CIR0 = 12
    _E0 , _I0 = 0.12 * N / 100 , 0.12 * N/ 100
    _S0 = N - _E0 - _I0 - _R0
    T = avgTest(start_date , end_date)
    log_avg_c_t = np.log(avgConfirmed(start_date , end_date))
    epochs = 1000000
    BETA = 0.5

    BETA ,S0 ,E0 , I0 , R0 , CIR0 = optimize(BETA, _S0 , _E0 , _I0 , _R0 ,_CIR0 ,delta_V ,T , start_date , end_date , epochs)
    s_date , e_date = datetime(2021,3,16) , datetime(2021,12,31)
    days = (e_date - s_date).days + 1
    delta_V = np.full(days , 2_00_000)
    store_delta_V = covid.loc[(covid.Date >= start_date) & (covid.Date <= end_date)]['First Dose Administered'].values

    delta_V[:len(store_delta_V)] = store_delta_V
    T  = avgTest(start_date , e_date)

    assert(len(T) == len(delta_V))

    new_cases_open_beta , s_open_beta = predictOpenLoopControl(BETA , S0 , E0 , I0 , R0 ,CIR0 ,delta_V, T, s_date , e_date)
    new_cases_open_beta_two_third,s_open_beta_two_third = predictOpenLoopControl(BETA *2/3 , S0 , E0 , I0 , R0 ,CIR0 ,delta_V, T, s_date , e_date)
    new_cases_open_beta_one_third,s_open_beta_one_third = predictOpenLoopControl(BETA / 3 , S0 , E0 , I0 , R0 ,CIR0 ,delta_V, T, s_date , e_date)
    new_cases_open_beta_half ,s_open_beta_half = predictOpenLoopControl(BETA *1/2 , S0 , E0 , I0 , R0 ,CIR0 ,delta_V, T, s_date , e_date)

    new_cases_close , s_close = predictCloseLoopControl(BETA , S0 , E0 , I0 , R0 ,CIR0 ,delta_V, T, s_date , e_date)

    if not os.path.exists('./Plots') :
        os.mkdir('./Plots')
    plt.figure(1)
    plt.plot(new_cases_open_beta, label = 'Open Loop - beta' )
    plt.plot(new_cases_open_beta_two_third, label = 'Open Loop - beta * 2/3' )
    plt.plot(new_cases_open_beta_one_third, label = 'Open Loop - beta * 1/3' )
    plt.plot(new_cases_open_beta_half, label = 'Open Loop - beta * 1/2' )
    plt.plot(new_cases_close , label = 'Closed Loop' )
    plt.plot(covid.loc[(covid.Date >= s_date) & (covid.Date <= e_date)]['Confirmed'].values , label = 'Actual Confirmed Cases')
    plt.axvline(x= 43 , ymin=0, ymax = 1_00_000, linewidth=0.5, color='k', linestyle='dashed')
    plt.legend()
    plt.title('Open and Closed Loop Predictions till 31/12/2021')
    plt.xlabel('Number of Days from 16/03/2021')
    plt.ylabel('Number of New Cases per day')
    plt.savefig('./Plots/New_Cases_Per_Day.png')

    plt.figure(2)
    plt.plot(s_open_beta, label = 'Open Loop - beta' )
    plt.plot(s_open_beta_two_third, label = 'Open Loop - beta * 2/3' )
    plt.plot(s_open_beta_one_third, label = 'Open Loop - beta * 1/3' )
    plt.plot(s_open_beta_half, label = 'Open Loop - beta * 1/2' )
    plt.plot(s_close , label = 'Closed Loop' )
    plt.axvline(x= 43 , ymin=0, ymax = 1, linewidth=0.5, color='k', linestyle='dashed')
    plt.legend()
    plt.title('Open and Closed Loop Predictions till 31/12/2021')
    plt.xlabel('Number of Days from 16/03/2021')
    plt.ylabel('Fraction of the Susceptible Population')
    plt.savefig('./Plots/Suspectible.png')
