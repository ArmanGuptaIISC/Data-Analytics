import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

df = pd.read_csv('../data/04_cricket_1999to2011.csv')
df = df[df.Innings == 1]

df_new = df.copy()[['Match' ,'Over' ,'Runs','Total.Runs', 'Innings.Total.Runs','Runs.Remaining' ,'Wickets.in.Hand','Innings.Total.Out','Total.Overs']]

### Preprocessing
# General Step : 
# 1 . Computing Over_Used_in_Future which indicates the number of remaining overs.
# 2. Adding a row for each match which indicates the 0th over i.e. starting of the inning.


# **********  Preprocessing 1 *************
## Innings total run (col: Innings.Total.Runs) must be consistent with the total runs 
# (col: Total.Runs ) at the end of the inning. If that is not the case, we need to correct
# the Total.Runs and Innings.Total.Runs columns to make these consistent

# Solution : When we have above incosistency, we are using runs (col : Runs) to calculate
# Total.Runs and Innings.Total.Runs columns, hence we are getting consistent Total.Runs and
# Innings.Total.Runs at the end of below preprocessing. 

# Note : Above solution will be applied if the sum of all the runs (col : runs) atleast matches with 
# either Total.Runs at the end of the innings or Innings.Total.Runs. As a consequence , we are
# deleting 10 matches as these matches' data are completely inconsistent.

# Matches Deleted : 65200, 64793, 66351, 64933, 64940, 217978, 267386, 385023, 424849, 506207

clean_df = pd.DataFrame()
delete_matches = []
for match_id in df_new['Match'].unique():
    temp =df_new.copy()[ df_new.Match == match_id]

    runs_sum = temp['Runs'].sum()
    total_runs_max = temp['Total.Runs'].max()
    innings_total_runs = temp['Innings.Total.Runs'].max()

    if total_runs_max != innings_total_runs:
        if runs_sum == total_runs_max or runs_sum == innings_total_runs:
            l = temp.shape[0]
            temp['Innings.Total.Runs'] = np.full(l, runs_sum , dtype =int)
            temp['Total.Runs'] = np.array([temp['Runs'][:i].sum() for i in range(1,l+1)])
            temp['Runs.Remaining'] = temp['Innings.Total.Runs'] - temp['Total.Runs']
        else:
            delete_matches.append(match_id)
    
    if match_id not in delete_matches:
        innings_total_runs = temp['Innings.Total.Runs'].max()
        innings_total_outs = temp['Innings.Total.Out'].max()
        temp.loc[len(temp.index)] = [match_id,0,0,0,innings_total_runs,innings_total_runs,10,innings_total_outs,50]
        Overs_Used_In_Future = temp['Total.Overs'] - temp['Over']
        temp['Overs_Used_In_Future'] = Overs_Used_In_Future
        temp = temp.sort_values('Over')
        clean_df = pd.concat([clean_df, temp])

# **********  Preprocessing 2 *************

## Since some matches are incomplete i.e. overs played are less than 50 and total outs 
# in inning is < 10. The below function gives the list of matches that are incomplete
incomplete_matches = []
for match_id in df_new['Match'].unique():
    temp =df_new[df_new.Match == match_id]
    overs_max = temp.Over.max()
    wicket_max = temp['Innings.Total.Out'].max()
    if overs_max == 50 or wicket_max == 10:
        pass
    else:
        incomplete_matches.append(match_id)
incomplete_matches

## We are eliminating the incomplete matches from the above preprocessed data and calling 
# resulting data frame as complete_clean_df

complete_clean_df = clean_df[clean_df.Match.isin(set(clean_df.Match).difference(incomplete_matches))]


# *********** Preprocessing 3 *************
## Considering Run as the correct column and deriving other columns from it. The reason for
# considering this option as I have observed that around 340 matches' data are incosistent.
# The Total.Runs and Innings.Total.Runs are not consistent with the Run Column which ulimately
# affecting the Remaining.Runs columns which we have to use in our model.

run_clean_df = pd.DataFrame()

for match_id in df_new['Match'].unique():
    temp =df_new.copy()[ df_new.Match == match_id]

    runs_sum = temp['Runs'].sum()

    l = temp.shape[0]
    temp['Innings.Total.Runs'] = np.full(l, runs_sum , dtype =int)
    temp['Total.Runs'] = np.array([temp['Runs'][:i].sum() for i in range(1,l+1)])
    temp['Runs.Remaining'] = temp['Innings.Total.Runs'] - temp['Total.Runs']
        
    innings_total_runs = temp['Innings.Total.Runs'].max()
    innings_total_outs = temp['Innings.Total.Out'].max()
    temp.loc[len(temp.index)] = [match_id,0,0,0,innings_total_runs,innings_total_runs,10,innings_total_outs,50]
    Overs_Used_In_Future = temp['Total.Overs'] - temp['Over']
    temp['Overs_Used_In_Future'] = Overs_Used_In_Future
    temp = temp.sort_values('Over')
    run_clean_df = pd.concat([run_clean_df, temp])


## This function return the dataframe for particular value of Wickets.in.Hands.
# The data frame contains columns  'Overs_Used_In_Future' that I have computed earlier
# as mentioned above and 'Runs.Remaining' which contains mean of all the values in earlier
# 'Runs.Remaining' column for a particular value of  'Overs_Used_In_Future' and 
# 'Wickets.in.Hands'. 

def getDFbyWickets(data , rem_wickets = 10):
    data_10 = data.loc[data['Wickets.in.Hand'] == rem_wickets]
    data_10 = data_10.sort_values('Overs_Used_In_Future').reset_index(drop=True)
    data_10 = data_10[['Overs_Used_In_Future' ,'Runs.Remaining']]
    data_10 = data_10.groupby('Overs_Used_In_Future').mean().reset_index()
    # data_10.loc[len(data_10.index)] = [50 , 0]
    return data_10

## Optimization Function using Scipy to optimize the parameters
# args : list of dataframes index by the wickets remaining.
def optimizer(df_wic): 
        def loss_fn(parameters):
                loss =0 
                for i in range(10):
                        x = df_wic[i]['Overs_Used_In_Future'].to_numpy()
                        y = df_wic[i]['Runs.Remaining'].to_numpy()
                        pred = parameters[i] * (1 - np.exp(-parameters[10] * x / parameters[i]))
                        loss += ((pred - y) ** 2).sum()
                return loss

        parameters = [None for i in range(11)]
        for i in range(10): # Z
                x = df_wic[i]['Overs_Used_In_Future'].to_numpy()
                y = df_wic[i]['Runs.Remaining'].to_numpy()
                parameters[i] = np.mean(y)
                # parameters
                print(f'Z init val for wicket {i+1} is {parameters[i]}')

        parameters[10] = 10 # L 

        optim = minimize(loss_fn , parameters , method='BFGS' )
        new_parameters = optim.x
        loss = loss_fn(new_parameters)
        return new_parameters , loss

# ******************  Using Data afte Preprocessing 1 **************************
## Optimising the parameters using clean_df which is observed at Preprocessing 1.
print('\n Using Data after Preprocessing 1 \n')

# Filtering required Columns
data = clean_df[['Overs_Used_In_Future' ,'Wickets.in.Hand' , 'Runs.Remaining']]

# Creating the list of dataframes index by (wickets-1) using getDfByWickets function
# And plotting the preprocessed data.
df_wic = []
fig = plt.figure()
for i in range(1,11):
    data_10 = getDFbyWickets(data,i)
    plt.scatter(data_10['Overs_Used_In_Future'] , data_10['Runs.Remaining'],label = i)
    df_wic.append(data_10)

plt.xlabel('Overs Remaining')
plt.ylabel('Average Run Achieved')
plt.title('Data Points after Preprocessing 1')
plt.legend()
fig.savefig('./1.1.png')

#optimising parameters
parameters,loss = optimizer(df_wic)

#Plotting Graph using optimised parameters
fig = plt.figure()
total = 0
for i in range(9, -1,-1):
    x_plot = np.arange(0,51)
    y_plot = parameters[i] * (1 - np.exp(-parameters[10] * x_plot / parameters[i]))
    plt.plot( x_plot , y_plot,label = i+1)
    
    total += len(df_wic[i])

plt.xlabel('Overs Remaining')
plt.ylabel('Average Run Achievable')
plt.title('Curve Plot using Optimised parameters from data after Preprocessing 1 ')
plt.legend()
fig.savefig('./1.2.png')

print('Parameters are :')
print('Z : ', parameters[:10])
print('Slope :', parameters[10:])
print('Normalised Squared Error :',loss / total)



# ******************  Using Data afte Preprocessing 2 **************************
## Optimising the parameters using clean_df which is observed at Preprocessing 2.
print('\n Using Data after Preprocessing 2 \n')

# Filtering required Columns
com_data = complete_clean_df[['Overs_Used_In_Future' ,'Wickets.in.Hand' , 'Runs.Remaining']]
fig = plt.figure()

# Creating the list of dataframes index by (wickets-1) using getDfByWickets function
# And plotting the preprocessed data.
com_df_wic = []
for i in range(1,11):
    data_10 = getDFbyWickets(com_data,i)
    plt.scatter(data_10['Overs_Used_In_Future'] , data_10['Runs.Remaining'],label = i)
    com_df_wic.append(data_10)

plt.xlabel('Overs Remaining')
plt.ylabel('Average Run Achieved')
plt.title('Data Points after Preprocessing 2')
plt.legend()
fig.savefig('./2.1.png')

#optimising parameters
com_parameters,com_loss = optimizer(com_df_wic)
fig = plt.figure()
#Plotting Graph using optimised parameters
com_total = 0
for i in range(9, -1,-1):
    x_plot = np.arange(0,51)
    y_plot = com_parameters[i] * (1 - np.exp(-com_parameters[10] * x_plot / com_parameters[i]))
    plt.plot( x_plot , y_plot,label = i+1)
    
    com_total += len(com_df_wic[i])

plt.xlabel('Overs Remaining')
plt.ylabel('Average Run Achievable')
plt.title('Curve Plot using Optimised parameters from data after Preprocessing 2 ')
plt.legend()
fig.savefig('./2.2.png')


print('Parameters are :')
print('Z : ', com_parameters[:10])
print('Slope :', com_parameters[10:])
print('Normalised Squared Error :',com_loss / com_total)




# ******************  Using Data afte Preprocessing 3 **************************
## Optimising the parameters using clean_df which is observed at Preprocessing 3.
print('\n Using Data after Preprocessing 3 \n')

# Filtering required Columns
run_data = run_clean_df[['Overs_Used_In_Future' ,'Wickets.in.Hand' , 'Runs.Remaining']]

# Creating the list of dataframes index by (wickets-1) using getDfByWickets function
# And plotting the preprocessed data.
fig = plt.figure()
run_df_wic = []
for i in range(1,11):
    data_10 = getDFbyWickets(run_data,i)
    plt.scatter(data_10['Overs_Used_In_Future'] , data_10['Runs.Remaining'],label = i)
    run_df_wic.append(data_10)

plt.xlabel('Overs Remaining')
plt.ylabel('Average Run Achieved')
plt.title('Data Points after Preprocessing 3')
plt.legend()
fig.savefig('./3.1.png')

#optimising parameters
run_parameters,run_loss = optimizer(run_df_wic)

#Plotting Graph using optimised parameters
run_total = 0
fig = plt.figure()

for i in range(9, -1,-1):
    x_plot = np.arange(0,51)
    y_plot = run_parameters[i] * (1 - np.exp(-run_parameters[10] * x_plot / run_parameters[i]))
    plt.plot( x_plot , y_plot,label = i+1)
    
    run_total += len(run_df_wic[i])

plt.xlabel('Overs Remaining')
plt.ylabel('Average Run Achievable')
plt.title('Curve Plot using Optimised parameters from data after Preprocessing 3 ')
plt.legend()
fig.savefig('./3.2.png')

print('Parameters are :')
print('Z : ', run_parameters[:10])
print('Slope :', run_parameters[10:])
print('Normalised Squared Error :',run_loss / run_total)