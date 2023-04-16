from tqdm import tqdm
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vars import locationBlackList, fieldBlacklist
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import warnings
from tabulate import tabulate
import time
from threading import Thread

scriptPath = os.getcwd()

def readCsv(dataset, filePath):
    print("Reading dataset from file: ", end='')
    with open(filePath, newline='') as inputFile:
        csvReader = csv.DictReader(inputFile)
        for record in csvReader:
            if record['location'] in locationBlackList:
                continue
            if record['location'] != '':
                dataset.append(record)  
        inputFile.close()
    print("Done")


def writeCsv(dataset, path):
    with open(path, 'w', newline='') as outputFile:
        writer = csv.DictWriter(outputFile, dataset[0].keys())
        writer.writeheader()
        for dict in tqdm(dataset, desc="Writing dataset to file"):
            writer.writerow(dict)
        outputFile.close()


def removeNull(dataset):
    keys = ['total_cases', 'total_deaths', 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
            'new_cases', 'new_deaths', 'population']
    for record in tqdm(dataset, desc="Removing null characters from dataset"):
        for key in keys:
            if record[key] == '':
                record[key] = '0'


def removeZero(dataset):
    lastValue = {'total_cases':'0', 'total_deaths':'0', 'total_vaccinations':'0', 'people_fully_vaccinated':'0'}
    lastLocation = dataset[0]['location']
    for record in tqdm(dataset, desc="Removing unnecessery zeros from dataset"):
        if record['location'] != lastLocation:
            lastLocation = record['location']
            lastValue = {'total_cases':'0', 'total_deaths':'0', 'total_vaccinations':'0', 'people_fully_vaccinated':'0'}
        for key in lastValue:
            if record[key] == '0':
                record[key] = lastValue[key]
            else:
                lastValue[key] = record[key]

def rrse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Relative Squared Error """
    EPSILON = 1e-10
    return np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))) + EPSILON )


def rae(actual: np.ndarray, predicted: np.ndarray):
    """ Relative Absolute Error (aka Approximation Error) """
    EPSILON = 1e-10
    return np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual - np.mean(actual))) + EPSILON)


def predict(locations, dsPath, targets, periods, continent):
    
    dataset = pd.read_csv(dsPath)
    
    for loc in locations:
        
        if loc == None:
            continue
        
        warnings.filterwarnings("ignore")
        
        print(f'\nStarting {loc}...\n')
        
        stime = time.time()
        
        model = RandomForestRegressor(n_estimators=160, criterion='absolute_error')
        train_ds = dataset[(dataset['location'] == loc) & (
            pd.to_datetime(dataset['date']) <= pd.Timestamp(2021, 10, 31))]
        test_ds = dataset[(dataset['location'] == loc) & (pd.to_datetime(dataset['date']).between(
            pd.Timestamp(2021, 11, 1), pd.Timestamp(2021, 12, 1)))]
        
        for target in targets:
            
            X_train = train_ds.drop(target, axis=1)
            X_train = X_train.drop(fieldBlacklist, axis=1)
            X_test = test_ds.drop(target, axis=1)
            X_test = X_test.drop(fieldBlacklist, axis=1)
            y_train = train_ds[target]
            y_test = test_ds[target] 
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            folderPath = f'{scriptPath}/resources/predictions/countries/{loc}/{target}/'
            if continent:
                folderPath = f'{scriptPath}/resources/predictions/continents/{loc}/{target}/'
            
            table = []
            table.append(['Period','Mean Squared Error(MSE)', 'Root Mean Square Error(RMSE)', 
                          'R-Squared', 'Mean Absolute Error(MAE)', 'Relative Absolute Error(RAE)',
                          'Root Relative Squared Error(RRSE)'])
            
            for period in periods:
                
                fig = plt.Figure(figsize=(10.8,7.2))
                fig.clf()
                ax = fig.add_subplot(111)
                
                xticks = test_ds[pd.to_datetime(test_ds['date']) <= pd.Timestamp(2021, 11, period)]['date']
                
                
                ax.plot(xticks, test_ds[pd.to_datetime(test_ds['date']) <= pd.Timestamp(2021, 11, period)][target], label="Actual_value")
                ax.plot(xticks, preds[0:period], label="Predicted", color='red', lw=2, ls='--')
                ax.ticklabel_format(style='plain', axis='y')
                ax.legend([f'Actual', f'Predicted'])
                ax.set_ylabel(target)
                ax.set_xlabel('date')
                ax.set_title(f"{period} day prediction of {target} in {loc}")
                ax.set_xticklabels(xticks, rotation=90)
                fig.tight_layout()
                
                if not os.path.exists(folderPath):
                    os.makedirs(folderPath)       
                fig.savefig(f'{folderPath}{period}.png', bbox_inches='tight')

                plt.close(fig)
                
                MSE = mean_squared_error(y_test[0:period], preds[0:period], squared=True)
                RMSE = mean_squared_error(y_test[0:period], preds[0:period], squared=False)
                R_Squared = r2_score(y_test[0:period], preds[0:period])
                MAE = mean_absolute_error(y_test[0:period], preds[0:period])
                RAE = rae(y_test[0:period], preds[0:period])
                RRSE = rrse(y_test[0:period], preds[0:period])
                
                table.append([period ,MSE, RMSE, R_Squared, MAE, RAE, RRSE])
                
                
            txt = tabulate(table, headers=("firstrow"), tablefmt="grid")
            with open(f'{folderPath}Evaluation_Metrics.txt', 'w') as file:
                file.write(txt)
        
        warnings.resetwarnings()
     
        print(f'\n{loc} Done in {time.time() - stime}\n')
        

def powerset(fullset):
  listsub = list(fullset)
  subsets = []
  for i in range(1,2**len(listsub)):
    subset = []
    for k in range(len(listsub)):            
      if i & 1<<k:
        subset.append(listsub[k])
    subsets.append(subset)        
  return sorted(subsets, key=len)


def find_best_params(locations, dsPath, targets, fields, continent):
    
    dataset = pd.read_csv(dsPath)
    
    subsets = powerset(set(fields))
        
    for loc in locations:
        
        if loc == None:
            continue
        
        warnings.filterwarnings("ignore")
        
        print(f'\nStarting {loc}...\n')
        
        stime = time.time()
        
        model = RandomForestRegressor(n_estimators=160, criterion='absolute_error')

        train_ds = dataset[(dataset['location'] == loc) & (
            pd.to_datetime(dataset['date']) <= pd.Timestamp(2021, 10, 31))]
        test_ds = dataset[(dataset['location'] == loc) & (pd.to_datetime(dataset['date']).between(
            pd.Timestamp(2021, 11, 1), pd.Timestamp(2021, 12, 1)))]
        
        for target in targets:
            
            table = []
            table.append(['Subset','Mean Squared Error(MSE)', 'Root Mean Square Error(RMSE)', 
                          'R-Squared', 'Mean Absolute Error(MAE)', 'Relative Absolute Error(RAE)',
                          'Root Relative Squared Error(RRSE)'])
            
            best_rsq = []
            best_mse = []
            best_rmse = []
            best_mae = []
            best_rae = []
            best_rrse = []

            for subset in subsets:
                
                blackList = []
                for field in fields:
                    if field not in subset and field != target :
                        blackList.append(field)
                
                if len(blackList) == (len(fields) - 1 ):
                    continue
                
                X_train = train_ds.drop(target, axis=1)
                X_train = X_train.drop(fieldBlacklist, axis=1)
                X_train = X_train.drop(blackList, axis=1)
                
                
                X_test = test_ds.drop(target, axis=1)
                X_test = X_test.drop(fieldBlacklist, axis=1)
                X_test = X_test.drop(blackList, axis=1)
                               
                y_train = train_ds[target]
                y_test = test_ds[target] 
               
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                folderPath = f'{scriptPath}/resources/params/countries/{loc}/{target}/'
                if continent:
                    folderPath = f'{scriptPath}/resources/params/continents/{loc}/{target}/'

                
                MSE = mean_squared_error(y_test[0:30], preds[0:30], squared=True)
                RMSE = mean_squared_error(y_test[0:30], preds[0:30], squared=False)
                R_Squared = r2_score(y_test[0:30], preds[0:30])
                MAE = mean_absolute_error(y_test[0:30], preds[0:30])
                RAE = rae(y_test[0:30], preds[0:30])
                RRSE = rrse(y_test[0:30], preds[0:30])

                table.append([subset ,MSE, RMSE, R_Squared, MAE, RAE, RRSE])
                        

                if len(best_mse) == 0:
                    best_mse.append((subset, MSE))
                else:
                    if best_mse[0][1] < MSE:
                        best_mse.clear()
                        best_mse.append((subset, MSE))


                if len(best_rmse) == 0:
                    best_rmse.append((subset, RMSE))
                else:
                    if best_rmse[0][1] < RMSE:
                        best_rmse.clear()
                        best_rmse.append((subset, RMSE))
                        
                
                if len(best_rsq) == 0:
                    best_rsq.append((subset, R_Squared))
                else:
                    if best_rsq[0][1] < R_Squared:
                        best_rsq.clear()
                        best_rsq.append((subset, R_Squared))
                        
                        
                if len(best_mae) == 0:
                    best_mae.append((subset, MAE))
                else:
                    if best_mae[0][1] < MAE:
                        best_mae.clear()
                        best_mae.append((subset, MAE))
                        
                if len(best_rae) == 0:
                    best_rae.append((subset, RAE))
                else:
                    if best_rae[0][1] < RAE:
                        best_rae.clear()
                        best_rae.append((subset, RAE))


                if len(best_rrse) == 0:
                    best_rrse.append((subset, RRSE))
                else:
                    if best_rrse[0][1] < RRSE:
                        best_rrse.clear()
                        best_rrse.append((subset, RRSE))


            if not os.path.exists(folderPath):
                    os.makedirs(folderPath)
                    
            txt = tabulate(table, headers=("firstrow"), tablefmt="grid")
            with open(f'{folderPath}Param_Analysis.txt', 'w') as file:
                file.write(txt)

            txt = tabulate(best_mse, headers=["Best subset", "MSE"], tablefmt="grid")
            with open(f'{folderPath}Best_Params.txt', 'w') as file:
                file.write(txt+f'\n\n')
                
            txt = tabulate(best_rmse, headers=["Best subset", "RMSE"], tablefmt="grid")
            with open(f'{folderPath}Best_Params.txt', 'a') as file:
                file.write(txt+f'\n\n')
                
            txt = tabulate(best_rsq, headers=["Best subset", "R Squared"], tablefmt="grid")
            with open(f'{folderPath}Best_Params.txt', 'a') as file:
                file.write(txt+f'\n\n')
                
            txt = tabulate(best_mae, headers=["Best subset", "MAE"], tablefmt="grid")
            with open(f'{folderPath}Best_Params.txt', 'a') as file:
                file.write(txt+f'\n\n')

            txt = tabulate(best_rae, headers=["Best subset", "RAE"], tablefmt="grid")
            with open(f'{folderPath}Best_Params.txt', 'a') as file:
                file.write(txt+f'\n\n')

            txt = tabulate(best_rrse, headers=["Best subset", "RRSE"], tablefmt="grid")
            with open(f'{folderPath}Best_Params.txt', 'a') as file:
                file.write(txt)                

        warnings.resetwarnings()
     
        print(f'\n{loc} Done in {time.time() - stime}\n')