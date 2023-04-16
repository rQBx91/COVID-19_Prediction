import utils
import vars
import os
from sys import platform
import time
from multiprocessing import Process
import numpy as np

PN = 4

def Main():

# File and script path for windows platform
    if platform == "win64" or platform == "win32":
        script_dir = os.getcwd()# get script execution path
        inputCSV = f'{script_dir}\\resources\\SARS-CoV-2 Dataset Updated.csv' # create absolute file path for input
        preprocessedCSV = f'{script_dir}\\resources\\SARS-CoV-2 Dataset-Preprocessed.csv' # create absolute file path for output


# File and script path for GNU/Linux palatform
    if platform == "linux" or platform == "linux2": # check for platform
        script_dir = os.getcwd() # get script execution path
        inputCSV = f'{script_dir}/resources/SARS-CoV-2 Dataset Updated.csv' # create absolute file path for input
        preprocessedCSV = f'{script_dir}/resources/SARS-CoV-2 Dataset-Preprocessed.csv' # create absolute file path for output


# Question 1
    dataset = []
    utils.readCsv(dataset, inputCSV) # read the dataset from specified csv file
    utils.removeNull(dataset) # replace all the null cells with zero
    utils.removeZero(dataset) # replace all the unnecessary zeros with appropriate values
    utils.writeCsv(dataset, preprocessedCSV)
 
    
# Question 2

    # Run configs
    allFields = ['total_cases', 'total_deaths', 'total_vaccinations','people_vaccinated',
                 'people_fully_vaccinated', 'new_cases', 'new_deaths', 'population']
    targets=['new_cases', 'new_deaths', 'total_cases', 'total_deaths', 'people_fully_vaccinated', 'total_vaccinations']
    periods=[7, 14, 21, 28, 30]
    locationList = vars.continentList
    continent = True
    
    # Run the predictions with a single process
    #print(f'\nStarting Predictions...\n')
    #utils.predict_worker(locationList, preprocessedCSV, targets, periods, continent)

    # Run the prediction with multiprocessing
    task_list = np.array_split(locationList, PN)
    processes = []
    print(f'\nStarting Predictions...\n')
    for i in range(PN):
        p = Process(target=utils.predict, args=(task_list[i], preprocessedCSV, targets, periods, continent))
        processes.append(p)
    for p in processes:
        p.start()
        time.sleep(2)
    for p in processes:
        p.join()
    

# Questions 3

    # Run param analisis with a single process
    #print(f'\nStarting Param Analisis...\n')
    #utils.find_best_params(locationList, preprocessedCSV, targets, allFields, continent)
    
    # Run param analisis with multiprocessing
    task_list = np.array_split(locationList, PN)
    processes = []
    print(f'\nStarting Param Analisis...\n')
    for i in range(PN):
        p = Process(target=utils.find_best_params, args=(task_list[i], preprocessedCSV, targets, allFields, continent))
        processes.append(p)
    for p in processes:
        p.start()
        time.sleep(2)
    for p in processes:
        p.join()

    
if __name__ == "__main__":
    stime = time.time()
    Main()
    print("\nscript execution time: {0}\n".format(time.time() - stime) )