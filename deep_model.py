from datetime import datetime,timedelta
import pandas as pd
import TestRunConfig
import PKDroneProperties
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import json
import time
from math import ceil
import tensorflow as tf
from tensorflow import keras


def remove_hours(samp):
    if "mins" in samp:
        return (int(float(samp.strip(" mins")))/10 + 1) *10
    elif "hours" in samp:
        return (int((float(samp.strip(" hours")) *60))/10 + 1) *10
    else:
        return 0


def remove_old_tasks(samp):
    six_months_ago = datetime.now() - timedelta(days=180)
    if samp is not None:
        try:
            if datetime.fromtimestamp(float(samp)) < six_months_ago:
                return 1
        except Exception as e:
            print samp,e
    return 0


def get_testnames(test,task):
    if test in task:
        return 0
    return 1


def get_branches(branch,task):
    if branch in task:
        return 0
    return 1


def get_machines(machine,task):
    if machine in task:
        return 0
    return 1


def accuracy2(calculated_output, actual_output):
    if (calculated_output <= 2 * actual_output) and (calculated_output >= 0.5 * actual_output):
        return 1
    else:
        return 0


def train_model(path):

    tm = pd.read_csv(path)
    print "Length of tm_csv initial: " + str(len(tm))

    tm = tm[tm['resultcode']!='fault']
    print "Resultcode not fault: " + str(len(tm))

    tm = tm[tm['resultcode']!='Timeout']
    print "Resultcode not Timeout: " + str(len(tm))

    tm = tm[tm['failuretype']!='Automation Fault']
    print "FailureType not AutomationFault: " + str(len(tm))

    tm['time_taken'] = tm['timestarttofinishstring'].apply(remove_hours)
    tm = tm[tm['remotemachine']!=None]
    print "If no Remote Machine: " + str(len(tm))

    tm = tm[tm['time_taken']<=600]
    print "After removing time more than 600: " + str(len(tm))

    tm = tm[tm['starttime']!=None]
    print "Start time not equal to None: " + str(len(tm))

    tm = tm[tm['time_taken']!= 0]
    print "Time taken not equal to zero: " + str(len(tm))

    # Remove in future for test cases which have no successful tests
    tm = tm[tm['resultcode'] != 'failure']
    print "Removing failed test cases: " + str(len(tm))

    # tm['latest'] = tm['starttime'].apply(remove_old_tasks)
    # tm = tm[tm['latest'] == 1]
    # print len(tm)

    tm.drop(['magicid','failuretype', 'remotehostname', 'starttime', 'time_of_last_progress','finishtime','submittimestring','starttimestring','shortsubmittimestring','shortstarttimestring','finishtimestring','shortfinishtimestring','timestarttofinishstring','timestarttofollowonfinishstring','lastfollowonfinishedtime'],axis=1,inplace=True)
    tm['branch'] = tm['name'].apply(lambda x: x.split(' -- ')[0])
    tm['testname'] = tm['name'].apply(lambda x: x.split(' -- ')[1])

    # print tm

    testnames = []
    for i in tm.testname.values:
        if i not in testnames:
            testnames.append(i)
    branches = []
    for i in tm.branch.values:
        if i not in branches:
            branches.append(i)
    machines = []
    for i in tm.remotemachine.values:
        if i not in machines:
            machines.append(i)
    print testnames
    print branches
    print machines
    for test in testnames:
        tm[test] = tm['testname'].apply(lambda x: int(test in x))
    for branch in branches:
        tm[branch] = tm['branch'].apply(lambda x: int(branch in x))
    for machine in machines:
        tm[machine] = tm['remotemachine'].apply(lambda x: int(machine in x))

    tm['success'] = tm['resultcode'].apply(lambda x:int('success' in x))
    tm['failure'] = tm['resultcode'].apply(lambda x: int('failure' in x))
    tm['win32'] = tm['plat'].apply(lambda x: int('win32' in x))
    tm['darwin'] = tm['plat'].apply(lambda x: int('darwin' in x))
    tm['linux2'] = tm['plat'].apply(lambda x: int('linux2' in x))
    # print tm
    tm.to_csv('out.csv')

    features = []
    for i in testnames:
        features.append(i)
    for i in branches:
        features.append(i)
    for i in machines:
        features.append(i)
    # features.append('success')
    # features.append('failure')
    features.append('win32')
    features.append('darwin')
    features.append('linux2')

    print "Length of features:" + str(len(features))
    x_data = tm[features]
    x_data.to_csv('features.csv')
    y_data = tm['time_taken']
    print y_data.shape

    x_data = x_data.values.reshape(len(x_data), len(features))
    y_data = y_data.values.reshape(len(y_data), 1)

    time_map = {
         "0": 0,
        "10": 1,
        "20": 2,
        "30": 3,
        "40": 4,
        "50": 5,
        "60": 6,
        "70": 7,
        "80": 8,
        "90": 9,
        "100": 10,
        "110": 11,
        "120": 12,
        "130": 13,
        "140": 14,
        "150": 15,
        "160": 16,
        "170": 17,
        "180": 18,
    }

    for i in range(len(y_data)):
        if y_data[i][0] >=0 and y_data[i][0] <= 180:
            new_value = time_map[str(y_data[i][0])]
            y_data[i][0] = new_value
        else:
            y_data[i][0] = 19

    print y_data
    print y_data.shape

    model = keras.Sequential([
        keras.layers.Dense(110, activation=tf.nn.relu),
        keras.layers.Dense(40, activation=tf.nn.relu),
        keras.layers.Dense(20, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_data, y_data, epochs=3)

    layer0 = model.layers[0]
    weights = layer0.get_weights()[0]

    predictions = model.predict(x_data)

    reverse_array = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]

    accuracy = 0
    for i in range(len(y_data)):

        Actual = reverse_array[y_data[i][0]]
        Calculated = reverse_array[np.argmax(predictions[i])]

        accuracy += accuracy2(Calculated, Actual)

        print str(i) + ". Actual:" + str(Actual) + "\t Calculated:" + str(Calculated)

    accuracy = float(accuracy)/len(y_data)
    print "Accuracy = " + str(accuracy)

    return model, features


def predict_data(model, features):
    data = pd.DataFrame(np.zeros(shape=(1,len(features))).astype(int),columns=features)
    print data
    branches = TestRunConfig.TESTRUN_DATA.keys()
    platforms = ['win32', 'darwin', 'linux2']
    machines = PKDroneProperties.drone_properties_info.keys()
    testnames=[]
    for branch in branches:
        for platform in TestRunConfig.TESTRUN_DATA[branch]['platforms']:
            tests = TestRunConfig.TESTRUN_DATA[branch]['platforms'][platform]
            for test in tests:
                if test not in testnames:
                    testnames.append(test)

    '''MainSRM = ['tm-mac-GPU-1', 'NO1SWU930', 'rhel7-tm1', 'no1010041188173', 'no1010040068104', 'tm-win-GPU-2',
               'no1010040066138', 'CTUSER-W7-1', 'mac109-feat', 'ctnoi-mac-311', 'NO1SWU929', 'kasingh-mac-1',
               'tm-win-GPU-1', 'rhel7-tm2', 'no1010040068115', 'NO1SWU963', 'NO1SWU583', 'no1010040068159',
               'no1010040068156', 'no1010041188161', 'NO1SWU520', 'ctnoi-mac-313']
    Green = ['no1010040068085', 'tm-mac-GPU-1', 'NO1SWU931', 'no1010040068102', 'no1010040068100', 'no1010040066139',
             'no1010040066234', 'tm-win-GPU-2', 'no1010040066138', 'CTUSER-W7-1', 'no1swu1253', 'rhel7-tm3',
             'ctnoi-mac-311', 'kasingh-mac-1', 'tm-win-GPU-1', 'NO1SWU962', 'no1010040068056', 'ctnoi-mac-313',
             'rhel7-tm5']
    heather = ['rhel7-tm4', 'NO1SWU932', 'no1010041188174', 'no1010040066237', 'rhel7-vm1', 'no1swu1252', 'NO1SWU1305',
               'NO1SWU1304']
    heather_acrobat = ['rhel7-tm4', 'NO1SWU932', 'no1010041188174', 'no1010040066237', 'rhel7-vm1', 'no1swu1252',
                       'NO1SWU1305', 'NO1SWU1304']
    zeppelin = []
    freesia = ['mac107-2-vm1', 'NO1SWU1303', 'mac109-feat', 'no1slu145', 'no1slu166', 'no1slu482', 'mac107-2',
               'NO1SWU1170', 'NO1SWU396', 'NO1SWU521', 'no1slu213']
    xanadu = ['mac106-vm3', 'mac106-vm4', 'tm2farm-linux64-2', 'tm2farm-linux1']
    jasmine = ['no1010040068084', 'no1010040066150', 'no1010040068230', 'NO1SWU1191', 'no1010041188172', 'NO1SWU1171']
    main_cc = ['no1010040068051', 'win764-5', 'win764-6']'''

    print testnames
    branch_test_platform_machine_dict = {}
    for branch in branches:
        branch_test_platform_machine_dict[branch] = {}
        for test in testnames:
            branch_test_platform_machine_dict[branch][test]={}
            for platform in platforms:
                branch_test_platform_machine_dict[branch][test][platform]={}
                for machine in machines:
                    data = pd.DataFrame(np.zeros(shape=(1, len(features))).astype(int), columns=features)
                    if branch in data:
                        data[branch] = 1
                    if platform in data:
                        data[platform] = 1
                    if machine in data:
                        data[machine] = 1
                    if test in data:
                        data[test] = 1

                    prediction = model.predict(data)

                    reverse_array = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
                                     180]

                    prediction = reverse_array[np.argmax(prediction)]

                    branch_test_platform_machine_dict[branch][test][platform][machine] = prediction
                    # print branch, test, platform, machine, branch_test_platform_machine_dict[branch][test][platform][machine]

    sst_ml = open('MLSpecifiedStaleTime.py', 'wb')
    sst_ml.write('LAST_UPDATED="' + time.strftime("%d/%m/%Y") + " " + time.strftime("%H:%M:%S")+'"\n')
    sst_ml.write("BRANCH_TEST_PLATFORM_MACHINE_DICT=")
    sst_ml.write(json.dumps(branch_test_platform_machine_dict, indent=4))
    sst_ml.close()


def main():
    print sys.argv
    #print np.zeros(shape=(1, 177)).astype(int)
    model, features = train_model('tmdata.csv')
    predict_data(model, features)


if __name__=='__main__':
    main()
