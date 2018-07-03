import numpy as np
from collections import Counter
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn import svm
import matplotlib.pyplot as plt
import glob

np.set_printoptions(linewidth=150, precision=3)

data_dir = 'methane-data'

reactions = []
numerators = []
denominators = []

rxns = {}

feature_reacs = []

training_x = []
training_y = []
testing_x = []
testing_y = []

for reaction_file, numer_file, denom_file in zip(sorted(glob.iglob(data_dir + '/reactdict_NPT*.txt')), sorted(glob.iglob(data_dir + '/numer*.txt')), sorted(glob.iglob(data_dir + '/denom*.txt'))):
    print(reaction_file, numer_file, denom_file)
    with open(reaction_file, 'r') as file: reactions += [reaction.split(': ')[1] for reaction in file.read().split('\n')[:-1]]
    with open(numer_file, 'r') as file: numerators = np.concatenate([numerators, np.fromstring(file.read(), dtype=np.float64, sep='\n')])
    with open(denom_file, 'r') as file: denominators = np.concatenate([denominators, np.fromstring(file.read(), dtype=np.float64, sep='\n')])
    print(len(reactions), len(numerators), len(denominators))

print('----------------------------------------')

counts = Counter(reactions)
for rxn in reactions:
    rxns[rxn] = [0, 0]
for index, rxn in enumerate(reactions):
    if (counts[rxn] == 6) and numerators[index] != 0.0 and denominators[index] != 0.0:
        rxns[rxn][0] += numerators[index]
        rxns[rxn][1] += denominators[index]

for rxn, rate in rxns.items():
    if rate[0] > 0 and rate[1] > 0:
        # print(sum(rate[0]), sum(rate[1]))
        rate = (rate[0] / rate[1]) / 0.012
        rxns[rxn] = rate
        # print(rxn)
        # print(rate)

with open('features.out', 'r') as file: features = [x.split('\n') for x in file.read().split('----------------------------------------') if x != '']
for feature in features:
    frame = feature[1].split(': ')[1]
    rxn = feature[2].split(': ')[1][:-2]
    # print(frame)
    if rxn in rxns:
        if feature_reacs.count(rxn) != 0:
            print('Not unique')
            continue
        feature_reacs.append(rxn)
        feats = [[int(y) for y in x.split(' ') if y != ''] for x in feature[4:8]]
        rate = rxns[rxn]
        print(rxn)
        # print(rate)
        # for row in feats:
        #     for feat in row:
        #         print(str(feat).rjust(3), end='')
        #     print()
        feats = np.concatenate(feats).ravel()
        print(feats)
        if len(training_x) < 10:
            training_x.append(feats)
            training_y.append(rate)
        else:
            testing_x.append(feats)
            testing_y.append(rate)
    else:
        print('Not found in reaction dictionary')
    # print(feats)

training_x = np.asarray(training_x)
training_y = np.asarray(training_y)
testing_x = np.asarray(testing_x)
testing_y = np.asarray(testing_y)

print()

print(len(training_x))
print(len(testing_x))

# print(training_x[0], training_y[0])

regressor = MLPRegressor( # lbfgs/adam
    hidden_layer_sizes=(1000,), activation='relu', solver='lbfgs', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = regressor.fit(training_x, training_y)

# regressor = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
# n = regressor.fit(training_x, training_y)

# dropout

in_sample = regressor.predict(training_x)
out_of_sample = regressor.predict(testing_x)

print()

print('In-sample Actual:')
print(training_y)
print('In-sample Predicted:')
print(in_sample)
print('In-sample Score:')
print(regressor.score(training_x, training_y))
print(mean_squared_error(training_y, in_sample))

print()

print('Out-of-sample Actual:')
print(testing_y)
print('Out-of-sample Predicted:')
print(out_of_sample)
print('Out-of-sample Score:')
print(regressor.score(testing_x, testing_y))
print(mean_squared_error(testing_y, out_of_sample))
