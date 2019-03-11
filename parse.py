import numpy as np
from collections import Counter
from itertools import groupby
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn import svm
import matplotlib.pyplot as plt
# from lifetimes import get_lifetimes
import os
import glob
import shutil

np.set_printoptions(linewidth=150, suppress=True, precision=6)

feature_file = 'molanal.out' # file containing features
data_dir = 'methane-data' # directory in which MD data is stored
xyz_dir = 'xyz' # directory in which to store 3D XYZ files
min_occurences = 2 # minimum occurences in reaction dictionaries to be considered
log_rates = True # whether to log rates
normalize_rates = True # whether to normalize rates between 0 and 1
train_test_split = 0.8 # percentage of data to be used for testing
eliminate_dup_feats = False # ignore reactions with the same feature set
# high_cutoff = 0.5
# low_cutoff = 0

reactions = []
numerators = []
denominators = []

rxns = {}

feature_reacs = []
feature_feats = []

all_data = []

training_x = []
training_y = []
testing_x = []
testing_y = []

min_rate = None
max_rate = None

total_rxns = 0
not_eligible = 0
unique_rxn_count = 0

testing_reacs = []

# lifetimes = get_lifetimes()

def normalize(arr):
    arr = np.asarray(arr)
    if log_rates:
        arr = np.log10(arr)
    if normalize_rates:
        arr -= min_rate
        arr /= (max_rate - min_rate)
    return arr

def denormalize(arr):
    if normalize_rates:
        arr *= (max_rate - min_rate)
        arr += min_rate
    if log_rates:
        arr = np.power(10, arr)
    return arr

for reaction_file, numer_file, denom_file in zip(sorted(glob.iglob(data_dir + '/reactdict_NPT*.txt')), sorted(glob.iglob(data_dir + '/numer*.txt')), sorted(glob.iglob(data_dir + '/denom*.txt'))):
    print(reaction_file, numer_file, denom_file)
    with open(reaction_file, 'r') as file: reactions += [" ".join(reaction.split(': ')[1].strip().split()) for reaction in file.read().split('\n')[:-1]]
    with open(numer_file, 'r') as file: numerators = np.concatenate([numerators, np.fromstring(file.read(), dtype=np.float64, sep='\n')])
    with open(denom_file, 'r') as file: denominators = np.concatenate([denominators, np.fromstring(file.read(), dtype=np.float64, sep='\n')])
    print(len(reactions), len(numerators), len(denominators))

print('----------------------------------------')

counts = Counter(reactions)
for rxn in reactions:
    rxns[rxn] = [0, 0]
    # print(rxn)
for index, rxn in enumerate(reactions):
    if (counts[rxn] >= min_occurences) and numerators[index] != 0.0 and denominators[index] != 0.0:
        rxns[rxn][0] += numerators[index]
        rxns[rxn][1] += denominators[index]

for rxn in list(rxns):
    rate = rxns[rxn]
    if rate[0] == 0 or rate[1] == 0:
        del rxns[rxn]

for rxn, rate in rxns.items():
    # print(sum(rate[0]), sum(rate[1]))
    rate = (rate[0] / rate[1]) / 0.012
    rxns[rxn] = rate
    # print(rxn)
    # print(rate)

to_delete = glob.glob(xyz_dir + '/*')
for f in to_delete:
    shutil.rmtree(f)

with open(feature_file, 'r') as file: features = [x.split('\n') for x in file.read().split('----------------------------------------') if x != '']
for feature in features:
    # print(feature)
    rxn = feature[2].split(': ')[1].strip()
    feats_split = [[int(y) for y in x.split(' ') if y != ''] for x in feature[4:8]]
    # feats = np.delete(feats, 8, 1)
    # for f_idx in range(len(feats)):
        # feats[f_idx][4] = abs(feats[f_idx][4])
    feats = np.concatenate(feats_split)
    if feature_reacs.count(rxn) != 0 or (eliminate_dup_feats and any((feats == x).all() for x in feature_feats)):
        # print('Not unique')
        continue
    unique_rxn_count += 1
    if rxn in rxns:
        feature_reacs.append(rxn)
        feature_feats.append(feats)

        frame = feature[1].split(': ')[1]
        rate = rxns[rxn]

        # reaction_sides = np.concatenate([x.split(' + ') for x in rxn.split(' => ')])
        # print(feats_split)
        # for index, feat in enumerate(feats_split):
        #     idx = 2 if index > 2 else index
        #     # print(feats_split[idx][:5])
        #     feats_split[index].insert(5, lifetimes[reaction_sides[idx]])
        # feats = np.concatenate(feats_split)

        feats = np.append(feats, int(feature[8]))
        feats = np.append(feats, int(feature[9]))

        path = xyz_dir + '/rxn_%d' % total_rxns
        os.makedirs(path)

        count = [0, 0]
        raw_coords = feature[11:] # used to be 9:
        sides = list(list(g) for k, g in groupby(raw_coords, key=lambda x: x != '----') if k)
        for s, side in enumerate(sides):
            coord = list(list(g) for k, g in groupby(side, key=lambda x: x != '') if k)
            for mol in coord:
                prefix = 'reac_%d' % count[0] if s == 0 else 'prod_%d' % count[1]
                f = open(path + '/' + prefix + '.xyz', 'w')
                # print next "frame", s
                cur_pos = f.tell()
                num_atoms = 0
                f.write('   \n')
                if (s == 0):
                    f.write('Reactant #%d: ' % count[0])
                    count[0] += 1
                else:
                    f.write('Product #%d: ' % count[1])
                    count[1] += 1
                f.write(mol[0] + '\n')
                for dim in mol[1:]:
                    dim = ' '.join(dim.split())
                    num_atoms += 1
                    f.write(dim + '\n')
                f.seek(cur_pos)
                f.write("%d" % num_atoms)
                f.seek(0, 2)
                f.close()

        print('Reaction #%d added' % total_rxns)
        print(rxn)
        print(feats)
        print(feature[8])
        print(feature[9])

        all_data.append({
            'frame': frame,
            'reaction': rxn,
            'features': feats,
            'rate': rate,
            'index': total_rxns
        })
        total_rxns += 1
        if log_rates:
            logged_rate = np.log10(rate)
        else:
            logged_rate = rate
        if not max_rate or logged_rate > max_rate:
            max_rate = logged_rate
        if not min_rate or logged_rate < min_rate:
            min_rate = logged_rate
    else:
        not_eligible += 1
        # print('Not found in reaction dictionary')
        # print(rxn)

    # print(feats)

for index, data in enumerate(all_data):
    if index < len(all_data) * train_test_split:
        training_x.append(data['features'])
        training_y.append(data['rate'])
        data['data_index'] = len(training_y) - 1
        data['partition'] = 'training'
    else:
        testing_x.append(data['features'])
        testing_y.append(data['rate'])
        testing_reacs.append(data['reaction'])
        data['data_index'] = len(testing_y) - 1
        data['partition'] = 'testing'

scaler = preprocessing.StandardScaler()
scaler.fit(training_x)
# training_x = scaler.transform(training_x)
# testing_x = scaler.transform(testing_x)

all_rates = training_y + testing_y
plt.figure(0)
plt.hist(all_rates, 20, facecolor='green', alpha=0.75)
plt.suptitle('Rate Histogram')

all_rates_normalized = normalize(all_rates)
plt.figure(1)
plt.hist(all_rates_normalized, 50, facecolor='green', alpha=0.75)
plt.suptitle('Log-scale Rate Histogram')

print('-------------------------------------')
print('ReaxFF Data:')
print('Total Reactions:', len(reactions))
print('Reactions Occuring More Than %d Times: %d' % (min_occurences, len(rxns)))
print()

print('Molecular Analyzer Data:')
print('Total Reactions:', len(features))
print('Total Unique Reactions:', unique_rxn_count)
print('Not Eligible (occured less than %d times in the reaction dictionaries): %d' % (min_occurences, not_eligible))
print('Total Usable Reactions:', total_rxns)
print('Training:', len(training_x))
print('Testing:', len(testing_x))
print()

print('Feature Information:')
print('Feature Vector Length:', len(training_x[0]))
print('# Features:', int(len(training_x[0]) / 4))
print()

print('> Normalizing and rescaling rate data')

print('Min Rate:', min_rate)
print('Max Rate:', max_rate)

# to_delete = []
# for index, tr in enumerate(training_y):
#     if (tr > high_cutoff or tr < low_cutoff):
#         to_delete.append(index)
# training_x = np.delete(training_x, to_delete, axis=0)
# training_y = np.delete(training_y, to_delete, axis=0)

# print(len(training_y))
# print(len(testing_y))

# to_delete = []
# for index, tr in enumerate(testing_y):
#     if (tr > high_cutoff or tr < low_cutoff):
#         to_delete.append(index)
# testing_x = np.delete(testing_x, to_delete, axis=0)
# testing_y = np.delete(testing_y, to_delete, axis=0)

training_x = np.asarray(training_x)
# print(testing_y)
training_y = normalize(training_y)
# print(denormalize(np.copy(training_y))[0])

testing_x = np.asarray(testing_x)
testing_y = normalize(testing_y)


print('> Saving data to CSVs')

# Saving data to CSV
np.savetxt('training_x.csv', training_x, delimiter=',')
np.savetxt('training_y.csv', training_y, delimiter=',')
np.savetxt('testing_x.csv', testing_x, delimiter=',')
np.savetxt('testing_y.csv', testing_y, delimiter=',')

# print(training_x[0], training_y[0])

print('> Training model')

# regressor = MLPRegressor( # lbfgs/adam alpha=0.001
#     hidden_layer_sizes=(16,8,), activation='relu', solver='lbfgs', alpha=0.001, batch_size='auto',
#     learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True,
#     random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
#     early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# n = regressor.fit(training_x, training_y)

# regressor = MLPRegressor( # lbfgs/adam alpha=0.001
#     hidden_layer_sizes=(13,13,13,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
#     learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True,
#     random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
#     early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# n = regressor.fit(training_x, training_y)

# regressor = MLPRegressor( # lbfgs/adam alpha=0.001
#     hidden_layer_sizes=(512,256,128,), activation='relu', solver='sgd', alpha=0.001, batch_size='auto',
#     learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True,
#     random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
#     early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# n = regressor.fit(training_x, training_y)

# regressor = linear_model.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 1e2])
# n = regressor.fit(training_x, training_y)

regressor = kernel_ridge.KernelRidge(alpha=1.0)
n = regressor.fit(training_x, training_y)

# regressor = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.00501, kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=True)
# n = regressor.fit(training_x, training_y)

print('> Training complete, testing in-sample and out-of-sample')

in_sample = regressor.predict(training_x)
out_of_sample = regressor.predict(testing_x)

print()

print('In-sample Actual:')
print(training_y)
print('In-sample Predicted:')
print(in_sample)
# print('In-sample Difference:')
# print(abs(training_y - in_sample))
print('In-sample MSE:')
print(mean_squared_error(training_y, in_sample))
print('In-sample R2:')
print(r2_score(training_y, in_sample))

# training_y = denormalize(training_y)
# in_sample = denormalize(in_sample)
plt.figure(2)
plt.scatter(training_y, in_sample)
plt.plot([0,1], [0,1])
plt.suptitle('In-sample Test')
plt.xlabel('Actual Rate')
plt.ylabel('Predicted Rate')

print()

print('Out-of-sample Actual:')
print(testing_y)
print('Out-of-sample Predicted:')
print(out_of_sample)
# print('Out-of-sample Difference:')
# print(abs(testing_y - out_of_sample))
print('Out-of-sample MSE:')
print(mean_squared_error(testing_y, out_of_sample))
print('Out-of-sample R2:')
print(r2_score(testing_y, out_of_sample))
print(r2_score(denormalize(np.copy(testing_y)), denormalize(np.copy(out_of_sample))))

diffs = abs(testing_y - out_of_sample)
# print(testing_reacs)
# print(diffs)
plt.figure(3)
plt.hist(diffs, 60)

# percent_error = (out_of_sample - testing_y)/testing_y * 100
# print('Percent Error Normalized:')
# print(percent_error)

# percent_error_actual = (denormalize(np.copy(out_of_sample)) - denormalize(np.copy(testing_y)))/abs(denormalize(np.copy(testing_y))) * 100
# print('Percent Error Actual:')
# print(percent_error_actual)
# print()
# print(np.average(abs(percent_error_actual)))
# print('Out-of-sample Actual Difference:')
# print(abs(denormalize(np.copy(testing_y)) - denormalize(np.copy(out_of_sample))))
# print('Actual Testing Y')
# print(denormalize(np.copy(testing_y)))

# print(denormalize(np.copy(testing_y))[-2])
# print('Relative Change')
# abs(denormalize(np.copy(testing_y)) - denormalize(np.copy(out_of_sample)))/denormalize(np.copy(testing_y))
# testing_y = denormalize(testing_y)
# out_of_sample = denormalize(out_of_sample)
plt.figure(4)
plt.scatter(testing_y, out_of_sample)
plt.plot([0,1], [0,1])
plt.suptitle('Out-of-sample Test')
plt.xlabel('Actual Rate')
plt.ylabel('Predicted Rate')

print('---------------------------------------------------')

print('Reactions with the same features:')
same_feats = []
for a in all_data:
    for b in all_data:
        reac_set = { a['reaction'], b['reaction'] }
        if a != b and same_feats.count(reac_set) == 0 and np.array_equal(a['features'], b['features']):
            same_feats.append(reac_set)
            print('-----')
            print(a['reaction'])
            print(b['reaction'])
            print(a['features'])
            print(a['rate'])
            print(b['rate'])
            print(a['index'])
            print(b['index'])

print('---------------------------------------------------')

print('Problematic reactions:')

problematic = []

for index, diff in enumerate(diffs):
    if diff > 0.22:
        problematic.append(index)

for d in all_data:
    for f in problematic:
        if d['data_index'] == f and d['partition'] == 'testing':
            print('-----')
            print(d['reaction'])
            print(d['features'])
            print(testing_y[f])
            print(out_of_sample[f])

plt.show(block=True)
