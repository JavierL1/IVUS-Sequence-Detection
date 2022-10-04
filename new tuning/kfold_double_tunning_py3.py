from scipy.ndimage.filters import gaussian_filter1d
import os, sys
import numpy as np
from UBMetrics import compute_PRAF
import pickle
# import cPickle
from tqdm import tqdm

# Main folder
# Usualy 'CNN{net}/Predictions/CNN{net}'
main_dir = sys.argv[1]
base_result_filename = sys.argv[2]#e.g. "CNN1a/Tuning/"
subset = int(sys.argv[3]) #0:train, 1:extra, 2:test, other:all
mode_tunning = int(sys.argv[4]) #0:mean among folds, other:median among folds

if not os.path.exists(base_result_filename):
	os.mkdir(base_result_filename)

# Ids of folds [1,10]
names_folds = [str(i) for i in range(1,11)]
subset_name = ''
# Sets predictions on wich base optimization
if subset == 0:
	list_subdirs_to_scan = [name_fold+'/CSV/TRAIN_CNN' for name_fold in names_folds]
	subset_name = 'TRAIN_CNN'
elif subset == 1:
	list_subdirs_to_scan = [name_fold+'/CSV/EXTRAIN' for name_fold in names_folds]
	subset_name = 'EXTRAIN'
elif subset == 2:
	list_subdirs_to_scan = [name_fold+'/CSV/TEST' for name_fold in names_folds]
	subset_name = 'TEST'
else:	
	list_subdirs_to_scan = [name_fold+'/CSV' for name_fold in names_folds]

# Sets constants
# \theta: threshold to determine wheter a value 'goes' to 1 or 0
theta_min = 0.1
theta_max = 0.9
# \sigma: smoothing
sigma_min = 0.1
sigma_max = 8
# Density of grid
nvals = 100

# Creates grid of values
candidates_theta = np.linspace(theta_min,theta_max,nvals)
candidates_sigma = np.linspace(sigma_min,sigma_max,nvals)
n_total_exps = int(len(candidates_sigma)*len(candidates_theta))

def tune_for_one_fold(files_fold, fold_description='None'):
	# Example: 'Tuning for FOLD 1/CSV/TRAIN'
	print("Tuning for FOLD %s"%fold_description)
	# Result matrix
	# Each row corresponds to a theta-sigma pair value
	results = np.zeros((n_total_exps,18))
	params = []
	# Amount of pullbacks in set
	n_cases = len(files_fold)
	all_data = []

	# For every csv pullback
	for f in files_fold: 
		# Loads data from all CSVs \hat{y}, y_real
		all_data.append(np.loadtxt(f,delimiter=','))

	global_exp_counter = 0

	# For every theta_i
	for exp_counter_1 in tqdm(range(len(candidates_theta))):
		# For every sigma_i
		for exp_counter_2 in tqdm(range(len(candidates_sigma))):
			# Select theta and sigma values
			theta=candidates_theta[exp_counter_1]
			sigma=candidates_sigma[exp_counter_2]
			# Creates matrix to store metrics (A,P,R,F) for every pullback
			metrics = np.zeros((n_cases,4))
			count_pullback = 0
			# For every csv pullback
			for f in files_fold:
				# Data equals values in pullback
				data = all_data[count_pullback]
				# Predictions
				probabilities = data[:,0]
				# Groundtruth
				groundtruth = data[:,1]
				# Gaussian filter application
				smoothed_probabilities = gaussian_filter1d(
					probabilities,
					sigma=sigma,
					mode='constant'
					)
				# Calculate accuracy, precision, recall and fmessure
				A,P,R,F = compute_PRAF(
					groundtruth,
					smoothed_probabilities,
					threshold=theta
					)
				# Store computed metrics for individual pullback
				metrics[count_pullback,0] = A
				metrics[count_pullback,1] = P
				metrics[count_pullback,2] = R 
				metrics[count_pullback,3] = F
				count_pullback +=1
			# Compute mean acc, pre, rec, and f1
			means = np.mean(metrics, axis=0)
			# Compute median acc, pre, rec, and f1
			medians = np.median(metrics, axis=0)
			# Compute quartil 1 acc, pre, rec, and f1
			q1 = np.percentile(metrics, 25, axis=0)
			# Compute quartil 3 acc, pre, rec, and f1
			q3 = np.percentile(metrics, 75, axis=0)
			params.append(theta)
			# Store theta value used
			results[global_exp_counter,0] = theta
			# Store sigma value used
			results[global_exp_counter,1] = sigma
			# Store mean acc, pre, rec and f1
			results[global_exp_counter,2:6] = means
			# Store median acc, pre, rec and f1
			results[global_exp_counter,6:10] = medians
			# Store quartil 1 acc, pre, rec, and f1
			results[global_exp_counter,10:14] = q1
			# Store quartil 3 acc, pre, rec, and f1
			results[global_exp_counter,14:18] = q3
			# Print F1 results for every pair
			#print "Average F1 for sigma=%f, theta=%f = %f"%(sigma,theta,means[-1])
			#print "Median F1 for sigma=%f, theta=%f = %f"%(sigma,theta,medians[-1])
			global_exp_counter+=1
	# Returns (100*100)x18 matrix
	return results


if mode_tunning == 0:
	global_results = np.zeros((n_total_exps,18))
	# Amount of folds. Usualy 10
	nfolds = len(list_subdirs_to_scan)
	# For every folder asignned to a fold with predictions
	# subddir example: '1/CSV/TRAIN'
	for subdir in list_subdirs_to_scan:
		# Path to folder filled with csv containing predictions for every pullback
		# cvsdir example: 'CNN1a/Predictions/CNN1a/1/CSV/TRAIN'
		cvsdir = os.path.join(main_dir,subdir)
		# List csv files for every pullback
		file_list = sorted(os.listdir(cvsdir))
		# Checks whether or not the file exists and is indeed a csv file
		# Creates list of paths to every single csv file
		# Example: 'CNN1a/Predictions/CNN1a/1/CSV/TRAIN/01_PD796ARA.csv'
		files = [os.path.join(cvsdir, f) for f in file_list 
				if (os.path.isfile(os.path.join(cvsdir, f)) and f.endswith(".csv"))]
		# Computes performance meassures for individual fold over parameter grid
		results_fold = tune_for_one_fold(files,fold_description=subdir)
		# Stores performance matrix
		global_results = np.add(global_results,results_fold)

	# Divide all values by number of folds
	global_results /= nfolds
	cvs_result_file = base_result_filename+"MEAN_SUBSET_%s"%(subset_name)
	# cPickle.dump(global_results, open(cvs_result_file+'.pkl', 'wb'))
	pickle.dump(global_results, open(cvs_result_file+'.pkl', 'wb'))
	#np.savetxt(cvs_result_file, global_results, fmt='%.6f', delimiter=',')

else:
	# Amount of folds. Usualy 10
	nfolds = len(list_subdirs_to_scan)
	stacked_results = np.zeros((nfolds,n_total_exps,18))
	counter_folds = 0
	# For every folder asignned to a fold with predictions
	# subddir example: '1/CSV/TRAIN'
	for subdir in list_subdirs_to_scan:
		# Path to folder filled with csv containing predictions for every pullback
		# cvsdir example: 'CNN1a/Predictions/CNN1a/1/CSV/TRAIN'
		cvsdir = os.path.join(main_dir,subdir)
		# List csv files for every pullback
		file_list = sorted(os.listdir(cvsdir))
		# Checks whether or not the file exists and is indeed a csv file
		# Creates list of paths to every single csv file
		# Example: 'CNN1a/Predictions/CNN1a/1/CSV/TRAIN/01_PD796ARA.csv'
		files = [os.path.join(cvsdir, f) for f in file_list 
				if (os.path.isfile(os.path.join(cvsdir, f)) and f.endswith(".csv"))]
		# Computes performance meassures for individual fold over parameter grid
		results_fold = tune_for_one_fold(files,fold_description=subdir)
		# Stores performance matrix
		stacked_results[counter_folds,:,:] = results_fold
		counter_folds += 1
	# Compute median among folds
	global_results = np.median(stacked_results,axis=0)
	cvs_result_file = base_result_filename+"MEDIAN_SUBSET_%s_STACKED0"%(subset_name)
	#np.savetxt(cvs_result_file, stacked_results, fmt='%.6f', delimiter=',')
	pickle.dump(stacked_results, open(cvs_result_file+'.pkl', "wb"))
	# cPickle.dump(stacked_results, open(cvs_result_file+'.pkl', "wb"))
	

