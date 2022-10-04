from scipy.ndimage.filters import gaussian_filter1d
import os, sys
import numpy as np
from UBMetrics import compute_PRAF
from UBMetrics import compute_confusion, PRAF_from_confusion
import pickle

# Main folder
# Usualy 'CNN{net}/Predictions/CNN{net}'
main_dir = sys.argv[1]
# 'CNN1a/Results/'
base_result_filename = sys.argv[2]#e.g. "KFOLD_DTUNNING_NET3L"
# 'CNN1a/Tuning/MEDIAN_SUBSET_{EXTRAIN/TRAIN}_STACKED0.pkl'
filename_to_extract = sys.argv[3]
# 1
subset_apply = int(sys.argv[4]) #0:train, 1:extra, 2:test, other:all
# 1
mode_tunning = int(sys.argv[5]) #0:mean among folds, other:median among folds

if not os.path.exists(base_result_filename):
	os.mkdir(base_result_filename)

origin_set = 'TRAIN_CNN'

mode_tunning_name = 'MEAN' if mode_tunning == 0 else 'MEDIAN'

where_to_extract_best = pickle.load(open(filename_to_extract, "rb"))

names_folds = [ str(i) for i in range(1,11) ]

name_set_to_apply = 'UNDEFINED'

if subset_apply == 0:
	list_subdirs_to_scan = [name_fold+'/CSV/TRAIN_CNN' for name_fold in names_folds]
	name_set_to_apply = 'TRAIN_CNN_(EACH_FOLD)'
elif subset_apply == 1:
	list_subdirs_to_scan = [name_fold+'/CSV/TRAIN_RNN' for name_fold in names_folds]
	name_set_to_apply = 'TRAIN_RNN_(EACH_FOLD)'
elif subset_apply == 2:
	list_subdirs_to_scan = [name_fold+'/CSV/VAL' for name_fold in names_folds]
	name_set_to_apply = 'VAL_(EACH_FOLD)'
elif subset_apply == 3:
	list_subdirs_to_scan = [name_fold+'/CSV/TEST' for name_fold in names_folds]
	name_set_to_apply = 'TEST_(EACH_FOLD)'

def evaluate_frame_based_one_fold(files_fold,sigma=2,theta=0.5,fold_description='None'):

	print("COMPUTING FRAME BASED FOR FOLD %s"%fold_description)
	print("Using values theta=%f,sigma=%f"%(theta,sigma))

	results = np.zeros((1,6))
	n_cases = len(files_fold)
	all_data = []

	for f in files_fold: #CVS corresponding to different pullback
		data = np.loadtxt(f,delimiter=',')
		all_data.append(data)

	count_pullback = 0

	FP_all = 0.0
	FN_all = 0.0
	TP_all = 0.0
	TN_all = 0.0

	for f in files_fold: #CVS corresponding to different pullback
		data = all_data[count_pullback] #np.loadtxt(f,delimiter=',')
		probabilities = data[:,0]
		groundtruth = data[:,1]
		smoothed_probabilities = gaussian_filter1d(probabilities,sigma=sigma,mode='constant')
		
		FP_one,FN_one,TP_one,TN_one = compute_confusion(groundtruth,smoothed_probabilities,threshold=theta)
		
		FP_all += FP_one
		FN_all += FN_one
		TP_all += TP_one
		TN_all += TN_one

		count_pullback +=1

	accuracy,precision,recall,F = PRAF_from_confusion(FP_all,FN_all,TP_all,TN_all)

	results[0,0] = theta
	results[0,1] = sigma
	results[0,2] = accuracy
	results[0,3] = precision
	results[0,4] = recall
	results[0,5] = F

	return results

def evaluate_for_one_fold(files_fold,sigma=2,theta=0.5,fold_description='None'):
	
	print("Appyling to FOLD %s"%fold_description)
	print("Using values theta=%f,sigma=%f"%(theta,sigma))

	results = np.zeros((1,18))
	n_cases = len(files_fold)
	all_data = []

	for f in files_fold: #CVS corresponding to different pullback
		data = np.loadtxt(f,delimiter=',')
		all_data.append(data)

	global_exp_counter = 0

	metrics = np.zeros((n_cases,4))
	count_pullback = 0
	for f in files_fold: #CVS corresponding to different pullback
		data = all_data[count_pullback] #np.loadtxt(f,delimiter=',')
		probabilities = data[:,0]
		groundtruth = data[:,1]
		smoothed_probabilities = gaussian_filter1d(probabilities,sigma=sigma,mode='constant')
		A,P,R,F = compute_PRAF(groundtruth,smoothed_probabilities,threshold=theta)
		metrics[count_pullback,0] = A
		metrics[count_pullback,1] = P
		metrics[count_pullback,2] = R 
		metrics[count_pullback,3] = F
		count_pullback +=1

	means = np.mean(metrics, axis=0)
	medians = np.median(metrics, axis=0)
	q1 = np.percentile(metrics, 25, axis=0)
	q3 = np.percentile(metrics, 75, axis=0)

	print("Average F1 for sigma=%f, theta=%f = %f"%(sigma,theta,means[1]))
	print("Average P for sigma=%f, theta=%f = %f"%(sigma,theta,means[2]))
	print("Average R for sigma=%f, theta=%f = %f"%(sigma,theta,means[3]))

	print("Median F1 for sigma=%f, theta=%f = %f"%(sigma,theta,medians[-1]))
	print("Median F1 for sigma=%f, theta=%f = %f"%(sigma,theta,medians[3]))
	print("Median P for sigma=%f, theta=%f = %f"%(sigma,theta,medians[1]))
	print("Median R for sigma=%f, theta=%f = %f"%(sigma,theta,medians[2]))


	results[0,0] = theta
	results[0,1] = sigma
	results[0,2:6] = means #mean A,P,R,F
	results[0,6:10] = medians #median A,P,R,F
	results[0,10:14] = q1
	results[0,14:18] = q3

	return results


nfolds = len(list_subdirs_to_scan)
results_all_folds = np.zeros((nfolds,18))
counter_folds = 0

frame_based_all_folds = np.zeros((nfolds,6))

for subdir in list_subdirs_to_scan:#for each fold
	
	results_this_fold = where_to_extract_best[counter_folds,:,:]
	
	F = results_this_fold[:,9]
	indices = np.argsort(-F)
	idx_best = indices[0]
	best_F = results_this_fold[idx_best,9]
	best_sigma = results_this_fold[idx_best,1]
	best_theta = results_this_fold[idx_best,0]
	# Path to folder filled with csv containing predictions for every pullback
	# cvsdir example: 'CNN1a/Predictions/CNN1a/1/CSV/TRAIN'
	cvsdir = os.path.join(main_dir,subdir)
	file_list = sorted(os.listdir(cvsdir))
	files = [os.path.join(cvsdir, f) for f in file_list 
			if (os.path.isfile(os.path.join(cvsdir, f)) and f.endswith(".csv"))]
	results_fold = evaluate_for_one_fold(
		files,
		sigma=best_sigma,
		theta=best_theta,
		fold_description=subdir
		)
	results_all_folds[counter_folds,:] = results_fold

	frame_based_one_fold = evaluate_frame_based_one_fold(
		files,
		sigma=best_sigma,
		theta=best_theta,
		fold_description=subdir
		)
	frame_based_all_folds[counter_folds,:] = frame_based_one_fold

	counter_folds += 1

median_results = np.median(results_all_folds,axis=0)
av_results = np.mean(results_all_folds,axis=0)

av_frame_based = np.mean(frame_based_all_folds,axis=0)

std_pullback_based = np.std(results_all_folds,axis=0)
std_frame_based = np.std(frame_based_all_folds,axis=0)

print(results_all_folds.shape)
print(frame_based_all_folds.shape)
print(median_results)

print("PULLBACK BASED METRICS ...")
print("MEDIAN F1 ON SET %s = %.2f"%(name_set_to_apply,median_results[9]))
print("MEDIAN PREC ON SET %s = %.2f"%(name_set_to_apply,median_results[7]))
print("MEDIAN REC ON SET %s = %.2f"%(name_set_to_apply,median_results[8]))
print("MEDIAN ACC ON SET %s = %.2f"%(name_set_to_apply,median_results[6]))

print("MEAN F1 ON SET %s = %.2f"%(name_set_to_apply,av_results[9]))
print("MEAN PREC ON SET %s = %.2f"%(name_set_to_apply,av_results[7]))
print("MEAN REC ON SET %s = %.2f"%(name_set_to_apply,av_results[8]))
print("MEAN ACC ON SET %s = %.2f"%(name_set_to_apply,av_results[6]))

print("FRAME BASED METRICS ...")
print("MEAN F1 ON SET %s = %.2f"%(name_set_to_apply,av_frame_based[5]))
print("MEAN PREC ON SET %s = %.2f"%(name_set_to_apply,av_frame_based[3]))
print("MEAN REC ON SET %s = %.2f"%(name_set_to_apply,av_frame_based[4]))
print("MEAN AC ON SET %s = %.2f"%(name_set_to_apply,av_frame_based[2]))

print("PULLBACK-BASED STD F1, P, R = %.2f, %.2f, %.2f"%(std_pullback_based[9],std_pullback_based[7],std_pullback_based[8]))
print("FRAME-BASED STD F1, P, R = %.2f, %.2f, %.2f"%(std_frame_based[5],std_frame_based[3],std_frame_based[4]))

cvs_result_file = base_result_filename+"ORIGIN_%s_MODE_%s_SUBSET_APPLY_%s_PULLBACK_BASED"%(origin_set, mode_tunning_name,name_set_to_apply)
with open(cvs_result_file, 'w') as result_file:	
	np.savetxt(result_file, results_all_folds, fmt='%.6f', delimiter=',')

cvs_result_file2 = base_result_filename+"ORIGIN_%s_MODE_%s_SUBSET_APPLY_%s_FRAME_BASED"%(origin_set, mode_tunning_name,name_set_to_apply)
with open(cvs_result_file2, 'w') as result_file2:
	np.savetxt(result_file2, frame_based_all_folds, fmt='%.6f', delimiter=',')


