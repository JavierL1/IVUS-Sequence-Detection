#from keras.callbacks import ModelCheckpoint, Callback
import numpy as np

class MultiLabelMetrics:

	def accuracy_one_class(self,ytrue,ypred):
		_ypred = ypred
		_ypred[_ypred>=0.5] = 1.0
		_ypred[_ypred<0.5] = 0.0		
		_acc = 1.0-np.mean(np.abs(ytrue - _ypred))
		return _acc

	def average_accuracy(self, Ytrue,Ypred):
		n,K = Ytrue.shape		
		_accs = []
		for i in range(K):
			_accs.append(self.accuracy_one_class(Ytrue[:,i],Ypred[:,i]))
		return np.mean(np.asarray(_accs))


	def precision_recall_one_class(self,ytrue,ypred,threshold=0.5):

		FP = 0
		FN = 0
		TP = 0
		TN = 0

		for idx in range(len(ytrue)):
			if (ypred[idx]>=threshold) and (ytrue[idx]>=threshold):
				TP = TP + 1
			if (ypred[idx]>=threshold) and (ytrue[idx]<threshold):
				FP = FP + 1
			if (ypred[idx]<threshold) and (ytrue[idx]>=threshold):
				FN = FN + 1
			if (ypred[idx]<threshold) and (ytrue[idx]<threshold):
				TN = TN + 1

		precision = 1.0
		recall = 1.0

		if (TP+FP) > 0:
			precision = float(TP)/float(TP+FP)

		if (TP+FN) > 0:
			recall = float(TP)/float(TP+FN)

		accuracy = float(TP+TN)/float(TP+TN+FP+FN)

		return precision,recall


	def average_precision_recall(self, Ytrue,Ypred):
		n,K = Ytrue.shape		
		precisions = []
		recalls = []
		for i in range(K):
			_prec,_rec =self.precision_recall_one_class(Ytrue[:,i],Ypred[:,i])
			precisions.append()
			recalls.append()

		return np.mean(np.asarray(precisions)),np.mean(np.asarray(recalls))

#THIS COMPUTATION IS ROBUST TO PADDING
def compute_PRAF(ytrue,ypred,threshold=0.5):
	# Fake positives
	FP = 0
	# Fake negatives
	FN = 0
	# True positives
	TP = 0
	# True negatives
	TN = 0

	for idx in range(len(ytrue)):
		if (ypred[idx]>=threshold) and (ytrue[idx]>=threshold):
			TP = TP + 1
		elif (ypred[idx]>=threshold) and (ytrue[idx]<threshold):
			FP = FP + 1
		elif (ypred[idx]<threshold) and (ytrue[idx]>=threshold):
			FN = FN + 1
		elif (ypred[idx]<threshold) and (ytrue[idx]<threshold):
			TN = TN + 1

	precision = 1.0
	recall = 1.0

	if (TP+FP) > 0:
		precision = float(TP)/float(TP+FP)

	if (TP+FN) > 0:
		recall = float(TP)/float(TP+FN)

	accuracy = float(TP+TN)/float(TP+TN+FP+FN)
	
	if precision+recall > 0:
		F = 2*((precision*recall)/(precision+recall))
	else:
		F = 0.0

	return accuracy,precision,recall,F

#THIS COMPUTATION IS ROBUST TO PADDING
def PRAF_from_confusion(FP,FN,TP,TN):

	precision = 1.0
	recall = 1.0

	if (TP+FP) > 0:
		precision = float(TP)/float(TP+FP)

	if (TP+FN) > 0:
		recall = float(TP)/float(TP+FN)

	accuracy = float(TP+TN)/float(TP+TN+FP+FN)
	
	if precision+recall > 0:
		F = 2*((precision*recall)/(precision+recall))
	else:
		F = 0.0

	return accuracy,precision,recall,F
	
def compute_confusion(ytrue,ypred,threshold=0.5):
	
	FP = 0
	FN = 0
	TP = 0
	TN = 0

	for idx in range(len(ytrue)):
		if (ypred[idx]>=threshold) and (ytrue[idx]>=threshold):
			TP = TP + 1
		if (ypred[idx]>=threshold) and (ytrue[idx]<threshold):
			FP = FP + 1
		if (ypred[idx]<threshold) and (ytrue[idx]>=threshold):
			FN = FN + 1
		if (ypred[idx]<threshold) and (ytrue[idx]<threshold):
			TN = TN + 1

	return FP,FN,TP,TN
