# predictor for AlixPartners Analytics Challenge, based on
# many different predictors

import csv
import numpy
from sklearn.metrics import roc_curve, auc, classification_report
# from sklearn.metrics import auc_score # version 0.12 or greater
from sklearn.linear_model import Lars, LassoLars, Lasso
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.linear_model import ElasticNet as ELN
from sklearn.linear_model import ARDRegression as ARDR
from sklearn.grid_search import GridSearchCV as GSCV
from sklearn import cross_validation
from sklearn.svm import OneClassSVM

# define Area under ROC function, to be used in Cross-validation
def my_auc_score(y_true,y_score):
    fpr, tpr, threshold = roc_curve(y_true,y_score)
    return auc(fpr,tpr)

# Load data file
# Will be split into different data sets later
raw_input = csv.reader( open('Problem2InputData.csv','rb') )
header = raw_input.next() # discard first line

########################
# Input data
trainingDataRows = [] # 250 rows
evalDataRows = [] # 19750 rows
# Practice/training vectors
trainPracticeRows = [] # 250 practice training numbers
targPracticeRows = [] # 19750 practice truths
trainEvalRows = [] # 250 eval training numbers

# [0] id;  [1] Training
# [2] Target Practice;  [3] Target Eval
# [-300:] Predictors
for row in raw_input : 
    if int(row[1]) is 1 : # training data
        trainingDataRows.append( row[-300:] )
        trainPracticeRows.append( row[2] )
        trainEvalRows.append( row[3] )
    else:
        evalDataRows.append( row[-300:] )
        targPracticeRows.append( row[2] )

del raw_input 

# convert to arrays, and specify data type
trainingData = numpy.array( trainingDataRows, dtype=float )
evalData = numpy.array( evalDataRows, dtype=float )
trainPractice = numpy.array( trainPracticeRows, dtype=float )
trainEval = numpy.array( trainEvalRows, dtype=float )
targPractice = numpy.array( targPracticeRows, dtype=float )

############################################
# Data Settings & Cross-validation method

isPractice = False  # true: print AUC of practice; false: predicted to file  
if isPractice:
    trainTarg = trainPractice
else:
    trainTarg = trainEval

Nfolds = 2 # number of folds in cross-validation.
# cv = cross_validation.KFold(len(trainEval),Nfolds)
cv = cross_validation.StratifiedKFold(trainTarg,Nfolds)
# cv = cross_validation.LeavePOut(len(trainEval), 4) 
# LPO not working; often end up with monotypic truths. Also, really slow. 

############################
# Lars
paramsLars = {'n_nonzero_coefs': range(1,len(trainEval)/Nfolds)}
modelLars = Lars(fit_intercept=True,normalize=True)

# Lasso
paramsLasso = {'alpha': [x/10000. for x in range(1,10000,200)] }
modelLasso = LassoLars(fit_intercept=True,normalize=True)

# Orthogonal Matching Pursuit
paramsOMP = {'n_nonzero_coefs': range(1,len(trainEval)/Nfolds)}
modelOMP = OMP(fit_intercept=True,normalize=True)

# Ridge
paramsRidge = {'alpha': [x/100. for x in range(1,100,20)] }
modelRidge = Ridge(fit_intercept=True,normalize=True)

# Elastic Net
paramsELN = {'alpha': [x/100. for x in range(1,101,20)],
             'rho': [x/100. for x in range(1,101,20)]}
modelELN = ELN(fit_intercept=True) # may not yet be implemented in version 0.10

# Bayesian Ridge
paramsBayR = {'alpha_1': [1./10.**(x) for x in range(5,8)],
              'alpha_2': [1./10.**(x) for x in range(5,8)],
              'lambda_1': [1./10.**(x) for x in range(5,8)],
              'lambda_2': [1./10.**(x) for x in range(5,8)]}
modelBayR = BayesianRidge(fit_intercept=True,normalize=True)

# Automatic Relevance Determination
paramsARDR = {'alpha_1': [1./10.**(x) for x in range(5,8)],
              'alpha_2': [1./10.**(x) for x in range(5,8)],
              'lambda_1': [1./10.**(x) for x in range(5,8)],
              'lambda_2': [1./10.**(x) for x in range(5,8)]}
modelARDR = ARDR(fit_intercept=True,normalize=True)

#############################
# make list of all models to be tested
models= []
models.append(["LARS",modelLars,paramsLars])
models.append(["Lasso",modelLasso,paramsLasso])
models.append(["OMP",modelOMP,paramsOMP])
models.append(["Ridge",modelRidge,paramsRidge])
models.append(["ElasticNet",modelELN,paramsELN])
models.append(["BayesianRidge",modelBayR,paramsBayR])
# models.append(["AutoRelevance",modelARDR,paramsARDR]) # comp. intensive
# models.append(["",model,params])

best_model = 0
best_score = 0.

# Loop over all models, and perform model selection. 
# Print results of each along the way.
for name,model,params in models:
    print name
    gscv = GSCV(model,params,
                score_func=my_auc_score, n_jobs=-1, refit=True)
    gscv.fit(trainingData, trainTarg, cv=cv)
    print 
    print gscv.best_estimator_
    print 'Best Training AUC: ', gscv.best_score_
    if isPractice:
        targPred = gscv.best_estimator_.predict( evalData )
        print 'Practice AUC: ', my_auc_score( targPractice, targPred )
        print
    if gscv.best_score_ > best_score : 
        best_score = gscv.best_score_
        best_model = gscv.best_estimator_

# After end of loop, use the best model to make the prediction
print 
print
print "Best model: "
print best_model
print best_score
print 

evalPred = best_model.predict( evalData )
# print out practice run, or save to file
if isPractice:
    print 'Practice AUC: ', my_auc_score( targPractice, evalPred )
else:
    outfile = open('predictions.dat','w')
    for i in range(1,len(targPractice)): 
        outfile.write(str(i+len(trainTarg))) # Offset ID
        outfile.write(',  ')
        outfile.write(str(evalPred[i]))
        outfile.write('\n')

    outfile.close()


print('done!');

