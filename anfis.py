import itertools
import numpy as np
import copy



def partial_dMF(x, mf_def, partial_param):
    mf_nm = mf_def[0]

    if mf_nm == 'gaussmf':

        sigma = mf_def[1]['sigma']
        mean = mf_def[1]['mean']

        if partial_param == 'sigma':
            res = (2. / sigma ** 3) * np.exp(-(((x - mean) ** 2) / (sigma) ** 2)) * (x - mean) ** 2
        elif partial_param == 'mean':
            res = (2. / sigma ** 2) * np.exp(-(((x - mean) ** 2) / (sigma) ** 2)) * (x - mean)

    return res




class ANFIS:

    def __init__(self, X, Y, memberfunction):
        self.X = np.array(copy.copy(X))
        self.Y = np.array(copy.copy(Y))
        self.XLen = len(self.X)
        self.memClass = copy.deepcopy(memberfunction)
        self.memberfuncs = self.memClass.MFList
        self.memberfunctionsByVariable = [[x for x in range(len(self.memberfuncs[z]))] for z in range(len(self.memberfuncs))]
        self.rules = np.array(list(itertools.product(*self.memberfunctionsByVariable)))
        self.consequents = np.empty(self.Y.ndim * len(self.rules) * (self.X.shape[1] + 1))
        self.consequents.fill(0)
        self.errors = np.empty(0)
        self.memberfunctionsHomo = all(len(i)==len(self.memberfunctionsByVariable[0]) for i in self.memberfunctionsByVariable)
        self.trainingType = 'Not trained yet'

    def LSE(self, A, B, initialGamma = 1000.):
        coeffMat = A
        rhsMat = B
        S = np.eye(coeffMat.shape[1])*initialGamma
        x = np.zeros((coeffMat.shape[1],1)) # need to correct for multi-dim B
        for i in range(len(coeffMat[:,0])):
            a = coeffMat[i,:]
            b = np.array(rhsMat[i])
            S = S - (np.array(np.dot(np.dot(np.dot(S,np.matrix(a).transpose()),np.matrix(a)),S)))/(1+(np.dot(np.dot(S,a),a)))
            x = x + (np.dot(S,np.dot(np.matrix(a).transpose(),(np.matrix(b)-np.dot(np.matrix(a),x)))))
        return x

    def trainHybridJangOffLine(self, epochs=5, tolerance=1e-5, initialGamma=1000, k=0.01):

        self.trainingType = 'trainHybridJangOffLine'
        convergence = False
        epoch = 1
        while (epoch < epochs) and (convergence is not True):


            [lfour, wSum, w] = forwardHalfPass(self, self.X)


            lfive = np.array(self.LSE(lfour,self.Y,initialGamma))
            self.consequents = lfive
            lfive = np.dot(lfour,lfive)

            #error
            error = np.sum((self.Y-lfive.T)**2)
            print('current error: '+ str(error))
            average_error = np.average(np.absolute(self.Y-lfive.T))
            self.errors = np.append(self.errors,error)

            if len(self.errors) != 0:
                if self.errors[len(self.errors)-1] < tolerance:
                    convergence = True


            if convergence is not True:
                cols = range(len(self.X[0,:]))
                dE_dAlpha = list(backprop(self, colX, cols, wSum, w, lfive) for colX in range(self.X.shape[1]))


            if len(self.errors) >= 4:
                if (self.errors[-4] > self.errors[-3] > self.errors[-2] > self.errors[-1]):
                    k = k * 1.1

            if len(self.errors) >= 5:
                if (self.errors[-1] < self.errors[-2]) and (self.errors[-3] < self.errors[-2]) and (self.errors[-3] < self.errors[-4]) and (self.errors[-5] > self.errors[-4]):
                    k = k * 0.9


            t = []
            for x in range(len(dE_dAlpha)):
                for y in range(len(dE_dAlpha[x])):
                    for z in range(len(dE_dAlpha[x][y])):
                        t.append(dE_dAlpha[x][y][z])

            eta = k / np.abs(np.sum(t))

            if(np.isinf(eta)):
                eta = k


            dAlpha = copy.deepcopy(dE_dAlpha)
            if not(self.memberfunctionsHomo):
                for x in range(len(dE_dAlpha)):
                    for y in range(len(dE_dAlpha[x])):
                        for z in range(len(dE_dAlpha[x][y])):
                            dAlpha[x][y][z] = -eta * dE_dAlpha[x][y][z]
            else:
                dAlpha = -eta * np.array(dE_dAlpha)


            for varsWithMemFuncs in range(len(self.memberfuncs)):
                for MFs in range(len(self.memberfunctionsByVariable[varsWithMemFuncs])):
                    paramLst = sorted(self.memberfuncs[varsWithMemFuncs][MFs][1])
                    for param in range(len(paramLst)):
                        self.memberfuncs[varsWithMemFuncs][MFs][1][paramLst[param]] = self.memberfuncs[varsWithMemFuncs][MFs][1][paramLst[param]] + dAlpha[varsWithMemFuncs][MFs][param]
            epoch = epoch + 1


        self.fittedValues = predict(self,self.X)
        self.residuals = self.Y - self.fittedValues[:,0]

        return self.fittedValues

    def showRes(self):
        if self.trainingType == 'Not trained yet':
            print(self.trainingType)
        else:
            import matplotlib.pyplot as plt
            plt.plot(range(len(self.fittedValues)),self.fittedValues,'r', label='trained')
            plt.plot(range(len(self.Y)),self.Y,'b', label='original')
            plt.legend(loc='upper left')
            plt.show()

    def showErr(self):
        if self.trainingType == 'Not trained yet':
            print(self.trainingType)
        else:
            import matplotlib.pyplot as plt
            plt.plot(range(len(self.errors)),self.errors,'ro', label='errors')
            plt.ylabel('error')
            plt.xlabel('epoch')
            plt.show()
    def predict_models(self, vars):
        return predict(self,vars)


def predict(ANFISObj, varsToTest):
    [lfour, wSum, w] = forwardHalfPass(ANFISObj, varsToTest)

    # layer five
    lfive = np.dot(lfour, ANFISObj.consequents)

    return lfive


def forwardHalfPass(ANFISObj, Xs):
    lfour = np.empty(0, )
    wSum = []

    for pattern in range(len(Xs[:, 0])):
        # layer one
        lone = ANFISObj.memClass.evaluateMF(Xs[pattern, :])

        # layer two
        miAlloc = [[lone[x][ANFISObj.rules[row][x]] for x in range(len(ANFISObj.rules[0]))] for row in
                   range(len(ANFISObj.rules))]
        ltwo = np.array([np.product(x) for x in miAlloc]).T
        if pattern == 0:
            w = ltwo
        else:
            w = np.vstack((w, ltwo))

        # layer three
        wSum.append(np.sum(ltwo))
        if pattern == 0:
            wNormalized = ltwo / wSum[pattern]
        else:
            wNormalized = np.vstack((wNormalized, ltwo / wSum[pattern]))

        # prep for layer four (bit of a hack)
        lthree = ltwo / wSum[pattern]
        rowHolder = np.concatenate([x * np.append(Xs[pattern, :], 1) for x in lthree])
        lfour = np.append(lfour, rowHolder)

    w = w.T
    wNormalized = wNormalized.T

    lfour = np.array(np.array_split(lfour, pattern + 1))

    return lfour, wSum, w


def backprop(ANFISObj, columnX, columns, theWSum, theW, theLayerFive):
    paramGrp = [0] * len(ANFISObj.memberfuncs[columnX])
    for MF in range(len(ANFISObj.memberfuncs[columnX])):

        parameters = np.empty(len(ANFISObj.memberfuncs[columnX][MF][1]))
        timesThru = 0
        for alpha in sorted(ANFISObj.memberfuncs[columnX][MF][1].keys()):

            bucket3 = np.empty(len(ANFISObj.X))
            for rowX in range(len(ANFISObj.X)):
                varToTest = ANFISObj.X[rowX, columnX]
                tmpRow = np.empty(len(ANFISObj.memberfuncs))
                tmpRow.fill(varToTest)

                bucket2 = np.empty(ANFISObj.Y.ndim)
                for colY in range(ANFISObj.Y.ndim):

                    rulesWithAlpha = np.array(np.where(ANFISObj.rules[:, columnX] == MF))[0]
                    adjCols = np.delete(columns, columnX)

                    senSit = partial_dMF(ANFISObj.X[rowX, columnX], ANFISObj.memberfuncs[columnX][MF], alpha)
                    # produces d_ruleOutput/d_parameterWithinMF
                    dW_dAplha = senSit * np.array(
                        [np.prod([ANFISObj.memClass.evaluateMF(tmpRow)[c][ANFISObj.rules[r][c]] for c in adjCols]) for r
                         in rulesWithAlpha])

                    bucket1 = np.empty(len(ANFISObj.rules[:, 0]))
                    for consequent in range(len(ANFISObj.rules[:, 0])):
                        fConsequent = np.dot(np.append(ANFISObj.X[rowX, :], 1.), ANFISObj.consequents[((ANFISObj.X.shape[1] + 1) * consequent):(((ANFISObj.X.shape[1] + 1) * consequent) + (ANFISObj.X.shape[1] + 1)),colY])
                        acum = 0
                        if (consequent in rulesWithAlpha):
                            acum = dW_dAplha[np.where(rulesWithAlpha == consequent)] * theWSum[rowX]

                        acum = acum - theW[consequent, rowX] * np.sum(dW_dAplha)
                        acum = acum / theWSum[rowX] ** 2
                        bucket1[consequent] = fConsequent * acum

                    sum1 = np.sum(bucket1)

                    if (ANFISObj.Y.ndim == 1):
                        bucket2[colY] = sum1 * (ANFISObj.Y[rowX] - theLayerFive[rowX, colY]) * (-2)
                    else:
                        bucket2[colY] = sum1 * (ANFISObj.Y[rowX, colY] - theLayerFive[rowX, colY]) * (-2)

                sum2 = np.sum(bucket2)
                bucket3[rowX] = sum2

            sum3 = np.sum(bucket3)
            parameters[timesThru] = sum3
            timesThru = timesThru + 1

        paramGrp[MF] = parameters

    return paramGrp
