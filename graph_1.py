import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.optimize

startDate = datetime.datetime(year=2020,month=3,day=1)
cureTime=14
cureRate = 0.05

infectionData = np.loadtxt('NY_31-03.csv')

def logisticDiffEq(t_end, daily_steps, infectivity, pop_size, cure_time):
    results = np.zeros(t_end)
    infected = np.zeros(t_end)
    results[0] = 1

    for i in range(t_end - 1):
        daily = np.zeros(daily_steps) + results[i]
        for j in range(daily_steps-1):
            newInf = daily[j] * infectivity * (1-daily[j]/pop_size)
            daily[j+1] = daily[j] + newInf
            infected[i] += newInf

        cured = 0
        if i > cure_time:
            cured = infected[i - cure_time]
        results[i+1]=daily[-1] - cured
        pop_size -= cured
        # pop_size = np.min([0,pop_size])

    return results


def coronavirus(t,r,pop_size):
    return logisticDiffEq(t,4,r,pop_size,cureTime)


### # OPTIMIZE:

def meanError(vars,debug=False):
    r = vars[0]
    pop_size = vars[1]
    estimatedCurve = coronavirus(len(infectionData),r,pop_size)
    difference = np.array([(estimatedCurve[i] - infectionData[i]) * np.exp(-1*np.abs(i - 1) *0.1) for i in range(15, len(infectionData))])
    deltaDifference = np.array([((estimatedCurve[i+1] - estimatedCurve[i]) - (infectionData[i+1] - infectionData[i])) * np.exp(-1 *(30-i)**2) for i in range(15, len(infectionData) - 1)])

    # difference = np.array([(estimatedCurve[i] - infectionData[i]) for i in range(len(infectionData))])
    # deltaDifference = np.array([((estimatedCurve[i+1] - estimatedCurve[i]) - (infectionData[i+1] - infectionData[i])) for i in range(len(infectionData) - 1)])

    if debug:
        print(difference)
        print(deltaDifference)
    return np.sqrt(np.sum(difference**2) + 10 * np.sum(deltaDifference**2))

opt = scipy.optimize.minimize(meanError,[0.02,100000], bounds=((0,None),(100000,None)))
print(opt.success)
print(opt.x)
print(meanError(opt.x, debug=True))

prediction = coronavirus(60,opt.x[0],opt.x[1])

#Statistical Analysis
peakDay = np.argmax(prediction).item()
print("maximum at: {}/{}".format((startDate + datetime.timedelta(days=peakDay)).day, (startDate + datetime.timedelta(days=peakDay)).month))

covariance = np.cov([infectionData, prediction[:len(infectionData)]])
print("Covariance is: {}".format(covariance))

#error computation:
deltas = np.sqrt(covariance[0][1])
grad = scipy.optimize.approx_fprime(np.array([opt.x[0]]), lambda x: np.argmax(coronavirus(60,x[0],opt.x[1])), 0.01)
print("gradient is: {}".format(grad))
stdDev = grad * deltas / opt.x[1]
print("Standard Deviation is: {}".format(stdDev))


plt.plot(range(60), prediction)
plt.plot([i for i in range(len(infectionData))], infectionData, 'ks')
plt.plot(np.array([1,1]) * peakDay, [0,100000], 'k--')
plt.xticks(range(0,60,5), ["{}/{}".format((startDate + datetime.timedelta(days=i)).day, (startDate + datetime.timedelta(days=i)).month) for i in range(0,60,5)])
plt.show()
