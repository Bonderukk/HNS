import numpy as np 

def MAV(data):
    length = len(data)
    mav = sum([abs(i) for i in data])
    return mav

def NZC(data,th = 1e-5):
    length = len(data)
    zero_crossing = 0
    for i in range(length-1):
        if (data[i+1] * data[i] < 0) and abs(data[i+1] - data[i]) >= th:
            zero_crossing += 1
    
    return zero_crossing

def NSSC(data,th = 1e-4):
    length = len(data)
    slope_sign_changes = 0
    for i in range(1,length-1):
        temp = (data[i] - data[i-1])*(data[i] - data[i+1])
        if temp >= th:
            slope_sign_changes += 1
    
    return slope_sign_changes

def RMS(data):
    length = len(data)
    temp = sum([x**2 for x in data])
    temp /= length

    rms = np.sqrt(temp)
    return rms

def WL(data):
    length = len(data)
    wl = sum([abs(data[i+1] - data[i]) for i in range(length-1)])
    return wl

def WAMP(data,th = 1e-2):
    length = len(data)
    wamp = 0 
    for i in range(length-1):
        if data[i] - data[i+1] >= th:
            wamp += 1
    
    return wamp

def PTP(data):
    length = len(data)
    ptp = max(data) - min(data)
    return ptp

def get_tdf(data):
    function_names = {
        MAV.__name__,NZC.__name__,NSSC.__name__,
        RMS.__name__,WL.__name__,WAMP.__name__,
        max.__name__,min.__name__,PTP.__name__
    }

    tdf = [
        MAV(data),NZC(data),NSSC(data),
        RMS(data),WL(data),WAMP(data),
        max(data), min(data), PTP(data)
    ]
    
    return tdf