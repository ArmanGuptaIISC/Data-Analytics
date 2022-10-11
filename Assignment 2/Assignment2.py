import  pandas as pd
import numpy as np
import datetime
from tqdm import tqdm


# Read the data and return the times from the data
def get_times(data):
    times = data[:,0:5]
    return times

# Change the longitude which is given in Zodiac Band, Degree, Minutes and Seconds to Radian.
def changeZDMSToR(opposition):
    degree  = 0
    degree += opposition[0] * 30 
    degree += opposition[1]
    degree += opposition[2] / 60
    degree += opposition[3] / 3600

    return np.radians(degree)

# Read the data and return the opposition data after converting data points to Radian using changeZDMSToR() function.
def get_oppositions(data):
    longitudes = data[:,5:9]
    # changing angles in oppositions to degrees only
    oppositions = [changeZDMSToR(opp) for opp in longitudes]
    return oppositions

    

# Return the predicted angular distance for each oppositional postions by computing time difference of each point from the starting 
# point and multiplying those time difference by angular speed s and finally adding angle z which equant 0 is making wrt Equant-Aries
# reference line
def getDeltaTime(times ,z, s):
    deltaTimes = [z]
    for i in range(1,len(times)):
        time1 = times[i-1]
        time2 = times[i]
        a = datetime.datetime(time1[0], time1[1], time1[2], time1[3], time1[4])
        b = datetime.datetime(time2[0], time2[1], time2[2], time2[3], time2[4])

        c = b-a
        minutes = c.total_seconds() / 60
        hours = minutes / 60
        days = hours / 24
        deltaTimes.append((deltaTimes[i-1] + days * s) % 360)


    return [np.radians(d) for d in deltaTimes]  

# return θ(e) in randians based on the quadrant (xe,ye) point lies in.
def getThetaE(xe,ye):
    theta_e = np.arctan(np.abs(ye / xe))
    if ye >=0 and xe>=0:
        return theta_e
    elif ye >=0 and xe < 0:
        return np.pi - theta_e
    elif ye < 0 and xe < 0:
        return np.pi + theta_e
    return 2 * np.pi - theta_e

# return the opposition discrepancy in minutes 
def oppositionDiscrepancy(c,r,e1,e2,z_,theta):
    cx,cy = np.cos(c) , np.sin(c)
    ex,ey = e1 * np.cos(e2) , e1 * np.sin(e2)

    m = np.tan(z_)
    beta = ey - m * ex - cy
    b = -2 * (cx- m * beta) / (1 + m**2)
    a = 1
    c = (beta **2 + cx ** 2 - r**2 )/ (1 + m**2)
    
    tmp1 = np.sqrt(b ** 2 - 4 *a * c)
    xe1 = (-b - tmp1) / (2 * a)
    ye1 = m * (xe1 - ex) + ey

    xe2 = (-b + tmp1)/ 2 * a 
    ye2 = m * (xe2 - ex) + ey
    theta_e1 = getThetaE(xe1,ye1) 
    theta_e2 = getThetaE(xe2,ye2)

    if np.abs(theta - theta_e1) > np.abs(theta - theta_e2):
        return np.degrees(theta_e2-theta) * 60
    return np.degrees(theta_e1-theta) * 60

# Question 1 
def MarsEquantModel(c,r,e1,e2,z,s,times,oppositions):
    # changing angle from degrees tp radians
    c = np.radians(c)
    e2 = np.radians(e2)
    deltatime = getDeltaTime(times,z,s)
    error = []
    for i in range(len(oppositions)):
        error.append(oppositionDiscrepancy(c,r,e1,e2, deltatime[i], oppositions[i]))
    return error , np.max([np.abs(e) for e in error])

# Question 2 
def bestOrbitInnerParams(r,s,times,oppositions):
    opt_maxE = np.Inf
    opt_error = None
    startZ = np.round(np.degrees(oppositions[0]) ,2)
    opt_c , opt_e1 ,opt_e2 , opt_z = None, None, None, None
    for c in tqdm(np.arange(140,151,1)):
        for e2 in np.arange(145,150,0.5):
            for e1 in np.arange(1,2,0.2):
                for z in np.arange(startZ-10,startZ +10, 1):
                    errors,maxError = MarsEquantModel(c, r ,e1 , e2, z, s ,times =times\
                                        ,oppositions = oppositions)
                    
                    if(maxError < opt_maxE):
                        opt_error = errors
                        opt_maxE = maxError
                        opt_c , opt_e1 ,opt_e2 , opt_z = c ,e1, e2, z
    return opt_c , opt_e1 ,opt_e2 , opt_z, opt_error , opt_maxE

# Question 3 
def  bestS(r,times,oppositions):
    opt_maxE = np.Inf
    opt_maxE = np.Inf
    opt_error = None
    explore_s = [ 0.524 - 0.05 + i * 0.01 for i in range(10)]
    # print(explore_s)
    best_s = None
    for s in tqdm(explore_s):
        c,e1,e2,z,errors,maxError = bestOrbitInnerParams(r,s,times,oppositions)
        if(maxError < opt_maxE):
            opt_error = errors
            opt_maxE = maxError
            best_s = s
            
    return best_s ,opt_error, opt_maxE

# Question 4 
## Calculating the average distance of the black dots (as describe din silde 31) 
# from the center of the circle and estimating it to next r.
def estimateR(c,e1,e2,z,s,times,oppositions):
    
    cx,cy = np.cos(c) , np.sin(c)
    ex,ey = e1 * np.cos(e2) , e1 * np.sin(e2)
    # print(times,z,s)
    deltatime = getDeltaTime(times,z,s)
    total_r = 0
    for i in range(len(oppositions)):
        m = np.tan(deltatime[i])
        x = (ey - m * ex ) / (np.tan(oppositions[i]) - m)
        y = x * np.tan(oppositions[i])
        total_r += np.sqrt((cx - x)**2 + (cy - y)**2)
    return total_r/len(oppositions)

def bestR(s,times,oppositions):
    opt_maxE = np.Inf
    opt_error = None
    best_r = None
    #initial guess of r
    r = 8.5
    for i in (range(10)):
        for _r in np.arange(r-0.3,r+0.3,0.1):
            c,e1,e2,z,errors,maxError = bestOrbitInnerParams(_r,s,times,oppositions)
            if(maxError < opt_maxE):
                opt_error = errors
                opt_maxE = maxError
                best_r = _r
        r = estimateR(c,e1,e2,z,s,times,oppositions)
            
    return best_r ,opt_error, opt_maxE

# Question 5
def bestMarsOrbitParams(times,oppositions):
    opt_maxE = np.inf
    explore_s = [ 0.524 - 0.05 + i * 0.01 for i in range(10)]
    for s in explore_s:
        # initial guess for r
        r = 8.5
        for i in tqdm(range(10)):
            for _r in np.arange(r-0.3,r+0.3,0.1):
                c,e1,e2,z,errors,maxError = bestOrbitInnerParams(_r,s,times,oppositions)
                if(maxError < opt_maxE):
                    opt_error = errors
                    opt_maxE = maxError
                    best_r = _r
                    best_s = s
                    opt_c , opt_e1 ,opt_e2 , opt_z = c ,e1, e2, z
            r = estimateR(c,e1,e2,z,s,times,oppositions)
    return best_r ,best_s, opt_c, opt_e1 ,opt_e2 , opt_z , opt_error , opt_maxE


# Main function as per template
if __name__ == "__main__":

    # Import oppositions data from the CSV file provided
    data = np.genfromtxt(
        "../data/01_data_mars_opposition_updated.csv",
        delimiter=",",
        skip_header=True,
        dtype="int",
    )

    # Extract times from the data in terms of number of days.
    # "times" is a numpy array of length 12. The first time is the reference
    # time and is taken to be "zero". That is times[0] = 0.0
    times = get_times(data)
    assert len(times) == 12, "times array is not of length 12"

    # Extract angles from the data in degrees. "oppositions" is
    # a numpy array of length 12.
    oppositions = get_oppositions(data)
    assert len(oppositions) == 12, "oppositions array is not of length 12"

    # Call the top level function for optimization
    # The angles are all in degrees
    r, s, c, e1, e2, z, errors, maxError = bestMarsOrbitParams(
        times, oppositions
    )

    assert max(list(map(abs, errors))) == maxError, "maxError is not computed properly!"
    print(
        "Fit parameters: r = {:.4f}, s = {:.4f}, c = {:.4f}, e1 = {:.4f}, e2 = {:.4f}, z = {:.4f}".format(
            r, s, c, e1, e2, z
        )
    )
    print("The maximum angular error = {:2.4f}".format(maxError))