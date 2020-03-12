import autograd.numpy as np

def chol_inv(L, y):
    tmp = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, tmp)

def erf(x):
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
                            
    # Save the sign of x
    sign = np.sign(x)
    x = np.abs(x)
                                                        
    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x**2)
                                                                    
    return sign*y

def normpdf(x):
    return np.exp(-x**2 / 2) / np.sqrt(2*np.pi)

def normcdf(x):
    return 0.5 + erf(x/np.sqrt(2)) / 2

def logphi(x):
    if x**2 < 0.0492:
        lp0 = -x/np.sqrt(2*np.pi)
        c = np.array([0.00048204, -0.00142906, 0.0013200243174, 0.0009461589032, -0.0045563339802, 0.00556964649138, 0.00125993961762116, -0.01621575378835404, 0.02629651521057465, -0.001829764677455021, 2*(1-np.pi/3), (4-np.pi)/3, 1, 1])
        f = 0
        for i in range(14):
            f = lp0*(c[i]+f)
        return -2*f-np.log(2)
    elif x < -11.3137:
        r = np.array([1.2753666447299659525, 5.019049726784267463450, 6.1602098531096305441, 7.409740605964741794425, 2.9788656263939928886])
        q = np.array([2.260528520767326969592, 9.3960340162350541504, 12.048951927855129036034, 17.081440747466004316, 9.608965327192787870698, 3.3690752069827527677])
        num = 0.5641895835477550741
        for i in range(5):
            num = -x*num/np.sqrt(2)+r[i]
        den = 1.0
        for i in range(6):
            den = -x*den/np.sqrt(2)+q[i]
        return np.log(0.5*np.maximum(0.000001,num/den))-0.5*(x**2)
    else:
        return np.log(0.5*np.maximum(0.000001,(1.0-erf(-x/np.sqrt(2)))))

# logphi_vector for autograd
def logphi_vector(x):
    # phi1
    lp0 = -x/np.sqrt(2*np.pi)
    c = np.array([0.00048204, -0.00142906, 0.0013200243174, 0.0009461589032, -0.0045563339802, 0.00556964649138, 0.00125993961762116, -0.01621575378835404, 0.02629651521057465, -0.001829764677455021, 2*(1-np.pi/3), (4-np.pi)/3, 1, 1])
    f = 0
    for i in range(14):
        f = lp0*(c[i]+f)
    phi1 = -2*f - np.log(2)

    # phi2 
    r = np.array([1.2753666447299659525, 5.019049726784267463450, 6.1602098531096305441, 7.409740605964741794425, 2.9788656263939928886])
    q = np.array([2.260528520767326969592, 9.3960340162350541504, 12.048951927855129036034, 17.081440747466004316, 9.608965327192787870698, 3.3690752069827527677])
    num = 0.5641895835477550741
    for i in range(5):
        num = -x*num/np.sqrt(2)+r[i]
    den = 1.0
    for i in range(6):
        den = -x*den/np.sqrt(2)+q[i]
    phi2 = np.log(0.5*np.maximum(0.000001,num/den))-0.5*(x**2)

    # phi3
    phi3 = np.log(0.5*np.maximum(0.000001,(1.0-erf(-x/np.sqrt(2)))))
    
    # phi
    x2 = x**2
    phi = phi1 * (x2 < 0.0492) + phi2 * (x < -11.3137) + phi3 * ((x >= -11.3137) | (x2 >= 0.0492))
    return phi





