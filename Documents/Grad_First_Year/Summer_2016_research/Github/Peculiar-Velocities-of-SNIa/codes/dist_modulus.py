import numpy            as np
import matplotlib.pylab as plt
from   astropy.io       import ascii
import time


def generate_file():
    
    name      = []
    zcmb      = []
    zhel      = []
    ez        = []
    mB        = []
    emB       = []
    X1        = []
    eX1       = []
    C         = []
    eC        = []
    log10Mst  = []
    elog10Mst = []
    tmax      = []
    etmax     = []
    cov_mBs   = []
    cov_mBc   = []
    cov_sc    = []
    set       = []
    RA        = []
    DEC       = []
    bias      = []
    
    data = ascii.read("/Users/anita/Documents/Grad_First_Year/Summer_2016_research/data/J_A+A_568_A22_tablef3.dat.txt")

    for i in range(740):
        #namef      = data[i][0]
        zcmbf      = float(data[i][1])
        zhelf      = float(data[i][2])
        ezf        = float(data[i][3])
        mBf        = float(data[i][4])
        e_mBf      = float(data[i][5])
        X1f        = float(data[i][6])
        eX1f       = float(data[i][7])
        Cf         = float(data[i][8])
        eCf        = float(data[i][9])
        log10Mstf  = float(data[i][10])
        elog10Mstf = float(data[i][11])
        tmaxf      = float(data[i][12])
        etmaxf     = float(data[i][13])
        cov_mBsf   = float(data[i][14])
        cov_mBcf   = float(data[i][15])
        cov_scf    = float(data[i][16])
        setf       = float(data[i][17])
        RAf        = float(data[i][18][0:10])
        DECf       = float(data[i][18][12:22])
        biasf      = float(data[i][19])

        #name.append(namef)
        zcmb.append(zcmbf)
        zhel.append(zhelf)
        ez.append(ezf)
        mB.append(mBf)
        emB.append(e_mBf)
        X1.append(X1f)
        eX1.append(eX1f)
        C.append(Cf)
        eC.append(eCf)
        log10Mst.append(log10Mstf)
        elog10Mst.append(elog10Mstf)
        tmax.append(tmaxf)
        etmax.append(etmaxf)
        cov_mBs.append(cov_mBsf)
        cov_mBc.append(cov_mBcf)
        cov_sc.append(cov_scf)
        set.append(setf)
        RA.append(RAf)
        DEC.append(DECf)
        bias.append(biasf)

    #name      = np.array(name)
    zcmb      = np.array(zcmb)
    zhel      = np.array(zhel)
    ez        = np.array(ez)
    mB        = np.array(mB)
    emB       = np.array(emB)
    X1        = np.array(X1)
    eX1       = np.array(eX1)
    C         = np.array(C)
    eC        = np.array(eC)
    log10Mst  = np.array(log10Mst)
    elog10Mst = np.array(elog10Mst)
    tmax      = np.array(tmax)
    etmax     = np.array(etmax)
    cov_mBs   = np.array(cov_mBs)
    cov_mBc   = np.array(cov_mBc)
    cov_sc    = np.array(cov_sc)
    set       = np.array(set)
    RA        = np.array(RA)
    DEC       = np.array(DEC)
    bias      = np.array(bias)

    data = np.column_stack((zcmb,zhel,ez,mB,emB,X1,eX1,C,eC,log10Mst,elog10Mst,tmax,etmax,cov_mBs,cov_mBc,cov_sc,set,RA,DEC,bias))
    np.savetxt("table3_new.txt",data,fmt="%.9e")



def dist_modul(dirname, filename, ptype):
    '''
    Parameters
    -------------------------------------------------------------------------------------------------
        dirname:  direcoty where the file exists
    
        filename: filename
     
        ptype:    C11 parameter set (JLA or SALT2 for instance)
    
    Return
    -------------------------------------------------------------------------------------------------
        redshift with reference to cmb and the distance modulus value 
        computed using equation 4 in Betoule 2014 (http://arxiv.org/abs/1401.4064)
    '''

    data = np.loadtxt(dirname+filename)

    zcmb      = data[:,0]
    zhel      = data[:,1]
    ez        = data[:,2]
    mB        = data[:,3]
    e_mB      = data[:,4]
    X1        = data[:,5]
    eX1       = data[:,6]
    C         = data[:,7]
    eC        = data[:,8]
    log10Mst  = data[:,9]
    elog10Mst = data[:,10]
    tmax      = data[:,11]
    etmax     = data[:,12]
    cov_mBs   = data[:,13]
    cov_mBc   = data[:,14]
    cov_sc    = data[:,15]
    set       = data[:,16]
    RA        = data[:,17]
    DEC       = data[:,18]
    bias      = data[:,19]
    
   

    C11_Combined       = {'Omegam':0.228, 'alpha':1.434, 'beta':3.272, 'MB1':-19.16, 'del_MB1':-0.047}
    C11_SALT2_stat_sys = {'Omegam':0.249, 'alpha':1.708, 'beta':3.306, 'MB1':-19.15, 'del_MB1':-0.044}
    C11_SiFTO_stat_sys = {'Omegam':0.225, 'alpha':1.360, 'beta':3.401, 'MB1':-19.15, 'del_MB1':-0.047}
    C11_SALT2_stat     = {'Omegam':0.246, 'alpha':1.367, 'beta':3.133, 'MB1':-19.15, 'del_MB1':-0.065}
    C11_SiFTO_stat     = {'Omegam':0.272, 'alpha':1.366, 'beta':3.049, 'MB1':-19.12, 'del_MB1':-0.064}
    C11_reanalized     = {'Omegam':0.230, 'alpha':0.140, 'beta':2.771, 'MB1':-19.06, 'del_MB1':-0.053}
    C11_recalibrated   = {'Omegam':0.291, 'alpha':0.136, 'beta':2.907, 'MB1':-19.02, 'del_MB1':-0.061}
    JLA                = {'Omegam':0.289, 'alpha':0.140, 'beta':3.139, 'MB1':-19.04, 'del_MB1':-0.060}

    if ptype == "C11_Combined":
        alpha   = C11_Combined['alpha']
        beta    = C11_Combined['beta']
        MB1     = C11_Combined['MB1']
        del_MB1 = C11_Combined['del_MB1']
    
    elif ptype == "C11_SALT2_stat_sys":
        alpha   = C11_SALT2_stat_sys['alpha']
        beta    = C11_SALT2_stat_sys['beta']
        MB1     = C11_SALT2_stat_sys['MB1']
        del_MB1 = C11_SALT2_stat_sys['del_MB1']

    elif ptype == "C11_SiFTO_stat_sys":
        alpha   = C11_SiFTO_stat_sys['alpha']
        beta    = C11_SiFTO_stat_sys['beta']
        MB1     = C11_SiFTO_stat_sys['MB1']
        del_MB1 = C11_SiFTO_stat_sys['del_MB1']


    elif ptype == "C11_SALT2_stat":
        alpha   = C11_SALT2_stat['alpha']
        beta    = C11_SALT2_stat['beta']
        MB1     = C11_SALT2_stat['MB1']
        del_MB1 = C11_SALT2_stat['del_MB1']


    elif ptype == "C11_SiFTO_stat":
        alpha   = C11_SiFTO_stat['alpha']
        beta    = C11_SiFTO_stat['beta']
        MB1     = C11_SiFTO_stat['MB1']
        del_MB1 = C11_SiFTO_stat['del_MB1']


    elif ptype == "C11_reanalized":
        alpha   = C11_reanalized['alpha']
        beta    = C11_reanalized['beta']
        MB1     = C11_reanalized['MB1']
        del_MB1 = C11_reanalized['del_MB1']

    elif ptype == "C11_recalibrated":
        alpha   = C11_recalibrated['alpha']
        beta    = C11_recalibrated['beta']
        MB1     = C11_recalibrated['MB1']
        del_MB1 = C11_recalibrated['del_MB1']

    elif ptype == "JLA":
        alpha   = JLA['alpha']
        beta    = JLA['beta']
        MB1     = JLA['MB1']
        del_MB1 = JLA['del_MB1']


    # absolute magnitude as a step function
    MB = np.zeros(740)
    MB[log10Mst<10.]  = MB1
    MB[log10Mst>=10.] = MB1 + del_MB1

    # distance modulus
    mu    = mB - (MB - (alpha * X1) + (beta * C))
    return zcmb, mu



def dist_lumin(mu):
    '''
    Parameters
    -------------------------------------------------------------------------------------------------
        mu: distance modulus
        
    Return
    -------------------------------------------------------------------------------------------------
        Luminosity distance based on equation 2 in Kosowsky 2011 (http://arxiv.org/abs/1008.2560)
    '''

    dL = np.exp((mu-25)/2.17) # Mpc
    return dL


def DH(h):
    '''
    Hubble Distance
    '''
    return 3000./h



if __name__ == '__main__':
    
    # Plotting distance modulus as a function of redshift for all C11 parameters
    types   = ["C11_Combined", "C11_SALT2_stat_sys", "C11_SiFTO_stat_sys", "C11_SALT2_stat", "C11_SiFTO_stat", "C11_reanalized", "C11_recalibrated", "JLA"]
    dirname = "/Users/anita/Documents/Grad_First_Year/Summer_2016_research/Codes/"
    cvals   = ["black","blue","red","green","yellow","orange","magenta","teal"]

    plt.ion()
    
    for i in range(len(types)):
        zcmb, mu= dist_modul(dirname,"table3_new.txt",ptype=types[i])
        plt.plot(zcmb, mu,"o", ms=4, color=cvals[i], label=str(types[i]))
        plt.title(types[i])
        plt.pause(1.5)
        plt.draw()

    plt.xlabel("$z_{cmb}$", fontsize=25)
    plt.ylabel(r"$\mu = m_B - (M_B - \alpha X_1 + \beta C)$", fontsize=20)
    plt.legend(loc='best')












