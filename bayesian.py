import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp
import os
import time as t

def PA(Q_l, U_l):
    pa = []
    for i in range(0,len(Q_l)):
        pa.append( np.arctan(U_l[i]/Q_l[i])/2 )
    return np.rad2deg(pa) + 90

def PD(Q_l, U_l, I_obs):
    pd = []
    for i in range(0,len(I_obs)):
        pd.append( np.sqrt( (Q_l[i]**2) + (U_l[i]**2) ) / I_obs[i] )
    return pd

######################################################################################################################################################################################################
######################################################################################################################################################################################################

pf_new = np.loadtxt('data_OJ287.txt', delimiter=',')[:,3]
pd_new = np.loadtxt('data_OJ287.txt', delimiter=',')[:,1]
pa_new = np.loadtxt('data_OJ287.txt', delimiter=',')[:,2]
time   = np.loadtxt('data_OJ287.txt', delimiter=',')[:,0]


I_obs = []
for i in range(0,len(time)):
    I_obs.append( pf_new[i]/pd_new[i] )

Q_obs = []
for i in range(0,len(time)):
    Q_obs.append( pf_new[i]*np.cos( np.deg2rad(2*pa_new[i]) ) )

U_obs = []
for i in range(0,len(time)):
    U_obs.append( pf_new[i]*np.sin(np.deg2rad(2*pa_new[i])) )


x    = [  Q_obs  ,  U_obs  ,  I_obs ]

######################################################################################################################################################################################################
######################################################################################################################################################################################################
home = os.getcwd()


w = 0.5 #* (np.sqrt( np.power(np.std(Q_obs), 2)+np.power(np.std(U_obs), 2) ))


'''
# Random Q_l and U_l (With variation along time)
def y1():
    # initial random Q_l and U_l
    Q_l = []
    U_l = []
    for i in range(0,len(time)):
        Q_l.append( np.random.uniform(np.mean(Q_obs)-np.std(Q_obs),np.mean(Q_obs)+np.std(Q_obs)) ) #(min(Q_obs)),(max(Q_obs)*3)) )
        U_l.append( np.random.uniform(np.mean(U_obs)-np.std(U_obs),np.mean(U_obs)+np.std(U_obs)) )#(min(U_obs)*3),(max(U_obs)*3)) )
    #print(Q_l)
    return [ Q_l , U_l ]
'''

# Random Q_l and U_l (With no variation along time)
def y1():
    Q_l = [ np.random.uniform( np.mean(Q_obs)-np.std(Q_obs) , np.mean(Q_obs)+np.std(Q_obs) ) ]*len(time)
    U_l = [ np.random.uniform( np.mean(U_obs)-np.std(U_obs) , np.mean(U_obs)+np.std(U_obs) ) ]*len(time)
    return [ Q_l , U_l ]


def Z_score(A):
    A_norm = []
    for i in range(0, len(A)):
        A_norm.append( (A[i] - np.mean(A))/np.std(A) )
    return A_norm


# Likelihood
def L(y, x):
    I_obs_norm = Z_score(x[2])

    PF_s = []
    for i in range(0,len(x[0])):
        PF_s.append( np.sqrt( np.power(x[0][i]-y[0][i], 2) + np.power(x[1][i]-y[1][i], 2) ) )
    PF_s_norm = Z_score(PF_s)

    sigma_PF_s = np.std( PF_s_norm )

    L = []
    for i in range(0,len(x[2])):
        L.append( np.exp(-(np.power(I_obs_norm[i]-PF_s_norm[i],2)/(2*(sigma_PF_s**2)))) / np.sqrt(2*np.pi*(sigma_PF_s**2)) )
    return np.prod(L)


def PI(Q):
    b = []
    for i in range(1, len(Q)):
        b.append( ( np.exp( ( -( (Q[i]-Q[i-1])**2 ) / (2*(w**2)) ) ) / np.sqrt(2*np.pi*(w**2)) ) )
    for i in range(0,len(b)):    
        if b[i]==0:
            print(Q[i], 'aa', Q[i-1])
            print(w)
    return np.prod(b)


# Posterior Probability
def P(y, x):
    C = 6000
    return L(y,x)*PI(y[0])*PI(y[1])/C


'''
# If it is the first time running the code, execute this following lines
os.mkdir('{}/OJ/w={}'.format(home, w))
os.mkdir('{}/OJ/w={}/results_oj'.format(home, w))
'''

# sample in
sample_in = y1()


def MCMC(sample_in, num_samples, step = len(Q_obs), x = x):
    acceptance_rate = 0
    samples = []
    

    count = 0
    d = []


    acceptance_prob = [0]*num_samples
    acceptance_prob[0] = P( [np.random.normal(np.mean(sample_in[0]), scale=np.std(sample_in[0]), size=step),np.random.normal(np.mean(sample_in[1]), scale=np.std(sample_in[1]), size=step)] , x)
    
    j = 0
    while j <= num_samples:
        sample_out = y1()
        acceptance_prob[j] = P( [np.random.normal(np.mean(sample_out[0]), scale=np.std(sample_out[0]), size=step),np.random.normal(np.mean(sample_out[1]), scale=np.std(sample_out[1]),size=step)],x)
        
        
        #u = np.random.uniform(0,1)
        
        if ((acceptance_prob[j])>(acceptance_prob[j-1])):# or ((acceptance_prob[j]/acceptance_prob[j-1])>u):
            count = 0
            print('prob: ', acceptance_prob[j], '	iteration: ', j, '			w={}'.format(w), '	time: ', (t.time()-start_time)/60, 'min' )
            acceptance_rate += 1

            Q_l_new = sample_out[0]
            U_l_new = sample_out[1]

            d.append( [time, Q_l_new, U_l_new] )

            np.savetxt('{}/OJ/constant_UQ/w={}/results_oj/Stokes_param_{}.txt'.format(home, w, j), np.c_[time, Q_l_new, U_l_new], delimiter=';', header='time;Q_l;U_l')
            j += 1
        
        else:
            count += 1
            if count%10000==0:
                print('count: ', count, '	w={}'.format(w), '	time: ', (t.time()-start_time)/60, 'min')
        
        if count == 25000000:
            break
    
    I_l = []
    for i in range(0,len(Q_l_new)):
        I_l.append( np.sqrt( (Q_l_new[i]**2) + (U_l_new[i]**2) ) )
    
        
    return Q_l_new , U_l_new, I_l, acceptance_prob, acceptance_rate, d

start_time = t.time()

Q_l_new, U_l_new, I_l, acceptance_prob, acceptance_rate, total_values_l = MCMC(sample_in, int(100000+10000))

print('elapsed time: {} min'.format((t.time()-start_time)/60))


Q_s_new = []
U_s_new = []
for i in range(0, len(total_values_l[0])):
    Q_s_new.append(Q_obs[i]-Q_l_new[i])
    U_s_new.append(U_obs[i]-U_l_new[i])



total_values_s = []
for i in range(0, len(total_values_l)):
    Q_s_total = []
    U_s_total = []
    for j in range(0, len(time)):
        Q_s_total.append( Q_obs[j] - total_values_l[i][1][j] )
        U_s_total.append( U_obs[j] - total_values_l[i][2][j] )
    total_values_s.append( [time, Q_s_total, U_s_total] )


d_l = []
d_l_2_initial = []
d_s = []
d_s_2_initial = []
for i in range(0, len(total_values_l)):
    d_l_ = []
    d_l__ = []
    d_s_ = []
    d_s__ = []
    for j in range(1, len(total_values_l[0][0])):
        d_l_.append( np.sqrt( np.power(total_values_l[i][1][j]-total_values_l[i][1][j-1], 2) + np.power(total_values_l[i][2][j]-total_values_l[i][2][j-1], 2) ) )
        d_l__.append( np.sqrt( np.power(total_values_l[i][1][j]-total_values_l[i][1][0], 2) + np.power(total_values_l[i][2][j]-total_values_l[i][2][0], 2) ) )
        d_s_.append( np.sqrt( np.power(total_values_s[i][1][j]-total_values_s[i][1][j-1], 2) + np.power(total_values_s[i][2][j]-total_values_s[i][2][j-1], 2) ) )
        d_s__.append( np.sqrt( np.power(total_values_s[i][1][j]-total_values_s[i][1][0], 2) + np.power(total_values_s[i][2][j]-total_values_s[i][2][0], 2) ) )
    d_l.append( np.sum(d_l_) )
    d_l_2_initial.append( np.sum(d_l__) )
    d_s.append( np.sum(d_s_) )
    d_s_2_initial.append( np.sum(d_s__) )


print('R: ', d_l[-1]/d_s[-1])

print('PA_l: ', np.mean( PA(Q_l_new, U_l_new) ), 'PA_l_err: ', np.std( PA(Q_l_new, U_l_new) ) )
print('PD_l: ', np.mean(PD(Q_l_new, U_l_new, I_obs)[1]), 'PD_l_err: ', np.std(PD(Q_l_new, U_l_new, I_obs)[1] ) )
print('PF_l: ', np.mean(PF(Q_l_new, U_l_new)), 'PF_l_err: ', np.std(PF(Q_l_new, U_l_new) ) )


'''
# Results

w=0.2
elapsed time: 2644.086145313581 min
prob:  0.860557201634505 			iteration:  41			w=0.2 		time:  678.1601667443912 min

w=0.3
elapsed time: 2229.9908804098764 min
prob:  8.5078709357055e-26 			iteration:  28 		w=0.3 		time:  155.10287055969238 min

w=0.4
elapsed time: 3893.6329302152 min
prob:  1.5435487893267976e-43 		iteration:  39		 	w=0.4 		time:  2048.593453705311 min

w=0.5
elapsed time: 4030.372562221686 min
prob:  2.674844323394462e-57 			iteration:  46 		w=0.5 		time:  2309.9878038724264 min
'''
