from math import *    
from numpy import *
#%matplotlib qt
from matplotlib.pyplot import *
from matplotlib.widgets import Slider, Button, RadioButtons
from datetime import *


COUNTY_DATA = {datetime(2020,3,6):[1,0],
               datetime(2020,3,7):[1,0],
               datetime(2020,3,8):[1,0],
               datetime(2020,3,9):[1,0],
               datetime(2020,3,10):[2,0],
               datetime(2020,3,11):[1,0],
               datetime(2020,3,12):[1,0],
               datetime(2020,3,13):['u',1],
               datetime(2020,3,14):['u',1],
               datetime(2020,3,15):['u',1],
               datetime(2020,3,16):['u',1],
               datetime(2020,3,17):['u',1],
               datetime(2020,3,18):['u',1],
               datetime(2020,3,19):['u',2],
               datetime(2020,3,20):['u',2],
               datetime(2020,3,21):[37,3],
               datetime(2020,3,22):[51,3],
               datetime(2020,3,23):['u',3],
               datetime(2020,3,24):[106,3],
               datetime(2020,3,25):[122,5],
               datetime(2020,3,26):[137,7],
               datetime(2020,3,27):[160,7],
               datetime(2020,3,28):['u',10],
               datetime(2020,3,29):[245,10],
               datetime(2020,3,30):[286,11],
               datetime(2020,3,31):[314,13],
               datetime(2020,4,1):['u',14],
               datetime(2020,4,2):[340,16],
               datetime(2020,4,3):[374,18],
               datetime(2020,4,4):[406,22],
               datetime(2020,4,5):[435,25],
               datetime(2020,4,6):[441,28],
               datetime(2020,4,7):[457,28],
               datetime(2020,4,8):[472,30],
               datetime(2020,4,9):[534,32],
               datetime(2020,4,10):[550,33],
               datetime(2020,4,11):['u',35],
               datetime(2020,4,12):[613,37],
               datetime(2020,4,13):[641,39],
               datetime(2020,4,14):[652,41],
               datetime(2020,4,15):[668,43],
               datetime(2020,4,16):[689,48],
               datetime(2020,4,17):[708,49],
               datetime(2020,4,17):[721,49],
               datetime(2020,4,19):[731,49]
               }

def dataHandler(start_date=datetime(2020,3,1), conf_case_delay=7):
    # Handles the El Paso County data from Case_Data.py
    
    death_times=[]; case_times=[]; deaths = []; cases = []
    for date in COUNTY_DATA.keys():
        if not COUNTY_DATA[date][0] == 'u':
            cases.append(COUNTY_DATA[date][0])
        if not COUNTY_DATA[date][1] == 'u':
            deaths.append(COUNTY_DATA[date][1])
    for date in COUNTY_DATA.keys():
        time = (date-start_date).days + (date-start_date).seconds/86400
        if not COUNTY_DATA[date][0] == 'u':
            if time-conf_case_delay >= 0:
                case_times.append(time-conf_case_delay)
                case_times.sort()
            else:
                cases = cases[1:]
        if not COUNTY_DATA[date][1] == 'u':
            death_times.append(time)
            death_times.sort()
    
    return [death_times, case_times, deaths, cases]

def SIR_nu(today=datetime.now(), P=472688, I=1000, R=300, rho=2.8, tau=5, 
            nu=1.5,location='Colorado Springs', MaxDays = 100):

    # What is today?
    #today = datetime.now()
    print("                                 Today's Date/Time: " + str(today))

    # Total population of Colorado Springs
    #P = 472688
    # Number of Infected people, as of today - THIS IS AN ESTIMATE / ASSUMPTION
    #I = 1000
    # Number of Recovered people, as of today - THIS IS AN ESTIMATE / ASSUMPTION
    #R = 300
    # Number of Susceptible people, as of today - THIS IS AN ESTIMATE / ASSUMPTION
    S = P - (I + R)

    print("                      Population of " + location + ": " + str(P))
    print("           Assumed number of infected people today: " + str(I))
    print("          Assumed number of recovered people today: " + str(R))
    print("        Assumed number of susceptible people today: " + str(S))

    # Define the 'basic reproductive number,' essentially the 'infectiousness' value, for Coronavirus.
    # It is the average number of people infected by a direct contact with a sick person, before their recovery.
    # For reference, smallpox is between 3 and 5, measles is between 16 and 18.
    #rho = 2.8

    # Define the number of days a sick person is infectious.
    #tau = 5

    # Define the population mixing value. For reference, in cities this is between 1.7 and 2.06.
    #nu = 1.5

    print("      Average # people spread to by infected (rho): " + str(rho))
    print("  Assumed # days a sick person is infectious (tau): " + str(tau))
    print("             Assumed population mixing factor (nu): " + str(nu))

    # Let's normalize P, S, I, and R for simplicity of the equations.
    p = P/P # I know this is obviously 1, just bear with me
    s = S/P
    i = I/P
    r = R/P

    # Here are the differential equations governing the infection dynamics.
    ds_dt = -(rho/tau)*i*(s ** nu) # The two asterisks next to each other is how you write an exponent in python.
    di_dt = (rho/tau)*i*(s ** nu) - (i/tau)
    dr_dt = (i/tau) # It should be noted that this last equation is redundant


    # Let's create a time axis now. We will make it 100 days long, with an interval of 30 minutes.
    t = 1/48 # 30 minute interval as a fraction of a day.
    T = arange(0,MaxDays,t)

    # We could try to find the integral of s, i, and r analytically, but why would we do that when we have a computer?
    # Let's use a numerical integration technique, such as Euler's method.
    # x_next = x_current + t_interval * dx_dt_current, approximately.

    # First, we need to realize that what we have defined currently for s, i, & p (and their derivatives) 
    # are the starting values, so let's turn them into lists where these values are the starting values.
    p = [p]
    s = [s]
    i = [i]
    r = [r]
    ds_dt = [ds_dt]
    di_dt = [di_dt]
    dr_dt = [dr_dt]

    # Now we will step through the whole 100 days, appending the new values for p, s, i & r to their respective lists.

    for time in T[1:]:  # What is T[1:] ? This means every value in T except for the first one, T[0]. We already have those.
        i.append(i[-1] + t*di_dt[-1])   # What is i[-1]? This means the last value in the list, the 'current' value for this step.
        s.append(s[-1] + t*ds_dt[-1])   # .append() just adds a new value to the end of that list.
        r.append(r[-1] + t*dr_dt[-1])
        di_dt.append((rho/tau)*i[-1]*(s[-1] ** nu) - (i[-1]/tau))
        ds_dt.append(-(rho/tau)*i[-1]*(s[-1] ** nu))
        dr_dt.append((i[-1]/tau))
        p.append(i[-1] + s[-1] + r[-1])

    # Now plot all of our data, which now spans 100 days from March 24th.    
    figure()
    plot(T,i, label="Infected")
    plot(T,s, label="Susceptible")
    plot(T,r, label="Recovered")
    legend()
    grid()
    xlabel('Days from ' + str(today.date()))
    ylabel('Proportion of Population')

    # So, when is the peak infection date?
    peak_inf_index = i.index(max(i))
    peak_inf_days_from_now = peak_inf_index*t
    peak_date = today + timedelta(days=peak_inf_days_from_now)
    thresh_index = i.index(list(filter(lambda k: k > 0.001, i))[0])
    thresh_days_from_now = thresh_index*t
    thresh_date = today + timedelta(days=thresh_days_from_now)
    print("                               Peak Infection Date: " + str(peak_date.date()))
    print("                      Peak Infected Simultaneously: " + str(int(max(i)*P)))
    print("                  Proportion Who Will Get Infected: %.1f%%" % (100*r[-1]))
    print("       Date when location will reach 0.1% infected: " + str(thresh_date.date()))
    
def SEIR_nu(today=datetime.now(), P=472688, E=300, I=700, R=300, rho=2.8, 
            tau=5, nu=1.5, mu=3, location='Colorado Springs', 
            MaxDays = 100):

    # What is today?
    #today = datetime.now()
    print("                                Starting Date/Time: " + str(today))

    # Number of Susceptible people, as of date
    S = P - (E + I + R)

    print("                      Population of " + location + ": " + str(P))
    print("           Assumed number of infected people today: " + str(I))
    print("         Assumed number of incubating people today: " + str(E))
    print("          Assumed number of recovered people today: " + str(R))
    print("        Assumed number of susceptible people today: " + str(S))
    
    # Define the average incubation period. During this time, people are infected, but not infectious.
    #mu = 3

    print("      Average # people spread to by infected (rho): " + str(rho))
    print("                    Assumed incubation period (mu): " + str(mu))
    print("  Assumed # days a sick person is infectious (tau): " + str(tau))
    print("             Assumed population mixing factor (nu): " + str(nu))

    # Let's normalize P, S, I, and R for simplicity of the equations.
    e = [E/P]
    s = [S/P]
    i = [I/P]
    r = [R/P]

    # Here are the differential equations governing the infection dynamics.
    ds_dt = [-(rho/tau)*i[0]*(s[0] ** nu)] 
    de_dt = [(rho/tau)*i[0]*(s[0] ** nu) - (e[0]/mu)]
    di_dt = [e[0]/mu - (i[0]/tau)]
    dr_dt = [(i[0]/tau)] 

    t = 1/48 # 30 minute interval as a fraction of a day.
    T = arange(0,MaxDays,t)

    for time in T[1:]:  
        s.append(s[-1] + t*ds_dt[-1])
        e.append(e[-1] + t*de_dt[-1])
        i.append(i[-1] + t*di_dt[-1])                               
        r.append(r[-1] + t*dr_dt[-1])
        ds_dt.append(-(rho/tau)*i[-1]*(s[-1] ** nu))
        de_dt.append((rho/tau)*i[-1]*(s[-1] ** nu) - e[-1]/mu)
        di_dt.append(e[-1]/mu - (i[-1]/tau))
        dr_dt.append((i[-1]/tau))

    # Now plot all of our data, which now spans 100 days from March 24th.    
    figure()
    plot(T,i, label="Infected")
    plot(T,s, label="Susceptible")
    plot(T,r, label="Recovered")
    plot(T,e, label="Exposed")
    legend()
    grid()
    xlabel('Days from ' + str(today.date()))
    ylabel('Proportion of Population')

    # So, when is the peak infection date?
    peak_inf_index = i.index(max(i))
    peak_inf_days_from_now = peak_inf_index*t
    peak_date = today + timedelta(days=peak_inf_days_from_now)
    thresh_index = i.index(list(filter(lambda k: k > 0.001, i))[0])
    thresh_days_from_now = thresh_index*t
    thresh_date = today + timedelta(days=thresh_days_from_now)
    print("                               Peak Infection Date: " + str(peak_date.date()))
    print("                      Peak Infected Simultaneously: " + str(int(max(i)*P)))
    print("                  Proportion Who Will Get Infected: %.1f%%" % (100*r[-1]))
    print("        Date when El Paso will reach 0.1% infected: " + str(thresh_date.date()))
    
def NBD_SEIR(today=datetime(2020,3,6), p=720403, e=0, i=1, r=0, 
             rho=2.5, tau=5, k=1.5e-1, mu=3, 
             location='Colorado Springs', MaxDays = 100, 
             suppress_output=0, rho_sched={}, d=0, mort_rate=0.02, 
             symp_2_death=15):
    
    # NBD-SEIR model, taking into account heterogeneous mixing with a 
    # different method than the power law scaling "nu" value proposed
    # by Stroud, et al, instead treating "rho" as a random variable
    # drawn from a Negative Binary Distribution (combination of a 
    # Poisson and a gamma distribution).
    # This model is provided by L. Kong, et al, "Modeling Heterogeneity in 
    # Direct Infectious Disease Transmission in a Compartmental Model"

    # Define:
    # rho = reproductive number. Avg # of secondary infections per capita.
    # tau = average contagious period for an infected person.
    # (gamma = 1/tau; this is the rate of removal of infected)
    # (beta = rho/tau; this is the rate of transmission)
    # k = probability distibution shaping factor for theta.
    # (theta = average number of infected a susceptible with mix with)
    # mu = average incubation period for exposed person.
    # (alpha = 1/mu; this is the rate of exposed becoming infected)
    
    # 30 minute interval as a fraction of a day.
    t = 1/48 
    
    # Number of Susceptible people, as of date
    s = p - (e + i + r)

    # This section handles multiple rho values over time, optionally passed in the rho_sched dictionary
    init_rho_sched = {today: rho}
    rho_sched[today]=rho
    rho_dates = []
    for rho_date in rho_sched.keys():
        rho_dates.append(rho_date); rho_dates.sort()
        if rho_date > today + timedelta(days=MaxDays):
            print("Error: A date given in the R0 schedule (%s) exceeds the value for MaxDays ($g)." %
                                             (str(rho_date.date()), MaxDays))
    rho_Ts = []
    prev_rho_MaxDays = 0
    rho_dates = rho_dates[1:]
    for rho_date in rho_dates:
        rho_MaxDays = abs(rho_date - today).days + abs(rho_date - today).seconds/86400
        rho_Ts.append(arange(prev_rho_MaxDays, rho_MaxDays, t))
        prev_rho_MaxDays = rho_MaxDays
    rho_Ts.append(arange(prev_rho_MaxDays, MaxDays, t))
    
    
    if suppress_output==0:
        print("                                     Starting Date: " + str(today.date()))
        print("                      Population of " + location + ": " + str(p))
        print("       Assumed initial number of infectious people: " + str(i))
        print("       Assumed initial number of incubating people: " + str(e))
        print("        Assumed initial number of recovered people: " + str(r))
        print("      Assumed initial number of susceptible people: " + str(s))

        # Define the average incubation period. During this time, people are infected, but not infectious.
        #mu = 3

        #print("      Average # people spread to by infected (rho): " + str(rho))
        if len(rho_Ts) > 1:
            for date in rho_sched.keys():
                print("   On %s, the reproduction number (rho) is: %0.1f" % (str(date.date()),rho_sched[date]))
                print("                       transmission rate (beta) is: %0.2f" % (int(rho_sched[date])/tau))
        print("                    Assumed incubation period (mu): " + str(mu))
        print("  Assumed # days a sick person is infectious (tau): " + str(tau))
        print("                     Assumed recovery rate (gamma): " + str(1/tau))
        print("                  Assumed heterogeneity factor (k): " + str(k))
    
    # Calculate other parameters
    beta = rho/tau
    gamma = 1/tau
    
    # Let's normalize P, S, I, and R for simplicity of the equations.
    e = [e]
    s = [s]
    i = [i]
    r = [r]
    d = [d]
    NBD_mean = [beta*i[0]/p]
    m = [k/NBD_mean[-1]]
    NBD_var = [k*(1+m[-1])/(m[-1]**2)]

    # Here are the differential equations governing the infection dynamics.
    ds_dt = [-k*log(1+(rho*i[0])/(tau*k*p))*s[0]] 
    de_dt = [k*log(1+(rho*i[0])/(tau*k*p))*s[0] - (e[0]/mu)]
    di_dt = [e[0]/mu - (i[0]/tau)]
    dr_dt = [(i[0]/tau)] 

    for T in rho_Ts:
        rho = rho_sched[today + timedelta(T[0])]
        T = list(T)
        if 0 in T:
            T.remove(0)
        for time in T:  
            #print(time)
            s.append(s[-1] + t*ds_dt[-1])
            e.append(e[-1] + t*de_dt[-1])
            i.append(i[-1] + t*di_dt[-1])                               
            r.append(r[-1] + t*dr_dt[-1])
            if time > (symp_2_death-tau):
                d.append(r[-int((symp_2_death-tau)/t)]*mort_rate)
            else:
                d.append(0)
            ds_dt.append(-k*log(1+(rho*i[-1])/(tau*k*p))*s[-1])
            de_dt.append(k*log(1+(rho*i[-1])/(tau*k*p))*s[-1] - (e[-1]/mu))
            di_dt.append(e[-1]/mu - (i[-1]/tau))
            dr_dt.append((i[-1]/tau))
            NBD_mean.append(beta*i[-1]/p)
            m.append(k/NBD_mean[-1])
            NBD_var.append(k*(1+m[-1])/(m[-1]**2))
        
    T = []    
    for rho_T in rho_Ts:
        T = T + list(rho_T)
    
    # So, when is the peak infection date?
    peak_inf_index = i.index(max(i))
    peak_inf_days_from_now = peak_inf_index*t
    peak_date = today + timedelta(days=peak_inf_days_from_now)
    if max(i)>0.1*p:
        thresh_index = i.index(list(filter(lambda j: j > 0.001*p, i))[0])
        thresh_days_from_now = thresh_index*t
        thresh_date = today + timedelta(days=thresh_days_from_now)
        thresh_date = thresh_date.date()
    else:
        thresh_date = "N/A"    
        
    if suppress_output == 0:
        # Now plot all of our data, which now spans 100 days from March 24th.    
        figure()
        plot(T,s, label="Susceptible")
        plot(T,e, label="Exposed")
        plot(T,i, label="Infectious")
        plot(T,r, label="Removed")
        legend()
        grid()
        xlabel('Days from ' + str(today.date()))
        ylabel('Population')

        figure()
        plot(T,NBD_mean,label='Mean of NBD')
        plot(T,NBD_var, label='Variance of NBD')
        print("                               Peak Infection Date: " + str(peak_date.date()))
        print("                      Peak Infected Simultaneously: " + str(int(max(i))))
        print("                  Proportion Who Will Get Infected: %.1f%%" % (100*r[-1]/p))
        print("        Date when El Paso will reach 0.1% infected: " + str(thresh_date))

    pdf = [NBD_mean, NBD_var, k, m]
    return [T,s,e,i,r,peak_date,d,pdf]
    
def plot_NBD(NBD_mean=3,k=5,max_value=20,suppress_output=0):
    m = k/NBD_mean
    NBD_var = k*(1+m)/(m**2)
    p = 1/(1+m)
    lamb = k*(p/(1-p))
    def binom(a,b):
        return factorial(a)/(factorial(b)*factorial(a-b))
    X = arange(0,max_value,1)
    P_NBD = []
    P_Pois = []
    for x in X:
        P_NBD.append(binom(x+k-1,x)* (m/(m+1))**k * (1/(m+1))**x)
        P_Pois.append((lamb**x)*(exp(-lamb))/factorial(x))
    p = 1/(1+m)
    lamb = k*(p/(1-p))
    
    if suppress_output==0:
        for i in range(len(X)):
            if P_Pois[i] > P_NBD[i]:
                bar(X[i],P_Pois[i],color='b',alpha=0.4)
                bar(X[i],P_NBD[i],color='r',alpha=0.4)
            else: 
                bar(X[i],P_NBD[i],color='r',alpha=0.4)
                bar(X[i],P_Pois[i],color='b',alpha=0.4)
        bar(0,0,color='b',alpha=0.4,label='Homogeneous Mixing')
        bar(0,0,color='r',alpha=0.4,label='Heterogeneous Mixing')
        legend()
        title("Negative Binomial Distribution; \n \
        Mean = %0.1f / StdDev = %0.1f / k = %0.2f" % (NBD_mean, sqrt(NBD_var), k))
    return [X,P_NBD,P_Pois]
    
def vary_NBD_SEIR(today=datetime.now(), p=472688, e=300, i=700, r=300, 
             rho=2.8, tau=5, k_min=1e-10, k_max=1e10, mu=3, 
             location='Colorado Springs', MaxDays = 100):
    
    print("SEIR Model with heterogeneous mixing modeled as a random" + 
          " varible drawn from a negative binary distribution.")
    print("This distribution is a combination of a gamma and a Poisson distribution, " +
          "lower bounded by 0 and right-hand skewed.")
    print("The random variable is theta: the rate at which a susceptible person interacts " +
          "with an infected person.")
    print("The parameter describing the variance of this distribution is k. \nFor large k, " +
          "this distribution collapses to a discrete value, i.e. approximates homogeneity.\n")
    print("                 Location: " + location)
    print("            Starting Date: " + str(today.date()))
    print("      Starting Population: " + str(p))
    print("      Starting Incubating: " + str(e))
    print("        Starting Infected: " + str(i))
    print("       Starting Recovered: " + str(r))
    print("Basic Reproductive Number: " + str(rho))
    print("    Average Latent Period: " + str(mu))
    print("Average Infectious Period: " + str(tau))
    
    fig = figure()
    ax = axes(projection='3d')
    runs = arange(log10(k_min), log10(k_max)+1, (log10(k_max) - log10(k_min))/5)
    print(runs)
    I = []
    t = 1/48 # 30 minute interval as a fraction of a day.
    T = arange(0,MaxDays,t)
    for r in runs:
        output = NBD_SEIR(today,p,e,i,r,rho,tau,eval('1e'+str(int(r))),mu,location,MaxDays,suppress_output=1)
        I.append(output[3])
        ax.plot3D(T,r*ones(len(T)),I[-1])
        if not str(output[5] - today)[0:3] == "0:0":
            print("Peak Infected Date for k = " + str(eval('1e'+str(int(r)))) + 
                  ": " + str(output[-1].date()))
    xlabel(str('Days after ' + str(today.date())))
    ylabel('k = 1eX')
    ax.set_zlabel('Infected Population')
    
def est_Infected(pop, deaths_today, mort_rate = 1.38, symp_2_death = 17.8, mu = 3):
    today = datetime.now()
    inf_2_death = mu + symp_2_death    # 'mu' is the incubation period, i.e. time from infection to symptoms
    inf_date = today - timedelta(days=inf_2_death)
    inf_num = deaths_today/(mort_rate/100)
    print('There were %g infected on ' % floor(inf_num) + str(inf_date.date()) + '.')
    
    output = NBD_SEIR(suppress_output=1)
    T = output[0]
    s = output[1]
    e = output[2]
    i = output[3]
    r = output[4]
    
    if max(i) > inf_num:
        inf_ind = i.index(list(filter(lambda j: j >= inf_num, i))[0])
        
    print("On this date there were:")
    print("%g exposed" % floor(e[inf_ind]))
    print("%g recovered" % floor(r[inf_ind]))
    
    output = NBD_SEIR(inf_date, pop, e[inf_ind], inf_num, r[inf_ind])
    
def vary_Rho_NBD_SEIR(date=datetime(2020,3,1),pop=720403,e=3,i=10,r=2,start_rho=4,
                      lock_rho=[3.5,2.5,1.7],tau=8,k=0.03,mu=5.1,
                      loc="El Paso County",days=250):
    
    output1 = NBD_SEIR(date,pop,e,i,r,start_rho,tau,k,mu,loc,days,suppress_output=1,
                       rho_sched={datetime(2020,3,26):lock_rho[0]})
    output2 = NBD_SEIR(date,pop,e,i,r,start_rho,tau,k,mu,loc,days,suppress_output=1,
                       rho_sched={datetime(2020,3,26):lock_rho[1]})
    output3 = NBD_SEIR(date,pop,e,i,r,start_rho,tau,k,mu,loc,days,suppress_output=1,
                       rho_sched={datetime(2020,3,26):lock_rho[2]})

    T = output1[0]
    I_lax = output1[3]
    I_mid = output2[3]
    I_str = output3[3]

    T_lax = T[I_lax.index(max(I_lax))]
    T_mid = T[I_mid.index(max(I_mid))]
    T_str = T[I_str.index(max(I_str))]

    figure()
    plot(T,I_lax, label='Lax Measures')
    plot([T_lax, T_lax], [0,1.02*max(I_lax)], 'k--')
    plot(T,I_mid, label='Intermediate Measures')
    plot([T_mid, T_mid], [0,1.02*max(I_lax)], 'k--')
    plot(T,I_str, label='Strict Measures')
    plot([T_str, T_str], [0,1.02*max(I_lax)], 'k--')
    legend()
    months = ['          Mar',
              '          Apr',
              '          May',
              '          Jun',
              '          Jul',
              '          Aug',
              '          Sep',
              '          Oct',
              '          Nov']
    xticks([0,31,61,92,122,153,183,214,245],months)
    ylabel("Population")
    title("Infectious Population Over Time: Starting Rho = %g, \n \
            Lockdown Rho = %g, %g, and %g" % (start_rho, lock_rho[0],lock_rho[1],lock_rho[2]))
    return [T, I_lax, I_mid, I_str]
    
def calibrate_NBD_SEIR(cumu_inf, cumu_deaths, death_rate=0.02, death_period=20, 
                       today=datetime.now().date(), loc='El Paso County', pop=720403,
                       start_rho=4,lock_date=datetime(2020,3,26),lock_rho=3.0):
    t = 1/48
    start_date = datetime(2020,3,1)
    start_inf = 1
    tau = 8
    mu = 5.1
    k = 5
    date_delta = today-start_date.date()
    date_diff = date_delta.days
    output = NBD_SEIR(start_date,pop,0,start_inf,0,start_rho,tau,k,mu,loc,date_diff,
                      suppress_output=1,rho_sched={lock_date:lock_rho})
    I = output[3]
    dead_bynow = sum(I[int(-death_period/t)])*death_rate
    dead_diff = floor(abs(cumu_deaths - dead_bynow))
    while dead_diff > 0:
        if dead_bynow < cumu_deaths:
            start_inf = start_inf + start_inf/2
            output = NBD_SEIR(start_date,pop,0,start_inf,0,start_rho,tau,k,mu,loc,date_diff,
                              suppress_output=1,rho_sched={lock_date:lock_rho})
            I = output[3]
            R = output[4]
            dead_bynow = R[int(-(death_period-tau)/t)]*death_rate
            dead_diff = floor(abs(cumu_deaths - dead_bynow))
        elif dead_bynow > cumu_deaths:
            start_inf = start_inf - start_inf/2
            output = NBD_SEIR(start_date,pop,0,start_inf,0,start_rho,tau,k,mu,loc,date_diff,
                              suppress_output=1,rho_sched={lock_date:lock_rho})
            I = output[3]
            R = output[4]
            dead_bynow = R[int(-(death_period-tau)/t)]*death_rate
            dead_diff = floor(abs(cumu_deaths - dead_bynow))
    close('all')
    lock_range = [4,3.5,2.5]
    out0 = vary_Rho_NBD_SEIR(date=start_date,pop=pop,e=0,i=start_inf,r=0,start_rho=4,
                             lock_rho=lock_range,tau=8,k=k,mu=5.1,loc=loc,days=250)
    out1 = NBD_SEIR(start_date,pop,0,start_inf,0,lock_range[0],tau,k,mu,loc,date_diff,suppress_output=1)
    out2 = NBD_SEIR(start_date,pop,0,start_inf,0,lock_range[1],tau,k,mu,loc,date_diff,suppress_output=1)
    out3 = NBD_SEIR(start_date,pop,0,start_inf,0,lock_range[2],tau,k,mu,loc,date_diff,suppress_output=1)
    out4 = NBD_SEIR(start_date,pop,0,80,0,3.3,tau,k,mu,loc,250,suppress_output=1,
                    rho_sched={datetime(2020,3,20):1.7}, mort_rate=death_rate, symp_2_death=death_period)
    
    T = out1[0]
    [I1, R1] = [out1[3],out1[4]]
    [I2, R2] = [out2[3],out2[4]]
    [I3, R3] = [out3[3],out3[4]]
    
    data_times=[]
    dates = []
    cases = []
    death_data = []
    for date in COUNTY_DATA.keys():
        dates.append(date); dates.sort()
        data_times.append((date-start_date).days + (date-start_date).seconds/86400)
        data_times.sort()
    for date in dates:
        cases.append(COUNTY_DATA[date][0])
        death_data.append(COUNTY_DATA[date][1])
    D = out4[6]
    T = out4[0]
    figure()
    plot(T[0:int(48/t)],D[0:int(48/t)])
    #plot(data_times,cases,'bo')
    plot(data_times,death_data,'ro')
    
    print("Reported total number of deaths in %s as of %s: \033[1m%g deaths\033[0m" % 
                                                          (loc, str(today),cumu_deaths))
    print("                Reported total number of infected as of %s: \033[1m%g people\033[0m" % 
                                                          (str(today),cumu_inf))
    print("                                            Assumed mortality rate: \033[1m%g percent\033[0m" % 
                                                          (death_rate*100))
    print("                           Assumed time from infection until death: \033[1m%g days\033[0m" % 
                                                          death_period)
    print("               Assumed starting reproduction number until lockdown: \033[1m%0.1f\033[0m" %
                                                          (start_rho))
    print("                                             Assumed lockdown date: \033[1m%s\033[0m" %
                                                          (str(lock_date)))
    print("                        Assumed reproduction number after lockdown: \033[1m%0.1f\033[0m" %
                                                          (lock_rho))
    print("                Estimated instantaneous # infectious on %s: \033[1m%g people\033[0m\n" % 
                                                          (str(start_date.date()),start_inf))
    
    print("                                  \033[1m--- LAX MEASURES ---\033[0m   ")
    print("                Estimated instantaneous # infectious on %s: \033[1m%g people\033[0m" % 
                                                          (str(today), floor(I1[-1])))
    print("  Estimated actual CUMULATIVE number of recovered as of %s: \033[1m%g people\033[0m" %
                                                          (str(today),floor(R1[-1])))
    print("                       Estimated tested to actually infected ratio: \033[1m1 in %0.3g\033[0m\n" %
                                                          (floor(R1[-1])/cumu_inf))
    
    print("                               \033[1m--- INTERMEDIATE MEASURES ---\033[0m   ")
    print("                Estimated instantaneous # infectious on %s: \033[1m%g people\033[0m" % 
                                                          (str(today), floor(I2[-1])))
    print("  Estimated actual CUMULATIVE number of recovered as of %s: \033[1m%g people\033[0m" %
                                                          (str(today),floor(R2[-1])))
    print("                       Estimated tested to actually infected ratio: \033[1m1 in %0.2g\033[0m\n" %
                                                          (floor(R2[-1])/cumu_inf))
    
    print("                                 \033[1m--- STRICT MEASURES ---\033[0m   ")
    print("                Estimated instantaneous # infectious on %s: \033[1m%g people\033[0m" % 
                                                          (str(today), floor(I3[-1])))
    print("  Estimated actual CUMULATIVE number of recovered as of %s: \033[1m%g people\033[0m" %
                                                          (str(today),floor(R3[-1])))
    print("                       Estimated tested to actually infected ratio: \033[1m1 in %0.2g\033[0m\n" %
                                                          (floor(R3[-1])/cumu_inf))
    
def compare_data_NBD_SEIR(start_date = datetime(2020,3,1), 
                          media_date = datetime(2020,3,14), 
                          lock_date = datetime(2020,3,26), 
                          post_lock_date = datetime(2020,4,26),
                          conf_case_delay = 7, init_rho = 5, media_rho = 3.2, 
                          lock_rho = 2, post_lock_rho = [1,2,3], init_inf = 40, 
                          mort_rate = 0.013, symp_2_death = 15, MaxDays = 350,
                          tau = 8, k = 0.5, mu = 5.1, suppress_SEIR=1):
    if not (type(post_lock_rho) is list):
        post_lock_rho = [post_lock_rho]
    rho_sched = {media_date:media_rho, lock_date:lock_rho}
    t = 1/48
    loc = "El Paso County"
    out = NBD_SEIR(start_date,720403,0,init_inf,0,
                    init_rho,tau,k,mu,loc,MaxDays,suppress_output=suppress_SEIR,
                    rho_sched=rho_sched, 
                    mort_rate=mort_rate, symp_2_death=symp_2_death)
    data_times=[]; dates = []; cases = []; death_data = []
    for date in COUNTY_DATA.keys():
        dates.append(date); dates.sort()
        data_times.append((date-start_date).days + (date-start_date).seconds/86400)
        data_times.sort()
    for date in dates:
        if COUNTY_DATA[date][0] == 'u':
            cases.append(0)
        else:
            cases.append(COUNTY_DATA[date][0])
        death_data.append(COUNTY_DATA[date][1])

    death_dates = data_times
    case_dates = []
    for date in data_times:
        if date-conf_case_delay >= 0:
            case_dates.append(date-conf_case_delay)
        else:
            cases = cases[1:]
    I = out[3]
    D = out[6]
    T = out[0]
    NBD_means = out[7][0]
    Mean_weekly_now = 7*NBD_means[int((datetime.now()-start_date).days/t)]
    Mean_weekly_max = 7*max(NBD_means)
    weekly_now = plot_NBD(Mean_weekly_now,7*k,max_value=5,suppress_output=1)
    weekly_max = plot_NBD(Mean_weekly_max,7*k,max_value=5,suppress_output=1)
    media_time = (media_date-start_date).days
    lock_time = (lock_date-start_date).days
    post_lock_time = ((post_lock_date-start_date).days + 
                      (post_lock_date-start_date).seconds/86400) 
    outs = []
    for rho in post_lock_rho:
        rho_sched[post_lock_date]=rho
        outs.append(NBD_SEIR(start_date,720403,0,init_inf,0,
                        init_rho,tau,k,mu,loc,MaxDays,suppress_output=1,
                        rho_sched=rho_sched, 
                        mort_rate=mort_rate, symp_2_death=symp_2_death))
        del rho_sched[post_lock_date]
    rho_sched[post_lock_date] = post_lock_rho
    rho_sched[start_date] = init_rho
    
    months = ['          Mar',
              '          Apr',
              '          May',
              '          Jun',
              '          Jul',
              '          Aug',
              '          Sep',
              '          Oct',
              '          Nov']
    
    
    figure()
    
    subplot(2,2,1)
    plot(T[0:int(48/t)],D[0:int(48/t)],label="Predicted Deaths")
    plot(death_dates,death_data,'ro',label="Reported Deaths")
    ##
    for date in rho_sched.keys():
        day = (date-start_date).days + (date-start_date).seconds/86400
        plot([day, day],[0, max(D[0:int(48/t)])/3],'k')
        text(day-2.5,max(D[0:int(48/t)])/3,"R0=%s"%rho_sched[date])
    xticks([0,31,61],months)
    legend(); grid(); title("Predicted vs. Reported Deaths; \n \
                        Mortality Rate = %0.2f%%, %g days avg onset of symptoms until death" % 
                            (100*mort_rate, symp_2_death))
        
    subplot(2,2,2)
    plot(T,D,label="Predicted Deaths, R0 = %0.2f" % lock_rho)
    plot(death_dates,death_data,'ro',label="Reported Deaths")
    for rho in range(len(post_lock_rho)):
        plot(outs[rho][0][int((post_lock_date-start_date).days/t):],
             outs[rho][6][int((post_lock_date-start_date).days/t):],
             label="R0 = %0.2f"%post_lock_rho[rho])
    plot([post_lock_time, post_lock_time], [0, max(D)/3], 'k')
    text(post_lock_time-10, max(D)/3, "R0 = %s" % rho_sched[post_lock_date])
    xticks([0,31,61,92,122,153,183,214,245],months)
    legend(); grid(); title("Long-Term Predicted Deaths; \n \
                Incubation Period = %0.1f days / Infectious Period = %0.1f days" % (mu, tau))
    
    I = array(I)
    subplot(2,2,3)
    plot(T[0:int(48/t)],I[0:int(48/t)],label="Predicted Total Infectious")
    plot(T[0:int(48/t)],0.5*I[0:int(48/t)],'b--',label="Symptomatic (50%)")
    plot(case_dates,cases,'co',label="Confirmed Cases at Time of Testing")
    xticks([0,31,61],months)
    title("Predicted Infectious, Symptomatic, and Confirmed Cases; \n \
        Assumed 50%% symptomatic, %g day test result delay" % conf_case_delay)
    legend(); grid()
    ##
    for date in rho_sched.keys():
        day = (date-start_date).days + (date-start_date).seconds/86400
        plot([day, day],[0, max(I[0:int(48/t)])/3],'k')
        text(day-2.5,max(I[0:int(48/t)])/3,"R0=%s"%rho_sched[date])

    subplot(2,2,4)
    plot(T,I,label="Predicted Total Infectious, R0 = %0.2f" % lock_rho)
    plot(case_dates,cases,'co',label="Confirmed Cases at Time of Testing")
    for rho in range(len(post_lock_rho)):
        plot(outs[rho][0][int((post_lock_date-start_date).days/t):],
             outs[rho][3][int((post_lock_date-start_date).days/t):],
             label="R0 = %0.2f"%post_lock_rho[rho])
    plot([post_lock_time, post_lock_time], [0, max(I)/3], 'k')
    text(post_lock_time-10, max(I)/3, "R0 = %s" % rho_sched[post_lock_date])
    xticks([0,31,61,92,122,153,183,214,245],months)
    legend(); grid(); title("Long-Term Predicted Infectious")
    
    
    figure()
    subplot(1,2,1)
    bar(weekly_now[0],weekly_now[1],color='r',alpha=0.4,label='Heterogeneous Mixing')
    bar(weekly_now[0],weekly_now[2],color='b',alpha=0.4,label='Homogeneous Mixing')
    for x in range(len(weekly_now[0])):
        text(x-0.25,weekly_now[1][x]+0.03,"%g%%"%(100*weekly_now[1][x]),color='r',alpha=0.7)
        text(x-0.25,weekly_now[1][x]+0.01,"%g%%"%(100*weekly_now[2][x]),color='b',alpha=0.7)
    grid(); legend(); ylabel("Probability"); xlabel("# Infection Events Experienced")
    title("Number of Weekly Infection Events Experienced by Average Healthy Person: \n \
           Probability Distribution for Today, %s. \n \
           Mean: %0.3g." % (datetime.now().date(),Mean_weekly_now))
    
    subplot(1,2,2)
    bar(weekly_max[0],weekly_max[1],color='r',alpha=0.4,label='Heterogeneous Mixing')
    bar(weekly_max[0],weekly_max[2],color='b',alpha=0.4,label='Homogeneous Mixing')  
    for x in range(len(weekly_max[0])):
        text(x-0.25,weekly_max[1][x]+0.03,"%g%%"%(100*weekly_max[1][x]),color='r',alpha=0.7)
        text(x-0.25,weekly_max[1][x]+0.01,"%g%%"%(100*weekly_max[2][x]),color='b',alpha=0.7)
    grid(); legend();  ylabel("Probability"); xlabel("# Infection Events Experienced")
    title("Number of Weekly Infection Events Experienced by Average Healthy Person: \n \
          Probability Distribution when Infectious Population is at its Peak. \n \
          Mean: %0.3f. Peak date: %s." % (Mean_weekly_max, 
                                          str((start_date+timedelta(days=T[NBD_means.index(max(NBD_means))])).date())))

def Slider_NBD_SEIR(start_date = datetime(2020,3,1), lock_date = datetime(2020,3,26),
                    post_lock_date = datetime(2020,4,26), init_rho = 4,
                    lock_rho = 1.4, post_lock_rho = 3, init_inf = 40, 
                    conf_case_delay = 7, 
                    mort_rate = 0.013, symp_2_death = 13, MaxDays=350,
                    tau = 8, k = 0.5, mu = 5.1):
                        
    fig, ax = subplots()     
    subplots_adjust(left=0.1, bottom=0.3)
    t = 1/48; today = datetime.now()
    loc = "El Paso County"
    rho_sched = {lock_date:lock_rho, post_lock_date:post_lock_rho}
    lock_time =      (lock_date-start_date).days + \
                     (lock_date-start_date).seconds/86400
    post_lock_time = (post_lock_date-start_date).days + \
                     (post_lock_date-start_date).seconds/86400
    out = NBD_SEIR(start_date,720403,0,init_inf,0,
                   init_rho,tau,k,mu,loc,MaxDays,suppress_output=1,
                   rho_sched=rho_sched, 
                   mort_rate=mort_rate, symp_2_death=symp_2_death)
    
    T = out[0]; I = out[3]; D = out[6]
    
    [death_times, case_times, deaths, cases] = dataHandler(start_date, conf_case_delay)
        
        
    max_ind = int((today-start_date).days/t)
    # Top-Left Plot
    subplot(2,2,1)
    L1, = plot(T[0:max_ind],
               D[0:max_ind],lw=2,label="Predicted Deaths")
    death_scatter1 = plot(death_times,deaths,'ro',label="El Paso Reported Deaths")
    plot([lock_time, lock_time],[0,max(D[0:max_ind])/3],'k')
    text(lock_time-5, max(D[0:max_ind])/3+2, "CO Stay-at-Home order")
    title("Predicted vs. Reported Deaths in El Paso County")
    grid(); legend()
    
    # Top-Right Plot
    subplot(2,2,2)
    L2, = plot(T,D,lw=2,label="Predicted Deaths")
    death_scatter2 = plot(death_times,deaths,'ro',label="El Paso Reported Deaths")
    title("Predicted Deaths over the Long Term")
    grid(); legend()
    ax.margins(x=0)
    
    # Bottom-Left Plot
    subplot(2,2,3)
    I = array(I)
    L3, = plot(T[0:max_ind],
               I[0:max_ind],'c',lw=2,
               label="Predicted Infectious Population")         
    L3_1, = plot(T[0:max_ind],
               0.5*I[0:max_ind],'c-',lw=2,
               label="50% Infectious Population")
    L3_2, = plot(T[0:max_ind],
               0.33*I[0:max_ind],'c--',lw=2,
               label="33% Infectious Population")
    case_scatter, = plot(case_times,cases,'bo',label="El Paso Positive Test Results")      
    title("Predicted vs. Reported Infectious in El Paso County")
    grid(); legend()
    
    # Bottom-Right Plot
    subplot(2,2,4)
    L4, = plot(T,I,'c',lw=2,label="Predicted Infectious Population")
    case_scatter2, = plot(case_times,cases,'bo',
                          label="El Paso Positive Test Results")      
    title("Predicted Infectious over the Long Term")
    grid(); legend()
    
    
    delta = 0.01
    axcolor = 'lightgoldenrodyellow'
    # First column of sliders
    ax_init_rho =       axes([0.1, 0.200, 0.32, 0.015], facecolor=axcolor)
    s_init_rho =        Slider(ax_init_rho, 'Initial R\u2080', 
                             0.5, 6, valinit=init_rho, valstep=delta)
    ax_lock_rho =       axes([0.1, 0.175, 0.32, 0.015], facecolor=axcolor)
    s_lock_rho =        Slider(ax_lock_rho, 'Stay-at-Home R\u2080', 
                             0.5, 6, valinit=lock_rho, valstep=delta)
    ax_post_lock_rho =  axes([0.1, 0.150, 0.32, 0.015], facecolor=axcolor)
    s_post_lock_rho =   Slider(ax_post_lock_rho, 'R\u2080 After Lockdown', 
                             0.5, 6, valinit=post_lock_rho, valstep=delta)    
    ax_post_lock_date = axes([0.1, 0.125, 0.32, 0.015], facecolor=axcolor)
    s_post_lock_date =  Slider(ax_post_lock_date, 'Stay-at-Home End (Days after 1 Mar)', 
                             50, 200, valinit=post_lock_time, valstep=10*delta) 
    ax_init_inf =       axes([0.1, 0.100, 0.32, 0.015], facecolor=axcolor)
    s_init_inf =        Slider(ax_init_inf, 'Infected on %s'%start_date.date(), 
                             1, 100, valinit=init_inf, valstep=10*delta) 
    ax_init_exp =       axes([0.1, 0.075, 0.32, 0.015], facecolor=axcolor)
    s_init_exp =        Slider(ax_init_exp, 'Exposed on %s'%start_date.date(), 
                             1, 100, valinit=0, valstep=10*delta)     
    ax_init_rec =       axes([0.1, 0.050, 0.32, 0.015], facecolor=axcolor)
    s_init_rec =        Slider(ax_init_rec, 'Recovered on %s'%start_date.date(), 
                             1, 100, valinit=0, valstep=10*delta)             
    # Second column of sliders
    ax_tau =            axes([0.58, 0.200, 0.32, 0.015], facecolor=axcolor)
    s_tau =             Slider(ax_tau, 'Infectious Period (Days)', 
                             1, 14, valinit=tau, valstep=delta)    
    ax_mu =             axes([0.58, 0.175, 0.32, 0.015], facecolor=axcolor)
    s_mu =              Slider(ax_mu,  'Incubation Period', 
                            0, 7,  valinit=mu,  valstep=delta)       
    ax_k =              axes([0.58, 0.150, 0.32, 0.015], facecolor=axcolor)
    s_k =               Slider(ax_k,  'Mixing Homogeneity', 
                            0.01, 10,  valinit=k,  valstep=delta)     
    ax_mort_rate =      axes([0.58, 0.125, 0.32, 0.015], facecolor=axcolor)
    s_mort_rate =       Slider(ax_mort_rate,  'Mortality Rate (%)', 
                            0.1, 10,  valinit=100*mort_rate,  valstep=delta)  
    ax_symp_2_death =   axes([0.58, 0.100, 0.32, 0.015], facecolor=axcolor)
    s_symp_2_death =    Slider(ax_symp_2_death,  'Symptom Onset to Death (days)', 
                            3, 20,  valinit=symp_2_death,  valstep=5*delta)  
    ax_conf_case_delay= axes([0.58, 0.075, 0.32, 0.015], facecolor=axcolor)
    s_conf_case_delay = Slider(ax_conf_case_delay,  'Testing Delay (days)', 
                            0, 10,  valinit=conf_case_delay,  valstep=5*delta) 
                              

    def update(val):
        init_rho =      s_init_rho.val
        lock_rho =      s_lock_rho.val
        post_lock_rho = s_post_lock_rho.val
        post_lock_date= s_post_lock_date.val
        post_lock_time= post_lock_date
        post_lock_date= start_date + timedelta(days=post_lock_time)
        init_inf =      s_init_inf.val
        init_exp =      s_init_exp.val
        init_rec =      s_init_rec.val
        tau =           s_tau.val
        mu =            s_mu.val
        k =             s_k.val
        mort_rate =     s_mort_rate.val/100
        symp_2_death =  s_symp_2_death.val
        conf_case_delay=s_conf_case_delay.val
        [death_times, case_times, deaths, cases] = dataHandler(start_date, 
                                                    conf_case_delay)
        rho_sched = {lock_date:lock_rho,
                     post_lock_date:post_lock_rho}
        #print(rho_sched)
        out2 = NBD_SEIR(start_date,720403,init_exp,init_inf,init_rec,
                        init_rho,tau,k,mu,loc,MaxDays,suppress_output=1,
                        rho_sched=rho_sched, 
                        mort_rate=mort_rate, symp_2_death=symp_2_death)
        L1.set_xdata(out2[0][0:max_ind])
        L1.set_ydata(out2[6][0:max_ind])
        L2.set_xdata(out2[0])
        L2.set_ydata(out2[6])
        L3.set_xdata(out2[0][0:max_ind])
        L3.set_ydata(out2[3][0:max_ind])
        L3_1.set_xdata(out2[0][0:max_ind])
        L3_1.set_ydata(0.5*array(out2[3][0:max_ind]))
        L3_2.set_xdata(out2[0][0:max_ind])
        L3_2.set_ydata(0.33*array(out2[3][0:max_ind]))
        case_scatter.set_xdata(case_times)
        case_scatter.set_ydata(cases)
        L4.set_xdata(out2[0])
        L4.set_ydata(out2[3])
        case_scatter2.set_xdata(case_times)
        case_scatter2.set_ydata(cases)
        fig.canvas.draw_idle()
        
    s_init_rho.on_changed(update)    
    s_lock_rho.on_changed(update)
    s_post_lock_rho.on_changed(update)
    s_post_lock_date.on_changed(update)
    s_init_inf.on_changed(update) 
    s_init_exp.on_changed(update) 
    s_init_rec.on_changed(update) 
    s_tau.on_changed(update)
    s_mu.on_changed(update)
    s_k.on_changed(update)
    s_mort_rate.on_changed(update)
    s_symp_2_death.on_changed(update)
    s_conf_case_delay.on_changed(update)
    
    resetax = axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    
    def reset(event):
        s_init_rho.reset()    
        s_lock_rho.reset()
        s_post_lock_rho.reset()
        s_post_lock_date.reset()
        s_init_inf.reset() 
        s_init_exp.reset() 
        s_init_rec.reset() 
        s_tau.reset()
        s_mu.reset()
        s_k.reset()
        s_mort_rate.reset()
        s_symp_2_death.reset()
        s_conf_case_delay.reset()
        return button
    button.on_clicked(reset)
    
    
    #rax = axes([0.025, 0.5, 0.15, 0.15], facecolor='b')
    #radio = RadioButtons(rax, ('red', 'blue', 'green'), active=1)
    
    #def colorfunc(label):
    #    l.set_color(label)
    #    fig.canvas.draw_idle()
    #radio.on_clicked(colorfunc)
    
    show()
    
    
Slider_NBD_SEIR()
