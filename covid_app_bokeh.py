''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''

from math import sqrt, log, factorial, exp
from numpy import arange, array, dot, mod, linspace, pi, sin
import matplotlib
from matplotlib.pyplot import subplots, subplots_adjust, subplot, figure, \
                              axes, text, bar, plot, legend, grid, title, \
                              xlabel, ylabel, show, ylim, xticks
from matplotlib.widgets import Slider, Button, RadioButtons
from datetime import datetime, timedelta 
# Bokeh imports
from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Slider, TextInput, RadioButtonGroup, \
                           Button
from bokeh.events import ButtonClick
from bokeh.plotting import figure
from bokeh.models.ranges import DataRange1d

COUNTY_DATA = {datetime(2020,3,6):[1,0], 
               datetime(2020,3,7):[1,0],
               datetime(2020,3,8):[1,0],
               datetime(2020,3,9):[1,0],
               datetime(2020,3,10):[2,0],
               datetime(2020,3,11):[1,0],
               datetime(2020,3,12):[1,0],               
               datetime(2020,3,13):[2,1],
               datetime(2020,3,14):[3,1],
               datetime(2020,3,15):[4,1],
               datetime(2020,3,16):[4,1],
               datetime(2020,3,17):[6,1],
               datetime(2020,3,18):[6,1],
               datetime(2020,3,19):[15,2],
               datetime(2020,3,20):[27,2],
               datetime(2020,3,21):[37,3],
               datetime(2020,3,22):[51,3],
               datetime(2020,3,23):[69,3],
               datetime(2020,3,24):[106,3],
               datetime(2020,3,25):[122,5],
               datetime(2020,3,26):[137,7],
               datetime(2020,3,27):[160,7],
               datetime(2020,3,28):[184,10],
               datetime(2020,3,29):[214,10],
               datetime(2020,3,30):[245,11],
               datetime(2020,3,31):[286,13],
               datetime(2020,4,1):[314,14],
               datetime(2020,4,2):[340,16],
               datetime(2020,4,3):[374,18],
               datetime(2020,4,4):[406,22],
               datetime(2020,4,5):[435,25],
               datetime(2020,4,6):[441,28],
               datetime(2020,4,7):[457,28],
               datetime(2020,4,8):[472,30],
               datetime(2020,4,9):[534,32],
               datetime(2020,4,10):[550,33],
               datetime(2020,4,11):[578,35],
               datetime(2020,4,12):[613,37],
               datetime(2020,4,13):[641,39],
               datetime(2020,4,14):[652,41],
               datetime(2020,4,15):[668,43],
               datetime(2020,4,16):[689,48],
               datetime(2020,4,17):[708,49],
               datetime(2020,4,18):[721,49],
               datetime(2020,4,19):[731,49],
               datetime(2020,4,20):[734,50],
               datetime(2020,4,21):[744,53],
               datetime(2020,4,22):[774,54],
               datetime(2020,4,23):[798,55],
               datetime(2020,4,24):[853,'u'], # Was 72, but state corrected due to 
               datetime(2020,4,25):[862,66],  # duplicates; unclear how many were El Paso
               datetime(2020,4,26):[867,68],
               datetime(2020,4,27):[879,68],
               datetime(2020,4,28):[884,69],
               datetime(2020,4,29):[907,69],
               datetime(2020,4,30):[932,70],
               datetime(2020,5, 1):[964,71],
               datetime(2020,5, 2):[988,74],
               datetime(2020,5, 3):[994,75],
               datetime(2020,5, 4):[1005,76],
               datetime(2020,5, 5):[1028,77],
               datetime(2020,5, 6):[1055,77],
               datetime(2020,5, 7):[1079,77],
               datetime(2020,5, 8):[1097,78],
               datetime(2020,5, 9):[1109,78],
               datetime(2020,5, 10):[1124,78],
               datetime(2020,5, 11):[1137,79],
               datetime(2020,5, 12):[1157,81],
               datetime(2020,5, 13):[1175,81],
               datetime(2020,5, 14):[1204,81],
               datetime(2020,5, 15):[1251,83],
               datetime(2020,5, 16):[1291,83],
               datetime(2020,5, 17):[1314,83],
               datetime(2020,5, 18):[1348,83],
               datetime(2020,5, 19):[1357,84],
               datetime(2020,5, 20):[1386,88],
               datetime(2020,5, 21):[1428,88],
               datetime(2020,5, 22):[1460,88],
               datetime(2020,5, 23):[1493,88],
               datetime(2020,5, 24):[1505,88],
               datetime(2020,5, 25):[1535,88],
               datetime(2020,5, 26):[1581,88],
               datetime(2020,5, 26):[1600,88],
               }                                

# https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/

months = ['          Mar',
          '          Apr',
          '          May',
          '          Jun',
          '          Jul',
          '          Aug',
          '          Sep',
          '          Oct',
          '          Nov']

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

def SIR_nu(start_date=datetime(2020,3,1), P=472688, I=1000, R=300, rho=1, tau=5, 
           nu=1.5,loc='Colorado Springs', MaxDays = 100, suppress_output=0,
           rho_sched = {}, d=0, mort_rate=0.02,symp_2_death=15, t=1/48):

    # Handle changing reproduction number over time
    rho_dates = []
    rho_sched[start_date] = rho
    for rho_date in rho_sched.keys():
        rho_dates.append(rho_date); rho_dates.sort()
        if rho_date > start_date + timedelta(days=MaxDays):
            print("Error: A date given in the R0 schedule (%s) exceeds the value for MaxDays ($g)." %
                                             (str(rho_date.date()), MaxDays))
    rho_Ts = []
    prev_rho_MaxDays=0
    rho_dates = rho_dates[1:]
    for rho_date in rho_dates:
        rho_MaxDays = abs(rho_date - start_date).days + abs(rho_date - start_date).seconds/86400
        rho_Ts.append(arange(prev_rho_MaxDays, rho_MaxDays, t))
        prev_rho_MaxDays = rho_MaxDays
    rho_Ts.append(arange(prev_rho_MaxDays, MaxDays, t))
    
    S = P - (I + R)
    
    if suppress_output ==0:
        print("                               Starting Date/Time: " + str(start_date))
        
        print("                      Population of " + loc + ": " + str(P))
        print("           Assumed initial number of infected people: " + str(I))
        print("          Assumed initial number of recovered people: " + str(R))
        print("        Assumed initial number of susceptible people: " + str(S))

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
    d = [d]
    ds_dt = [ds_dt]
    di_dt = [di_dt]
    dr_dt = [dr_dt]

    # Now we will step through the whole 100 days, appending the new values for p, s, i & r to their respective lists.
    for T in rho_Ts:
        rho = rho_sched[start_date + timedelta(T[0])]
        T = list(T)
        if 0 in T:
            T.remove(0)
        for time in T:  # What is T[1:] ? This means every value in T except for the first one, T[0]. We already have those.
            i.append(i[-1] + t*di_dt[-1])   # What is i[-1]? This means the last value in the list, the 'current' value for this step.
            s.append(s[-1] + t*ds_dt[-1])   # .append() just adds a new value to the end of that list.
            r.append(r[-1] + t*dr_dt[-1])
            if time > (symp_2_death-tau):
                # Deaths are a fraction of the recovered population in the past.
                d.append(r[-int((symp_2_death-tau)/t)]*mort_rate)
            else: # If too early, nobody has died yet.
                d.append(0)
            di_dt.append((rho/tau)*i[-1]*(s[-1] ** nu) - (i[-1]/tau))
            ds_dt.append(-(rho/tau)*i[-1]*(s[-1] ** nu))
            dr_dt.append((i[-1]/tau))
            p.append(i[-1] + s[-1] + r[-1])

    T = [] # Stitch together the timeframes of different rho values
    for rho_T in rho_Ts:
        T = T + list(rho_T)
        
    if suppress_output==0:
        # Now plot all of our data, which now spans 100 days from March 24th.    
        figure()
        plot(T,i, label="Infected")
        plot(T,s, label="Susceptible")
        plot(T,r, label="Recovered")
        legend()
        grid()
        xlabel('Days from ' + str(start_date.date()))
        ylabel('Proportion of Population')

        # So, when is the peak infection date?
        peak_inf_index = i.index(max(i))
        peak_inf_days_from_now = peak_inf_index*t
        peak_date = start_date + timedelta(days=peak_inf_days_from_now)
        thresh_index = i.index(list(filter(lambda k: k > 0.001, i))[0])
        thresh_days_from_now = thresh_index*t
        thresh_date = start_date + timedelta(days=thresh_days_from_now)
        print("                               Peak Infection Date: " + str(peak_date.date()))
        print("                      Peak Infected Simultaneously: " + str(int(max(i)*P)))
        print("                  Proportion Who Will Get Infected: %.1f%%" % (100*r[-1]))
        print("       Date when location will reach 0.1% infected: " + str(thresh_date.date()))
        
    s = array(s)*P  
    i = array(i)*P
    r = array(r)*P
    d = array(d)*P
    return [T,s,i,r,d]
    
def SEIR_nu(start_date=datetime(2020,3,1), P=472688, E=300, I=700, R=300, rho=2.8, 
            tau=5, nu=1.5, mu=3, loc='Colorado Springs', MaxDays = 100, 
            suppress_output=0, rho_sched={}, d=0, mort_rate=0.02,
            symp_2_death=15,t=1/48):

    # Handle changing reproduction number over time
    rho_dates = []
    rho_sched[start_date] = rho
    for rho_date in rho_sched.keys():
        rho_dates.append(rho_date); rho_dates.sort()
        if rho_date > start_date + timedelta(days=MaxDays):
            print("Error: A date given in the R0 schedule (%s) exceeds the value for MaxDays ($g)." %
                                             (str(rho_date.date()), MaxDays))
    rho_Ts = []
    prev_rho_MaxDays=0
    rho_dates = rho_dates[1:]
    for rho_date in rho_dates:
        rho_MaxDays = abs(rho_date - start_date).days + abs(rho_date - start_date).seconds/86400
        rho_Ts.append(arange(prev_rho_MaxDays, rho_MaxDays, t))
        prev_rho_MaxDays = rho_MaxDays
    rho_Ts.append(arange(prev_rho_MaxDays, MaxDays, t))
    
    S = P - (E + I + R)
    
    if suppress_output==0:
        print("                                Starting Date/Time: " + str(start_date))

        print("                      Population of " + loc + ": " + str(P))
        print("           Assumed number of infected people today: " + str(I))
        print("         Assumed number of incubating people today: " + str(E))
        print("          Assumed number of recovered people today: " + str(R))
        print("        Assumed number of susceptible people today: " + str(S))
        
        print("      Average # people spread to by infected (rho): " + str(rho))
        print("                    Assumed incubation period (mu): " + str(mu))
        print("  Assumed # days a sick person is infectious (tau): " + str(tau))
        print("             Assumed population mixing factor (nu): " + str(nu))

    # Let's normalize P, S, I, and R for simplicity of the equations.
    e = [E/P]
    s = [S/P]
    i = [I/P]
    r = [R/P]
    d = [d]

    # Here are the differential equations governing the infection dynamics.
    ds_dt = [-(rho/tau)*i[0]*(s[0] ** nu)] 
    de_dt = [(rho/tau)*i[0]*(s[0] ** nu) - (e[0]/mu)]
    di_dt = [e[0]/mu - (i[0]/tau)]
    dr_dt = [(i[0]/tau)] 

    T = arange(0,MaxDays,t)

    for T in rho_Ts:
        rho = rho_sched[start_date + timedelta(T[0])]
        T = list(T)
        if 0 in T:
            T.remove(0)
        for time in T:  
            s.append(s[-1] + t*ds_dt[-1])
            e.append(e[-1] + t*de_dt[-1])
            i.append(i[-1] + t*di_dt[-1])                               
            r.append(r[-1] + t*dr_dt[-1])
            if time > (symp_2_death-tau):
                # Deaths are a fraction of the recovered population in the past.
                d.append(r[-int((symp_2_death-tau)/t)]*mort_rate)
            else: # If too early, nobody has died yet.
                d.append(0)
            ds_dt.append(-(rho/tau)*i[-1]*(s[-1] ** nu))
            de_dt.append((rho/tau)*i[-1]*(s[-1] ** nu) - e[-1]/mu)
            di_dt.append(e[-1]/mu - (i[-1]/tau))
            dr_dt.append((i[-1]/tau))

    T = [] # Stitch together the timeframes of different rho values
    for rho_T in rho_Ts:
        T = T + list(rho_T)

    if suppress_output==0:
        # Now plot all of our data, which now spans 100 days from March 24th.    
        figure()
        plot(T,i, label="Infected")
        plot(T,s, label="Susceptible")
        plot(T,r, label="Recovered")
        plot(T,e, label="Exposed")
        legend()
        grid()
        xlabel('Days from ' + str(start_date.date()))
        ylabel('Proportion of Population')

        # So, when is the peak infection date?
        peak_inf_index = i.index(max(i))
        peak_inf_days_from_now = peak_inf_index*t
        peak_date = start_date + timedelta(days=peak_inf_days_from_now)
        thresh_index = i.index(list(filter(lambda k: k > 0.001, i))[0])
        thresh_days_from_now = thresh_index*t
        thresh_date = start_date + timedelta(days=thresh_days_from_now)
        print("                               Peak Infection Date: " + str(peak_date.date()))
        print("                      Peak Infected Simultaneously: " + str(int(max(i)*P)))
        print("                  Proportion Who Will Get Infected: %.1f%%" % (100*r[-1]))
        print("        Date when El Paso will reach 0.1% infected: " + str(thresh_date.date()))
    
    s = array(s)*P  
    e = array(e)*P
    i = array(i)*P
    r = array(r)*P
    d = array(d)*P
    return [T,s,e,i,r,d]
    
def NBD_SEIR(start_date=datetime(2020,3,6), p=720403, e=0, i=1, r=0, 
             rho=2.5, tau=5.1, k=1.5e-1, mu=3, 
             loc='Colorado Springs', MaxDays = 100, 
             suppress_output=0, rho_sched={}, d=0, mort_rate=0.02, 
             symp_2_death=15, t=1/48, county='El Paso'):
    
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
    
    # Number of Susceptible people, as of date
    s = p - (e + i + r)

    # This section handles multiple rho values over time, optionally passed in the rho_sched dictionary
    rho_sched[start_date]=rho
    rho_dates = []
    for rho_date in rho_sched.keys():
        rho_dates.append(rho_date); rho_dates.sort()
        if rho_date > start_date + timedelta(days=MaxDays):
            print("Error: A date given in the R0 schedule (%s) exceeds the value for MaxDays ($g)." %
                                             (str(rho_date.date()), MaxDays))
    rho_Ts = []
    prev_rho_MaxDays = 0
    rho_dates = rho_dates[1:]
    for rho_date in rho_dates:
        rho_MaxDays = abs(rho_date - start_date).days + abs(rho_date - start_date).seconds/86400
        rho_Ts.append(arange(prev_rho_MaxDays, rho_MaxDays, t))
        prev_rho_MaxDays = rho_MaxDays
    rho_Ts.append(arange(prev_rho_MaxDays, MaxDays, t))
    
    
    if suppress_output==0:
        print("                                     Starting Date: " + str(start_date.date()))
        print("                      Population of " + loc + ": " + str(p))
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
        rho = rho_sched[start_date + timedelta(T[0])]
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
    
    peak_date = "Set suppress_output = 0 to see the peak infectious date."
        
    if suppress_output == 0:
        # So, when is the peak infection date?
        peak_inf_index = i.index(max(i))
        peak_inf_days_from_now = peak_inf_index*t
        peak_date = start_date + timedelta(days=peak_inf_days_from_now)
        if max(i)>0.1*p:
            thresh_index = i.index(list(filter(lambda j: j > 0.001*p, i))[0])
            thresh_days_from_now = thresh_index*t
            thresh_date = start_date + timedelta(days=thresh_days_from_now)
            thresh_date = thresh_date.date()
        else:
            thresh_date = "N/A"    
    
        # Now plot all of our data, which now spans 100 days from March 24th.    
        figure()
        plot(T,s, label="Susceptible")
        plot(T,e, label="Exposed")
        plot(T,i, label="Infectious")
        plot(T,r, label="Removed")
        legend()
        grid()
        xlabel('Days from ' + str(start_date.date()))
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

def slider_app_bokeh(start_date = datetime(2020,3,1), lock_date = datetime(2020,3,26),
                    post_lock_date = datetime(2020,4,26), init_rho = 3.7,
                    lock_rho = 0.7, post_lock_rho = 1.6, init_inf = 61, 
                    conf_case_delay = 7, init_exp=0, init_rec=0,
                    mort_rate = 0.013, symp_2_death = 14, MaxDays=500,
                    tau = 8, k = 0.5, nu=1.7, mu = 5.1, model='NBD_SEIR', 
                    county='El Paso'):
                    

    models = {'SIR_nu':0, 'SEIR_nu':1, 'NBD_SEIR':2}
    model_text = {'SIR_nu':'Simple S-I-R Infection Model with \n'+
                           'Heterogeneous Mixing Simulated using \n'+
                           'a Power-Law Scaling Term (''nu'')',
                  'SEIR_nu':'S-E-I-R Infection Model with \n'+
                            'Heterogeneous Mixing Simulated using \n'+
                            'a Power-Law Scaling Term (''nu'')',
                  'NBD_SEIR':'S-E-I-R Infection Model with \n'+
                             'Heterogeneous Mixing Simulated using \n'+
                             'Negative Binomially-Distributed \n'+
                             'Infection Events (PDF shape ''k'')'}

    t = 7/24; today = datetime.now()
    pct_tst = 0.15
    global zm_toggle, legend_toggle
    zm_toggle = -1
    legend_toggle = 1
    rho_sched = {lock_date:lock_rho, post_lock_date:post_lock_rho}
    lock_time =      (lock_date-start_date).days + \
                     (lock_date-start_date).seconds/86400
    post_lock_time = (post_lock_date-start_date).days + \
                     (post_lock_date-start_date).seconds/86400

    if model == 'SIR_nu':
        I_ind = 2; R_ind = 3; D_ind = 4
        out = SIR_nu(start_date,P=720403,I=init_inf,R=0,
                   rho=init_rho,tau=tau,nu=1.7,MaxDays=MaxDays,suppress_output=1,
                   rho_sched=rho_sched, mort_rate=mort_rate, symp_2_death=symp_2_death,t=t)
    elif model == 'SEIR_nu':
        I_ind = 3; R_ind = 4; D_ind = 5
        out = SEIR_nu(start_date,P=720403,E=0,I=init_inf,R=0, mu = mu,
                   rho=init_rho,tau=tau,nu=1.7,MaxDays=MaxDays,suppress_output=1,
                   rho_sched=rho_sched, mort_rate=mort_rate, symp_2_death=symp_2_death,t=t)
    elif model == 'NBD_SEIR':
        I_ind = 3; R_ind = 4; D_ind = 6
        out = NBD_SEIR(start_date,720403,0,init_inf,0,
                   init_rho,tau,k,mu,county,MaxDays,suppress_output=1,
                   rho_sched=rho_sched, t=t,
                   mort_rate=mort_rate, symp_2_death=symp_2_death)
    C = int(tau/t)
    max_ind = int((today-start_date).days/t)                 
    T = out[0]; I = out[I_ind]; R = out[R_ind]; D = out[D_ind]
    T1 = T; T2 = T[0:max_ind]; T3 = T[0:-C]
    CC1 = array(R[C:max_ind+C]); CC1p = pct_tst*CC1
    CC2 = array(R[C:]); CC2p = pct_tst*CC2
    
    [death_times, case_times, deaths, cases] = dataHandler(start_date, 
                                                            conf_case_delay)

    # def days_2_ms(T):
        # Dt = []
        # for i in T:
            # Dt.append(i*24*3600*1000)
        # return Dt  
      
    # Dt1 = days_2_ms(T1)
    # Dt2 = days_2_ms(T2)
    # Dt3 = days_2_ms(T3)
    # DTd = days_2_ms(death_times)
    # DTc = days_2_ms(case_times)
    
    # Set up data:  ---
    # source1 = ColumnDataSource(data=dict(T1=Dt1, I=I, D=D))
    # source2 = ColumnDataSource(data=dict(T2=Dt2, CC1=CC1,CC1p=CC1p))
    # source3 = ColumnDataSource(data=dict(T3=Dt3, CC2=CC2, CC2p=CC2p))
    # scatter_source1 = ColumnDataSource(data=dict(Td=DTd, d=deaths))
    # scatter_source2 = ColumnDataSource(data=dict(Tc=DTc, c=cases))
    
    source1 = ColumnDataSource(data=dict(T1=T1, I=I, D=D))
    source2 = ColumnDataSource(data=dict(T2=T2, CC1=CC1,CC1p=CC1p))
    source3 = ColumnDataSource(data=dict(T3=T3, CC2=CC2, CC2p=CC2p))
    scatter_source1 = ColumnDataSource(data=dict(Td=death_times, d=deaths))
    scatter_source2 = ColumnDataSource(data=dict(Tc=case_times, c=cases))
    
    # Set up plots:  ---
    # Top-left plot
    plot1 = figure(plot_height=400, plot_width=400, 
                   title="Predicted vs. Reported Deaths in "+county+" County",
                   tools="crosshair,pan,reset,save,wheel_zoom", # x_axis_type='datetime',
                   x_range=[0, T[max_ind]], y_range=[0, max(deaths)])         
    plot1.line('T1'[0:max_ind], 'D'[0:max_ind], source=source1, 
                line_color='red', line_width=3, line_alpha=0.6, muted_alpha=0.2,
                legend_label='Predicted Cumulative Deaths')
    plot1.circle('Td','d',source=scatter_source1, line_color='red', muted_alpha=0.2,
                legend_label='Reported Deaths in '+county+' County')
    plot1.legend.location = 'top_left'
    plot1.legend.click_policy = 'mute'
    plot1.xaxis.ticker = [0,31,61]
    plot1.xaxis.major_label_overrides = {0:'Mar', 31:'Apr', 61:'May'}
    
    # Top-right plot
    plot2 = figure(plot_height=400, plot_width=400, 
                   title="Predicted Deaths over the Long Term",
                   tools="crosshair,pan,reset,save,wheel_zoom", #x_axis_type='datetime',
                   x_range=[0, MaxDays], y_range=[0, max(D)])         
    plot2.line('T1', 'D', source=source1, 
                line_color='red', line_width=3, line_alpha=0.6, muted_alpha=0.2,
                legend_label='Predicted Cumulative Deaths')
    plot2.circle('Td','d',source=scatter_source1, line_color='red', muted_alpha=0.2,
                legend_label='Reported Deaths in'+county+' County')
    plot2.legend.location = 'top_left'
    plot2.legend.click_policy = 'mute'
    plot2.xaxis.ticker = [0,31,61,92,122,153,183,214,245,275]
    plot2.xaxis.major_label_overrides = {0:'Mar', 31:'Apr',
                                        61:'May', 92:'Jun',
                                       122:'Jul',153:'Aug',
                                       183:'Sep',214:'Oct',
                                       245:'Nov',275:'Dec'}
    
    # Bottom-left plot
    plot3 = figure(plot_height=400, plot_width=400, 
                  title="Predicted vs. Reported Infectious in El Paso County",
                  tools="crosshair,pan,reset,save,wheel_zoom", #x_axis_type='datetime',
                  x_range=[0, T[max_ind]], y_range=[0, R[max_ind]])         
    plot3.line('T1'[0:max_ind], 'I'[0:max_ind], source=source1, 
                line_color='blue', line_width=3, line_alpha=0.6, muted_alpha=0.2,
                legend_label='Predicted Infectious Population')
    plot3.line('T2', 'CC1', source=source2, 
                line_color='green', line_width=3, line_alpha=0.6,  muted_alpha=0.2,
                legend_label='Predicted Cumulative Cases')
    plot3.line('T2', 'CC1p', source=source2, 
                line_color='green', line_dash = 'dotted',line_width=3, line_alpha=0.6,
                muted_alpha=0.2,
                legend_label='Predicted Confirmed Cases based on Tested Percentage')
    plot3.circle('Tc','c',source=scatter_source2, line_color='blue', muted_alpha=0.2,
                legend_label='Reported Confirmed Cases in '+county+' County')
    plot3.legend.location = 'top_left'
    plot3.legend.click_policy = 'mute'
    plot3.xaxis.ticker = [0,31,61]
    plot3.xaxis.major_label_overrides = {0:'Mar', 31:'Apr', 61:'May'}
    
    # Bottom-right plot
    plot4 = figure(plot_height=400, plot_width=400, 
                  title="Predicted Infectious over the Long Term",
                  tools="crosshair,pan,reset,save,wheel_zoom", #x_axis_type='datetime',
                  x_range=[0, MaxDays], y_range=[0, max(R)])         
    plot4.line('T1', 'I', source=source1, 
                line_color='blue', line_width=3, line_alpha=0.6, muted_alpha=0.2,
                legend_label='Predicted Infectious Population')
    plot4.line('T3', 'CC2', source=source3, 
                line_color='green', line_width=3, line_alpha=0.6, muted_alpha=0.2,
                legend_label='Predicted Cumulative Cases')
    plot4.line('T3', 'CC2p', source=source3, 
                line_color='green', line_dash='dotted', line_width=3, line_alpha=0.6,
                muted_alpha=0.2,
                legend_label='Predicted Confirmed Cases based on Tested Percentage')
    plot4.circle('Tc','c',source=scatter_source2, line_color='blue', muted_alpha=0.2,
                legend_label='Reported Confirmed Cases in '+county+' County')
    plot4.legend.location = 'top_left'
    plot4.legend.click_policy = 'mute'
    plot4.xaxis.ticker = [0,31,61,92,122,153,183,214,245,275]
    plot4.xaxis.major_label_overrides = {0:'Mar', 31:'Apr',
                                        61:'May', 92:'Jun',
                                       122:'Jul',153:'Aug',
                                       183:'Sep',214:'Oct',
                                       245:'Nov',275:'Dec'}
                                       
    # Set up widgets
    delta = 0.1
    sliders = []
    text                = TextInput(title="title", 
                            value='my sine wave')
    s_init_rho          = Slider(title="Initial R_0", 
                            value=init_rho, start=0.1, end=6.0, step=delta/10)
    sliders.append(s_init_rho)
    s_lock_rho          = Slider(title="Stay-at-home R_0", 
                            value=lock_rho, start=0.1, end=6.0, step=delta/10)
    sliders.append(s_lock_rho)
    s_post_lock_rho     = Slider(title="R_0 after End of Stay-at-home order", 
                            value=post_lock_rho, start=0.1, end=6.0, step=delta/10)
    sliders.append(s_post_lock_rho)
    s_post_lock_time    = Slider(title="End of Stay-at-home posture (Days after 1 Mar)", 
                            value=post_lock_time, start=50, end=200, step=1)
    sliders.append(s_post_lock_time)
    s_init_inf          = Slider(title="Initial Infectious", 
                            value=init_inf, start=1, end=100, step=1)
    sliders.append(s_init_inf)
    s_init_exp          = Slider(title="Initial Exposed", 
                            value=init_exp, start=0, end=100, step=1)
    sliders.append(s_init_exp)
    s_init_rec          = Slider(title="Initial Recovered", 
                            value=init_rec, start=0, end=100, step=1)
    sliders.append(s_init_rec)
    s_pct_tst           = Slider(title="Percent of Infected Population Tested & Positive", 
                            value=pct_tst*100, start=1, end=100, step=delta)
    sliders.append(s_pct_tst)
    s_tau               = Slider(title="Infectious Period (days)", 
                            value=tau, start=2, end=14, step=delta)
    sliders.append(s_tau)
    s_mu                = Slider(title="Incubation Period (days)", 
                            value=mu, start=0, end=7, step=delta)
    sliders.append(s_mu)
    if model == 'SIR_nu' or model == 'SEIR_nu':
        s_het               = Slider(title="Power-Law Heterogeneity",
                                    value=nu, start=1.0, end=3.3, step=delta)
    elif model == 'NBD_SEIR':
        s_het               = Slider(title="NBD Homogeneity",
                                   value=k, start=0.001, end=5, step=0.01)
    sliders.append(s_het)
    s_mort_rate         = Slider(title="Mortality Rate (%)", 
                            value=mort_rate*100, start=0.1, end=10, step=delta)
    sliders.append(s_mort_rate)
    s_symp_2_death      = Slider(title="Avg Delay from Symptoms until Death (days)", 
                            value=symp_2_death, start=3, end=20, step=delta)
    sliders.append(s_symp_2_death)
    s_conf_case_delay   = Slider(title="Avg Delay from Test Processing (days)", 
                            value=conf_case_delay, start=0, end=14, step=delta)
    sliders.append(s_conf_case_delay)
    s_t                 = Slider(title="Timestep (minutes)", 
                            value=t*(60*24), start=15, end=60*24, step=5)
    sliders.append(s_t)

    # Set up callbacks
    def update_title(attrname, old, new):
        plot1.title.text = text.value

    #text.on_change('value', update_title)

    def update_data(attrname, old, new):        
        # Get the current slider values
        init_rho        = s_init_rho.value
        lock_rho        = s_lock_rho.value
        post_lock_rho   = s_post_lock_rho.value
        post_lock_time  = s_post_lock_time.value
        post_lock_date  = start_date + timedelta(days=post_lock_time)
        init_inf        = s_init_inf.value
        init_exp        = s_init_exp.value
        init_red        = s_init_rec.value
        pct_tst         = s_pct_tst.value/100
        tau             = s_tau.value
        mu              = s_mu.value
        if model_sel.active in [0,1]: # Power law models (nu)
            nu              = s_het.value
        elif model_sel.active == 2:   # NBD model (k)
            k               = s_het.value
        mort_rate       = s_mort_rate.value/100
        symp_2_death    = s_symp_2_death.value
        conf_case_delay = s_conf_case_delay.value
        t               = s_t.value/(60*24)
        
        max_ind = int((today-start_date).days/t)
        [death_times, case_times, deaths, cases] = dataHandler(start_date, 
                                                    conf_case_delay)
        rho_sched = {lock_date:lock_rho,
                     post_lock_date:post_lock_rho}
        if model_sel.active == 0: # SIR_nu
            I_ind = 2; R_ind = 3; D_ind = 4
            out2 = SIR_nu(start_date,P=720403,I=init_inf,R=init_rec,
                   rho=init_rho,tau=tau,nu=nu,MaxDays=MaxDays,suppress_output=1,
                   rho_sched=rho_sched, mort_rate=mort_rate, symp_2_death=symp_2_death,t=t)
        elif model_sel.active == 1: # SEIR_nu 
            I_ind = 3; R_ind = 4; D_ind = 5
            out2 = SEIR_nu(start_date,P=720403,E=init_exp,I=init_inf,R=init_rec, mu = mu,
                       rho=init_rho,tau=tau,nu=nu,MaxDays=MaxDays,suppress_output=1,
                       rho_sched=rho_sched, mort_rate=mort_rate, symp_2_death=symp_2_death,t=t)
        elif model_sel.active == 2: # NBD_SEIR
            I_ind = 3; R_ind = 4; D_ind = 6
            out2 = NBD_SEIR(start_date,720403,init_exp,init_inf,init_rec,
                        init_rho,tau,k,mu,county,MaxDays,suppress_output=1,
                        rho_sched=rho_sched, 
                        mort_rate=mort_rate, symp_2_death=symp_2_death, t=t)
        T = out2[0]; I = out2[I_ind]; R = out2[R_ind]; D = out2[D_ind]
        T1 = T; T2 = T[0:max_ind]; T3 = T[0:-C]
        CC1 = array(R[C:max_ind+C]); CC1p = pct_tst*CC1
        CC2 = array(R[C:]); CC2p = pct_tst*CC2
        
        [death_times, case_times, deaths, cases] = dataHandler(start_date, 
                                                            conf_case_delay)
        plot1.y_range.start = 0 
        plot1.y_range.end   = max(deaths)
        plot2.y_range.start = 0 
        plot2.y_range.end   = max(D)
        if zm_toggle == -1:
            plot3.y_range.start = 0
            plot3.y_range.end   = R[max_ind]
        plot4.y_range.start = 0
        plot4.y_range.end   = max(R)
        
        # Generate the new curve
        source1.data=dict(T1=T1, I=I, D=D)
        source2.data=dict(T2=T2, CC1=CC1,CC1p=CC1p)
        source3.data=dict(T3=T3, CC2=CC2, CC2p=CC2p)
        scatter_source1.data=dict(Td=death_times, d=deaths)
        scatter_source2.data=dict(Tc=case_times, c=cases)
    
    def update_radio_buttons(attrname, old, new):
        if model_sel.active in [0,1]:
            s_het.title="Power-Law Heterogeneity"
            s_het.value=nu
            s_het.start=1.0; s_het.end=3.3; s_het.step=delta
        elif model_sel.active == 2:
            s_het.title="NBD Heterogeneity"
            s_het.value=k
            s_het.start=0.0001; s_het.end=5; s_het.step=delta/10
        update_data(attrname,old,new)    
    
    def zm_button_event(event):
        global zm_toggle
        zm_toggle = -zm_toggle
        if zm_toggle == 1:
            plot3.y_range.start = 0
            plot3.y_range.end   = max(cases)
        elif zm_toggle == -1:
            plot3.y_range.start = 0
            plot3.y_range.end   = R[max_ind]            
    
    def legend_button_event(event):
        global legend_toggle
        legend_toggle = -legend_toggle
        if legend_toggle == 1:
            plot1.legend.visible = True
            plot2.legend.visible = True
            plot3.legend.visible = True
            plot4.legend.visible = True
        elif legend_toggle == -1:
            plot1.legend.visible = False
            plot2.legend.visible = False
            plot3.legend.visible = False
            plot4.legend.visible = False
    
    for s in sliders:
        s.on_change('value', update_data)
    
    model_sel = RadioButtonGroup(labels=['SIR-nu','SEIR-nu','NBD_SEIR'],
                                 active=2)
    model_sel.on_change('active', update_radio_buttons)
    
    zm_button = Button(label='Zoom to Confirmed Cases', button_type='success')
    zm_button.on_event(ButtonClick, zm_button_event)
    legend_button = Button(label='Hide Plot Legends', button_type='success')
    legend_button.on_event(ButtonClick, legend_button_event)
    # Set up layouts and add to document
    inputs = column(model_sel, s_init_rho, s_lock_rho, s_post_lock_rho, s_post_lock_time,
                    s_init_inf, s_init_exp, s_init_rec, s_pct_tst, s_tau, s_mu, s_het, #s_nu, s_k,
                    s_mort_rate, s_symp_2_death, s_conf_case_delay, s_t)
    grid = gridplot([[plot1, plot2],[plot3, plot4]])
    middle = column(grid, zm_button)
    curdoc().add_root(row(inputs, middle, legend_button, width=800))
    curdoc().title = "El Paso County COVID-19 App"

slider_app_bokeh()
