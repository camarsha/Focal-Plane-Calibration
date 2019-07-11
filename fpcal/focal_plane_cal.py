import re
import numpy as np
from scipy import optimize
from scipy import stats
import pandas as pd
from . import residual_plot
import matplotlib.pyplot as plt
import string
import emcee
import corner
from scipy import interpolate

"""
Program is intended for use with TUNL Enge Splitpole.
Using NNDC 2016 mass evaluation for masses as Q-Values.

Caleb Marshall, TUNL/NCSU, 2017
"""

# convert to MeV/c^2
u_convert = 931.4940954

# read in the mass table with the format provided
pretty_mass = '/home/caleb/Research/Longland/Code/AnalysisScripts/EnergyCalibration/pretty.mas16' # pretty mass path
mass_table = pd.read_csv(pretty_mass, sep='\s+')  # Read in the masses


# once again the chi_square objective function
def chi_square(poly, rho, channel, unc):
    poly = np.poly1d(poly)
    theory = poly(channel)
    temp = ((theory-rho)/unc)**2.0
    return np.sum(temp)


# for all values with uncertainty
def measured_value(x, dx):
    return {'value': x,
            'unc': dx}

# relativistic calculation of the magnetic rigidity
def relativist_rho(a, A, b, B,
                    B_field, E_lab, E_level, q, theta):
    
    theta = theta*(np.pi/180.0) #to radians 
    s = (A+a+E_lab)**2.0-(2*a*(E_lab)+(E_lab)**2.0) #relativistic invariant
    pcm = np.sqrt(((s-a**2.0-A**2.0)**2.0-(4.*a**2.0*A**2.0))/(4.*s)) # com p
    chi = np.log((pcm+np.sqrt(A**2.0+pcm**2.0))/A) # rapidity
    p_prime = np.sqrt(((s-b**2.0-(B+E_level)**2.0)**2.0-(4.*b**2.0*(B+E_level)**2.0))/(4.*s)) # com p of products
    # now solve for momentum of b, terms are just for ease  
    term1 = np.sqrt(b**2.0+p_prime**2.0)*np.sinh(chi)*np.cos(theta) 
    term2 = np.cosh(chi)*np.sqrt(p_prime**2.0-(b**2.0*np.sin(theta)**2.0*np.sinh(chi)**2.0))
    term3 = 1.0+np.sin(theta)**2.0*np.sinh(chi)**2.0
    p = ((term1+term2)/term3)
    rho = (.3335641*p)/(q*B_field) # from enge's paper     
    return rho


def reverse_rho(a, A, b, B,
                B_field, E_lab, rho, q, theta):
    
    theta = theta*(np.pi/180.0) #to radians
    pb = (B_field*rho*q)/.3335641 #momentum of b
    pa = np.sqrt(2.0*a*E_lab+E_lab**2.0)
    pB = np.sqrt(pa**2.0+pb**2.0-(2.0*pa*pb*np.cos(theta))) # conservation gives you this
    #now do conservation of Energy
    E_tot = E_lab+a+A # what we start out with
    Eb = np.sqrt(pb**2.0+b**2.0)
    EB = np.sqrt(pB**2.0+B**2.0)
    Ex = E_tot - Eb - EB
    return Ex


# gather all the nuclei data into a class
class Nuclei():

    def __init__(self, name):
        # parse the input string
        self.name = name
        self.A = int(re.split('\D+', name)[0])
        self.El = re.split('\d+', name)[1]
        self.get_mass_charge()
       
    # searches the mass table for the given isotope
    def get_mass_charge(self, table=mass_table):
        m = table[(table.El == self.El) &
                  (table.A == self.A)]['Mass'].values*u_convert
        dm = table[(table.El == self.El)
                   & (table.A == self.A)]['Mass_Unc'].values*u_convert
        self.m = measured_value(m, dm)
        self.Z = table[(table.El == self.El) &
                       (table.A == self.A)]['Z'].values

    # just if you want to check data quickly
    def __call__(self):
        print('Nuclei is '+str(self.A)+str(self.El)+'\n')
        print('Mass =', self.m['value'], '+/-', self.m['unc'])


# class that handles defines the reaction parameters
class Reaction():

    def __init__(self, a, A, b, B, B_field,
                 E_lab, theta, mass_unc=False, E_lab_unc=0.0):
        """
        Parse reaction names,looks up there masses, and calculates Q-value(MeV)
        E_lab = MeV
        B_field = Tesla
        """
        self.a = Nuclei(a)
        self.A = Nuclei(A)
        self.b = Nuclei(b)
        self.B = Nuclei(B)
        __Q = ((self.a.m['value'] + self.A.m['value']) -
               (self.b.m['value'] + self.B.m['value']))
        # compute Q-value unc using quadrature
        __dQ = np.sqrt(self.a.m['unc']**2+self.A.m['unc']**2 +
                       self.b.m['unc']**2 + self.B.m['unc']**2)
        self.Q = measured_value(__Q, __dQ)
        self.B_field = B_field  # mag field
        self.q = self.b.Z  # charge of projectile
        if E_lab_unc:
            self.E_lab = measured_value(E_lab, E_lab_unc)
        else:
            self.E_lab = E_lab
        self.theta = theta
        if not mass_unc:
            self.a.m['unc'] = 0.0
            self.A.m['unc'] = 0.0
            self.b.m['unc'] = 0.0
            self.B.m['unc'] = 0.0

    def name(self):
        print(self.a.name+' + '+self.A.name+' -> '+self.B.name+' + '+self.b.name)


class Focal_Plane_Fit():

    def __init__(self):

        self.reactions = {}
        # Points is a list of dictionaries with rho,channel entry structure.
        # Each of those has a value/uncertainty component.
        self.points = []  # calibration points
        self.fits = {}  # fit parameters of a normal fit
        self.fits_bay = {}  # fit parameters of a Bayesian fit
        self.output_peaks = []  # calibrated energy values
        self.poly_model = {}  # dict of mcmc objects for fits
        self.sys_unc = 0.0  # systematic uncertainty for peak centroid widths

    def add_reaction(self):
        # take user input for reaction
        a = str(input('Enter projectile \n'))
        A = str(input('Enter target \n'))
        b = str(input('Enter detected particle \n'))
        B = str(input('Enter residual particle \n'))
        B_field = float(input('What was the B field setting? \n'))
        E_lab = float(input('Beam energy? \n'))
        E_lab_unc = float(input('Beam energy uncertainty? \n'))
        theta = float(input('What is the lab angle? \n'))
        self.reactions[len(list(self.reactions.keys()))] = Reaction(a, A, b, B, B_field,
                                                              E_lab, E_lab_unc, theta)
        print('Reaction', (len(list(self.reactions.keys()))-1),'has been defined as '+a+' + '+A+' -> '+B+' + '+b)
        print('E_beam =', E_lab,'+/- MeV', E_lab_unc, 'With B-Field', B_field,'T') 

    def add_point(self, reaction, level,
                  level_unc, channel, channel_unc):
        # get rho and uncertainty
        rho, rho_unc = self.calc_rho(reaction,
                                     level, level_unc)
        rho = measured_value(rho, rho_unc)  # convert to dict
        channel = measured_value(channel, channel_unc)
        point = {'rho': rho, 'channel': channel}
        self.points.append(point)

    # add a calibration point which has rho with an associated channel value
    def input_point(self):
        reaction = int(input('Which reaction(0...n)? \n'))
        channel = float(input('Enter the peak channel number. \n'))
        channel_unc = float(input('What is the centroid uncertainty? \n'))
        level = float(input('Enter the peak energy (MeV). \n'))
        level_unc = float(input('Enter the peak uncertainty (MeV). \n'))
        self.add_point(reaction, level, level_unc, channel, channel_unc)

    def create_distributions(self, reaction):
        reaction_variables = vars(reaction)  # get variables from reaction
        normals = {}  # dictionary for all quantities in our model
        # define distributions, all normal for now
        # this loop is pulling variables directly
        # from the reaction.__dict__ method
        for var in reaction_variables:
            # test to see if value has uncertainty
            if type(reaction_variables[var]) == dict:
                temp_mu = reaction_variables[var]['value']
                temp_sigma = reaction_variables[var]['unc']
                # tau = 1/sigma^2 var is name of variable
                temp_normal = pm.Normal(var, temp_mu, (1.0/temp_sigma)**2.0)
                normals[var] = temp_normal
            # special cases for masses
            elif isinstance(reaction_variables[var], Nuclei):
                temp = reaction_variables[var].m
                temp_mu = temp['value']
                temp_sigma = temp['unc']
                if temp_sigma == 0.0:
                    normals[var] = temp_mu
                else:    
                    temp_normal = pm.Normal(var,temp_mu,(1.0/temp_sigma)**2.0) 
                    normals[var] = temp_normal
            else:
                normals[var] = reaction_variables[var] #these are just constant parameters
                
        return normals
    
    #added Monte Carlo error propagation for rho     
    def calc_rho(self,reaction,E_level,E_level_unc):
        reaction = self.reactions[reaction] #just for short hand pick out reaction

        E_level = stats.norm.rvs(loc=E_level, scale=E_level_unc, size=100000)

        # Get all the variable values
        a = reaction.a.m['value']
        b = reaction.b.m['value']
        A = reaction.A.m['value']
        B = reaction.B.m['value']
        B_field = reaction.B_field
        E_lab = reaction.E_lab
        theta = reaction.theta
        q = reaction.q

        rho = relativist_rho(a, A, b, B,
                              B_field, E_lab, E_level, q, theta)
        
        print("Mean Value:", rho.mean())
        print("Standard Deviation :", rho.std())
        print()
        return rho.mean(), rho.std()

    #chi square fit will be good for quick cal 
    def fit(self,order=2,plot=True):
        N = len(self.points) # Number of data points
        if N > (order+1): #check to see if we have n+2 points where n is the fit order
            print("Using a fit of order",order)
            x_rho = np.zeros(N) #just decided to start with arrays. maybe dumb
            x_channel = np.zeros(N)
            x_unc = np.zeros(N) #rho unc
            x_channel_unc = np.zeros(N) #channel unc
            coeff = np.ones(order+1) #these are the n+1 fit coefficients for the polynomial fit
            for i,point in enumerate(self.points):
                #collect the different values needed for chi_square fit
                x_rho[i] = point['rho']['value']
                x_channel[i] = point['channel']['value']
                x_unc[i] = point['rho']['unc']
                x_channel_unc = point['channel']['unc']
                
            #now scale channel points
            channel_mu = np.sum(x_channel)/float(N) #scale will be average of all calibration peaks
            x_channel_scaled = x_channel-channel_mu
            #create bounds
            abound = lambda x:(-100.0,100.0) #creates a tuple
            bounds = [abound(x) for x in range(order+1)] #list of tuples
            #differential_evolution method, much faster than basin hopping with nelder-mead and seems to get same answers
            sol = optimize.differential_evolution(chi_square,bounds,maxiter=100000,args=(x_rho,x_channel_scaled,x_unc))
            #tell myself what I did
            chi2 = sol.fun/(N-(order+1))
            print("Chi Square is", chi2)
            print("Fit parameters are (from highest order term to lowest)",sol.x)
            #now adjust uncertainty to do fit again
            x_unc[:] = self.adjust_unc(np.poly1d(sol.x),x_channel_scaled,x_rho,x_channel_unc,x_unc)
            print("Now doing adjusted unc fit")
            sol = optimize.differential_evolution(chi_square,bounds,maxiter=100000,args=(x_rho,x_channel_scaled,x_unc))
            chi2 = sol.fun/(N-(order+1))
            print("Chi Square is", chi2)
            print("Adjusted fit parameters are (from highest order term to lowest)",sol.x)
            self.fits[order] = np.poly1d(sol.x) #add to dictionary the polynomial object
            print("Fit stored in member variable fits[%d]" %order)
            #create the a plot showing the fit and its residuals
            if plot:
                residual_plot(x_channel_scaled,x_rho,x_unc,self.fits[order])
                plt.show()            
        else:
            print("Not enough points to preform fit of order",order,"or higher")

    #this is based on description in spanc on how they estimate uncertainty
    @staticmethod
    def adjust_unc(poly,x,y,x_unc,y_unc):
        dpoly = np.polyder(poly) #compute derivative
        dpoly = dpoly(x) #evaluate
        new_unc = np.sqrt((dpoly*x_unc)**2.0+y_unc**2.0) #add in quadrature
        return new_unc
    
    def input_peak(self):
        channel = float(input("Enter channel number."))
        channel_unc = float(input("Enter channel uncertainty."))
        channel = measured_value(channel,channel_unc)
        self.peak_energy(channel)
        
    #finally given channel number use a fit to give energy         
    def peak_energy(self, reaction, channel, fit_order=2, kde_samples=50000):
        if type(channel) == dict:
            reaction_number = reaction
            reaction = self.reactions[reaction] #just for short hand pick out reaction

            # Get all the variable values
            a = reaction.a.m['value']
            b = reaction.b.m['value']
            A = reaction.A.m['value']
            B = reaction.B.m['value']
            B_field = reaction.B_field
            E_lab = reaction.E_lab
            theta = reaction.theta
            q = reaction.q

            #calc rho from normal distributions in polynomial fit
            x_mu = self.fits_bay['Order_'+str(fit_order)][-1]
            coeff = []
            letters = string.ascii_uppercase[0:(fit_order+1)]
            # get the coeff trace arrays
            for ele in letters:
                coeff.append(self.poly_model[fit_order].trace(ele)[:])
            
            # We are going to model the coefficients based on a Gaussian kde
            # this avoids the problems with the auto-correlation in our samples
            # and means we can get more out of shorter runs
            coeff_kde = [stats.gaussian_kde(i) for i in coeff]
            coeff_kde_samples = [i.resample(kde_samples)[0] for i in coeff_kde]
            coeff_kde_samples = np.asarray(coeff_kde_samples).T

            # Now draw the random x samples
            x_samples = stats.norm.rvs(loc=channel['value'],
                                       scale=np.sqrt(channel['unc']**2.0+self.sys_unc**2.0),
                                       size=kde_samples)
            x = x_samples - (x_mu*np.ones(kde_samples))  # scale them
            
            # now calculate the fit rho values from trace and random samples
            rho = []
            for i,j in zip(coeff_kde_samples, x):
                value = np.poly1d(i)(j)
                rho.append(value)
                
            # Calculate and record the stats
            rho = np.asarray(rho)  # Now we have 1d array of rho values 
            mu = rho.mean()
            sig = rho.std()
            # Reverse the kinematics
            Ex = reverse_rho(a, A, b, B, B_field, E_lab, rho, q, theta)
            E = Ex.mean()
            E_sig = Ex.std()
            print('Energy :', Ex.mean(), '+/-', Ex.std())
        
            output = {'Reaction':reaction_number,'Rho':measured_value(mu,sig),
                      'E_level':measured_value(E,E_sig),'Channel':channel,
                       'Rho_trace': rho} #gather values into dictionary
            self.output_peaks.append(output) #append to list 
        else:
            print("Need a dictionary of value and uncertainty!!")

    # function to read in a file with calibration points and preform a fit
    def read_calibration(self, cal_file, reaction=None, scale_factor=1.0):
        # can just pick, or ask for user input for reaction
        if type(reaction) != int:
            reaction = int(input('Which reaction(0...n)? \n'))
        data = pd.read_csv(cal_file, sep='\s+')
        for i in data.index:
            level = data["level"][i]
            level_unc = data["level_unc"][i]
            channel = (data["channel"][i])/scale_factor
            channel_unc = (data["channel_unc"][i])/scale_factor
            self.add_point(reaction, level, level_unc,
                           channel, channel_unc)
            
    # reads file with channel and channel_unc and gives fitted values
    def read_peaks(self, peak_file, reaction=None,
                   fit_order=None, scale_factor=1.0, kde_samples=50000):
        if type(reaction) != int:
            reaction = int(input('Which reaction(0...n)? \n'))
        if type(fit_order) != int:
            reaction = int(input('Which fit(0...n)? \n'))
        data = pd.read_csv(peak_file, sep='\s+')
        for i in data.index:
            channel = (data["channel"][i])/scale_factor
            channel_unc = (data["channel_unc"][i])/scale_factor
            self.peak_energy(reaction, measured_value(channel, channel_unc),
                             fit_order=fit_order, kde_samples=kde_samples)

    # predicted output energies
    def write_energies(self, filename='predicted_energies.dat'):
        t = '   '
        with open(filename, 'w') as f:
            f.write('reaction' + t + 'channel' + t +
                    'channel_unc' + t + 'E' + t + 'E_unc' + '\n')
            for ele in self.output_peaks:
                f.write(str(ele['Reaction']) +
                        t + str(ele['Channel']['value']) +
                        t + str(ele['Channel']['unc']) +
                        t + str(ele['E_level']['value']) +
                        t + str(ele['E_level']['unc']) + '\n')


class Ensemble_Fit(Focal_Plane_Fit):

    def __init__(self):
        Focal_Plane_Fit.__init__(self)

    def lnprior(self, x, x0):
        return stats.norm.logpdf(x, loc=x0, scale=100.0)

    def ln_sys_unc(self, x):
        return stats.halfcauchy.logpdf(x)
    
    def lnprob(self, x, x0, mu_0, y, y_unc, x_unc):
        # Will need to separate coefficients, x data, and systematic term.
        sys_prior = self.ln_sys_unc(x[-1])
        if np.isinf(sys_prior):
            return -1.*np.inf
        
        coeff_prior = np.sum(self.lnprior(x[:len(x0)], x0))
        
        # x likelihood first
        x_like = np.sum(stats.norm.logpdf(x[len(x0):-1], loc=mu_0,
                                          scale=np.sqrt(x_unc**2.0 +
                                                        (1./np.sqrt(x[-1]))**2.0)))
        f = np.poly1d(x[0:len(x0)])(x[len(x0):-1])
        y_like = np.sum(stats.norm.logpdf(f, loc=y, scale=y_unc))
        return sys_prior + coeff_prior + x_like + y_like


    def bay_fit(self, order=2, trace_plot=False, plot=True, corner_plot=False,
                iterations=5000, nwalkers=50, burn=2500, thin=10):

        # Ensemble sampler needs to an initial fit
        self.fit(order=order, plot=False)
        coeff_x0 = self.fits[order].c
        # These coeff define the prior ranges.
        #get data
        x_obs = np.asarray([ele['channel']['value'] for ele in self.points])
        self.x_mu = np.sum(x_obs)/float(len(x_obs))
        x_scaled = x_obs - self.x_mu
        x_unc = np.asarray([ele['channel']['unc'] for ele in self.points])
        y_obs = np.asarray([ele['rho']['value'] for ele in self.points])
        y_unc = np.asarray([ele['rho']['unc'] for ele in self.points])

        # Initialize the walkers in a ball around the starting values
        init_array = np.concatenate((coeff_x0, x_scaled, np.array([1.0])))
        ndim = len(init_array)
        p0 = init_array + (1e-4*init_array)*np.random.randn(nwalkers, ndim) 

        #Now lets sample with standard stretch move.
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,
                                             args=(coeff_x0, x_scaled, y_obs, y_unc, x_unc))
        self.sampler.run_mcmc(p0, iterations, progress=True)

        if trace_plot:
            for i in range(ndim):
                temp = plt.plot(self.sampler.get_chain()[:, :, i])
            plt.show()
            
        # Store samples based on burn/thin. You can always go back and select
        # different values
        self.samples = self.sampler.get_chain(flat=True, discard=burn, thin=thin)
        # Print the lnprob mean to give estimate of goodness of fit.
        mean_lnprob = self.sampler.get_log_prob(flat=True, discard=burn, thin=thin).mean()
        print('Average lnprob =', mean_lnprob)
        self.sys_unc = 1.0/np.sqrt(self.samples[:, -1].mean())
        if corner_plot:
            letters = string.ascii_uppercase[0:len(coeff_x0)]
            # Just plot coefficients
            corner.corner(self.samples[:, 0:len(coeff_x0)], labels = letters)
            plt.show()
            #Generate

        if plot:
            fit = np.poly1d([ele.mean() for ele in self.samples[:, :order+1].T])
            residual_plot(x_scaled, y_obs, y_unc, fit, xerr=x_unc)
            plt.show()
            
            
    #  Going to try and do everything using samples from MCMC         
    def peak_energy(self, reaction, channel, order):
        if type(channel) == dict:
            reaction_number = reaction
            reaction = self.reactions[reaction] #just for short hand pick out reaction

            # Get all the variable values
            a = reaction.a.m['value']
            b = reaction.b.m['value']
            A = reaction.A.m['value']
            B = reaction.B.m['value']
            B_field = reaction.B_field
            E_lab = reaction.E_lab
            theta = reaction.theta
            q = reaction.q

            # Raw samples motherfucker!
            coeff_samples = self.samples[:, 0:(order+1)]
            
            # Now draw the random x samples
            x_samples = stats.norm.rvs(loc=channel['value'],
                                       scale=np.sqrt(channel['unc']**2.0+self.sys_unc**2.0),
                                       size=self.samples.shape[0])
        
            x = x_samples - self.x_mu  # scale them
            
            # now calculate the fit rho values from trace and random samples
            rho = []
            for i, j in zip(coeff_samples, x):
                value = np.poly1d(i)(j)
                rho.append(value)

            # Calculate and record the stats
            rho = np.asarray(rho)  # Now we have 1d array of rho values 
            mu = rho.mean()
            sig = rho.std()
            # Reverse the kinematics
            Ex = reverse_rho(a, A, b, B, B_field, E_lab, rho, q, theta)
            E_ci = self.credability_interval(Ex)
            # Moving to confidence intervals.
            print('Channel', format(channel['value'], '.1f'), 'Energy :', format(E_ci[0], '.4f'), '+/-', format(E_ci[1], '.4f')+'/'+format(E_ci[2], '.4f'), 'E_mu ='+format(Ex.mean(), '.4f'), 'E_sigma ='+format(Ex.std(), '.4f'))  
            # Gather info into dictionary
            output = {'Reaction': reaction_number,'Rho': measured_value(mu,sig),
                      'E_level': E_ci,'Channel':channel,
                       'Rho_trace': rho, 'E_trace': Ex, 'E_mu': Ex.mean(),
                      'E_std': Ex.std()} 
            self.output_peaks.append(output) # append to list 
        else:
            print("Need a dictionary of value and uncertainty!!")

    def read_peaks(self, peak_file, reaction,
               fit_order, scale_factor=1.0):
        data = pd.read_csv(peak_file, sep='\s+')
        for i in data.index:
            channel = (data["channel"][i])/scale_factor
            channel_unc = (data["channel_unc"][i])/scale_factor
            self.peak_energy(reaction, measured_value(channel, channel_unc),
                             fit_order)


    def plot_fit(self, order, x_min=0.0, x_max=4000.0, n_samples=100):
        """
        Draw from the MCMC samples and plot the
        regression.
        """

        # Select random samples to plot.  
        s = self.samples[np.random.randint(self.samples.shape[0], size=n_samples)]
        s = s[:, :(order + 1)]
        x = np.linspace(x_min, x_max, 10000)
        for i in s:
            plt.plot(x-self.x_mu, np.poly1d(i)(x-self.x_mu), alpha=.1, color='k')

        # Plots for the data points.
        
        x_obs = np.asarray([ele['channel']['value'] for ele in self.points])
        x_scaled = x_obs - self.x_mu
        x_unc = np.asarray([ele['channel']['unc'] for ele in self.points])
        y_obs = np.asarray([ele['rho']['value'] for ele in self.points])
        y_unc = np.asarray([ele['rho']['unc'] for ele in self.points])

        plt.errorbar(x_scaled, y_obs, yerr=y_unc, xerr=x_unc, fmt='ob', 
                     markersize=5.0)
        plt.show()


    def credability_interval(self, samples):
        v = np.percentile(samples, [16, 50, 84], axis=0)
        values = np.array([v[1], v[2]-v[1], v[1]-v[0]])
        return values

    
    def confidence_bands(self, x, samples, percentiles=[16, 50, 84]):
 
        all_lines = np.zeros([len(x), samples.shape[0]])
        for i, ele in enumerate(samples):
            for j in range(len(x)):
                all_lines[j][i] = np.poly1d(ele)(x[j])
        return np.percentile(all_lines, percentiles, axis=1)


    def plot_ci(self, order, n_points=100,
                x_min=0.0, x_max=4000.0):
        """
        Set up a plot with 68%, 95%, and 99% credibility bands.
        """

        samples = self.samples[:, :(order + 1)]
        x = np.linspace(x_min, x_max, n_points)
        x = x - self.x_mu
        
        # Purple son
        fill_colors = reversed(['#e0ecf4', '#9ebcda', '#8856a7'])
        # Setup the three levels of credibility.
        levels = [68.0, 95.0, 99.0]
        zorder = 0  # Controls which object is on top.
        for color, i in zip(fill_colors, levels):
            # Get the intervals
            low = 50. - (i/2.0)
            high = 50. + (i/2.0)
            ci = self.confidence_bands(x, samples,
                                       percentiles=[low, 50.0, high])
            # Spline and plot
            spline_x = np.linspace(x.min(), x.max(), 100000)

            lower_spline = interpolate.splrep(x, ci[0, :])
            lower = interpolate.splev(spline_x, lower_spline)

            upper_spline = interpolate.splrep(x, ci[-1, :])
            upper = interpolate.splev(spline_x, upper_spline)

            plt.fill_between(spline_x,
                             lower, upper,
                             color=color, zorder=zorder, alpha=1.0)
            zorder = zorder-1
        # Plot the data
        x_obs = np.asarray([ele['channel']['value'] for ele in self.points])
        x_scaled = x_obs - self.x_mu
        x_unc = np.asarray([ele['channel']['unc'] for ele in self.points])
        y_obs = np.asarray([ele['rho']['value'] for ele in self.points])
        y_unc = np.asarray([ele['rho']['unc'] for ele in self.points])

        plt.errorbar(x_scaled, y_obs, yerr=y_unc, xerr=x_unc, fmt='ob', 
                     markersize=5.0, zorder=1, color='#e66101')

        plt.xlim(x.min()-.5, x.max()+.5)
        plt.xlabel('Channel', fontsize=32)
        plt.ylabel(r'$\rho$ (cm)', fontsize=32)
        plt.tight_layout()

    def write_csv(self, sep=',', filename='Output_stats.csv'):
        """
        Output a csv file for all of the output states.
        """
        df = pd.DataFrame.from_dict(self.output_peaks)

        # This will save channel and all relevant energy statics. 
        df = df[['Channel', 'E_level', 'E_mu', 'E_std']] 
        # Channel and E_level are a dictionary and a list, so clean up
        df['Channel'] = [x['value'] for x in df['Channel']]
        df['50_percentile'] = [x[0] for x in df['E_level']]
        df['upper_ci'] = [x[1] for x in df['E_level']]
        df['lower_ci'] = [x[2] for x in df['E_level']]
        df_new = df[['Channel', 'E_mu', 'E_std',
                     '50_percentile', 'upper_ci', 'lower_ci']]
        df_new.to_csv(filename, sep=sep, index=False)

        
