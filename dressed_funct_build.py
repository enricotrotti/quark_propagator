##########################################################################################
#####################                    Authors                     #####################
################     Chevalier Cyrille, Cimino Lorenzo, Trotti Enrico     ################
##########################################################################################

##########################################################################################
#####################           Solution of the quark DSE            #####################
##########################################################################################


import math
import numpy
import matplotlib.pyplot as plt
from module_spline import *
from module_integration import *
from datetime import date
from matplotlib.animation import FuncAnimation
import cmath

#----------------------------- Solution object -------------------------------#


class Solution:
    """ This class contains an approximated solution for the Dyson-Schwinger
    equation. This approximated solution can the be improved with an iterative
    process.
    """
    def __init__(self,N,pMin=1e-3,pMax=1e3,m=4e-3,mu=19,nLeg=1000,nCheby=50):
        """
        Input: (units are GeV)
            N : a positive integer.
                It is the number of points where the sigma functions will be
                computed.
            pMin : a positive float.
                The minimal value of p for which the sigma functions will be 
                computed.
            pMax : a positive float smaller than pMin.
                The last value of p for which the sigma functions will be 
                computed.
            m : a float.
                It is the bare quark mass.
            mu : a positive float.
                It is the renormalisation point.
            nLeg : an integer.
                It is the number of points used for the Gauss-Legendre 
                integration. 
                Warning: the corresponding txt file must be in the same folder
                (mandatory name: roots_weights_[insert nLeg].txt).
            nCheby : an integer.
                It is the number of points used for the Gauss-Chebyshev 
                integration.                
        Output:
            It creates an object of class Solution wich has 21 attributes that
            I will sort in five different categories. A letter (C) at the end
            of the description of the attribute means that this attribute will 
            change during the convergence.
                The inputed physical parameters :
                    bareMass is the bare quark mass,
                        this must be chosen around 4e-3 GeV;
                    mu is the renormalisation point,
                        this must be chosen around 19 GeV.
                The attributes related to the function G(k^2) and F(q,z,p):
                    Z2 is the renormalisation constant, (C)
                        its initial value should be around 1;
                    lambda_ is a parameter in the function alpha,
                        this must be chosen around 0.72 GeV;
                    gamma is a parameter in the function alpha,
                        this must be chosen around 12/25;
                    lambdaQCD is a parameter in the function alpha,
                        this must be chosen around 0.234 GeV;
                    eta is a parameter in the function alpha,
                        this must be chosen around 1.8;
                    alpha is a given function (cfr. Tandy-Maris model),
                        this function uses the parameters eta, lambda_, gamma,
                        lambdaQCD and eta.
                    F is a function of q,z and p,
                        this function is defined in equation (21) (see project
                        description);
                    G is a function of k^2 that merges alpha and Z2, (C)
                        this function is defined on equation (16) (see project
                        description).
                The Grid attributes:
                    gridA is an object of class grid, (C)
                        this grid represents the first dressing function for 
                        the quark propagator;
                        gridA.xSet is defined by the inputed numbers N, pMin 
                        and pMax;
                        the initial gridA.fSet is identically chosen as 1;
                    gridM  is an object of class grid, (C)
                        this grid represents the second dressing function for 
                        the quark propagator;
                        gridM.xSet is defined by the inputed numbers N, pMin 
                        and pMax;
                        the initial gridM.fSet is identically chosen as the 
                        bare quark mass (or as 1 if the bare mass is zero);
                    gridSigmaA is an object of class Grid, (C)
                        this grid represents the \Sigma_A function in the
                        project description (first component of the self-energy
                        function);
                        gridSigmaA.xSet is defined by the inputed numbers N, 
                        pMin and pMax;
                        for now gridSigmaA.fSet is empty;
                    gridSigmaM is another object of class Grid, (C)
                        this grid represents the \Sigma_M function in the
                        project description (second component of the 
                        self-energy function);
                        gridSigmaM.xSet is defined by the inputed numbers N, 
                        pMin and pMax;
                        for now gridSigmaM.fSet is empty.
                The integration parameters:
                    nLeg is the number of points used for the Gauss-Legendre 
                    integration,
                        this attribute is fixed as the inputed integer nLeg; 
                    nCheby is the number of points used for the Gauss-Chebyshev 
                    integration,
                        this attribute is fixed as the inputed interger nCheby; 
                    pMin is the lower integration boundary,
                        this attribute is fixed as the inputed float pMin;
                    pMax is the upper integration boundary,
                        this attribute is fixed as the inputed float pMax.
                The attribute related to convergence:
                    epsilon defines the wanted accuracy,
                        this attribute is fixed at 1e-3;
                    converged can be either False or True, (C)
                        whenever this attribute becomes True the convergence is
                        reached;
                        initially converged is False;
                    convState is a list of three elements that describe the
                    current state of convergence, (C)
                        initially convState is [None,0,0];
                        the first element of the list is the difference of 
                        magnitude between the new and the previous values  
                        obtained for the dressing functions;
                        the second element labels the different iterations;
                        the last element saves how many iterations the 
                        condition defined by the attribute epsilon is 
                        fullfilled.
        """
        # We define some parameters.
        self.bareMass = m
        self.mu = mu
        # We define the functions F and G.
        self.Z2=1
        self.lambda_ = 0.72
        self.gamma = 12/25
        self.lambdaQCD = 0.234
        self.eta = 1.8
        k2 = lambda q,z,p : p**2+q**2-2*p*q*z
        self.alpha = lambda k2 : math.pi * self.eta**7 * (k2/self.lambda_**2)**2 * math.exp(-self.eta**2*k2/self.lambda_**2) + (2*math.pi*self.gamma*(1-math.exp(-k2)))/math.log(math.e**2 - 1 + (1 + k2/self.lambdaQCD**2)**2)
        self.F = lambda q,z,p : 3*q*z/p - 2*q**2/k2(q,z,p)*(1-z**2)
        self.G = lambda k2 : self.Z2**2 * 16*math.pi/3 * self.alpha(k2)/k2
        # We define four grids for the A,B and sigma functions. For A and B, we
        # fill their fSet one first time.
        self.gridA = Grid(N)
        self.gridA.mapping_2(pMin,pMax)
        self.gridA.AddData(lambda q : 1)
        self.gridM = Grid(N)
        self.gridM.mapping_2(pMin,pMax)
        if self.bareMass!=0:
            self.gridM.AddData(lambda q : self.bareMass)
        else:
            self.gridM.AddData(lambda q : 1) 
        self.gridSigmaA = Grid(N)
        self.gridSigmaA.mapping_2(pMin,pMax)
        self.gridSigmaM = Grid(N)
        self.gridSigmaM.mapping_2(pMin,pMax)
        # We define parameters for the integration.
        self.nCheby=nCheby
        self.nLeg=nLeg
        self.pMin=pMin
        self.pMax=pMax
        # Finally, the last attributes are related to convergence.
        self.epsilon=1e-6
        self.converged=False
        self.convState=[None,0,0]

    def updateZ2(self):
        """
        Input:
            Nothing.
        Output:
            It updates the attribute self.Z2 with the evaluation of the current
            SigmaA function on p=self.mu. This evaluation is obtained thanks  
            to a cubic slpine interpolation of self.gridSigmaA.
        """
        self.Z2 = 1 - SplineCubInt(self.mu,self.gridSigmaA)
        
    def updateG(self):
        """
        Input:
            Nothing.
        Output:
            It updates the function self.G with the actual value of self.Z2.
        """
        self.G = lambda k2 : self.Z2**2 * 16*math.pi/3 * self.alpha(k2)/k2
        
    def updateSigmas(self):
        """ 
        Input:
            Nothing.
        Output:
            It updates gridSigmaA.fSet and gridSigmaM.fSet list by computing
            the integrals of equation (19) in the project description.
        """
        k2 = lambda q,z,p : p**2+q**2-2*p*q*z
        funcToInt_M = lambda q,z,p : 1/(2*math.pi)**3 * 2*q**3 * 3 * self.G(k2(q,z,p)) * (SplineCubInt(q,self.gridM)/(SplineCubInt(q,self.gridA)*(q**2+SplineCubInt(q,self.gridM)**2)))
        funcToInt_A = lambda q,z,p : 1/(2*math.pi)**3 * 2*q**3 * self.G(k2(q,z,p)) * (1/(SplineCubInt(q,self.gridA)*(q**2+SplineCubInt(q,self.gridM)**2))) * self.F(q,z,p)
        title='roots_weights_'+str(self.nLeg)+'.txt'
        funcIntegrated_M = lambda p : GaussQuad2D_var(funcToInt_M,p,title,self.nCheby,a_q=self.pMin,b_q=self.pMax,transf_q='log')
        funcIntegrated_A = lambda p : GaussQuad2D_var(funcToInt_A,p,title,self.nCheby,a_q=self.pMin,b_q=self.pMax,transf_q='log')
        # print('SigmaM computation')
        self.gridSigmaM.AddData(funcIntegrated_M)
        # print('SigmaA computation')
        self.gridSigmaA.AddData(funcIntegrated_A)
    
    def updateAM(self):
        """ 
        Input:
            Nothing.
        Output:
            It updates the self.gridA.fSet and self.gridM.fSet thanks to the 
            current gridSigmaA and gridSigmaM fSet. It also tests the 
            convergence and updates the attributes self.convState and 
            self.converged.
        """
        convDegree = 0
        aSet = []
        mSet = []
        for i in range(len(self.gridSigmaA.xSet)):
            aValue = 1 + self.gridSigmaA.fSet[i] - SplineCubInt(self.mu,self.gridSigmaA)
            aSet.append(aValue)
            convDegree += abs(aValue-self.gridA.fSet[i])
        self.gridA.fSet = aSet
        self.gridA.fsecSet = []
        for i in range(len(self.gridSigmaA.xSet)):
            mValue = 1/self.gridA.fSet[i] * (self.bareMass +self.gridSigmaM.fSet[i] - SplineCubInt(self.mu,self.gridSigmaM))
            mSet.append(mValue)
            convDegree += abs(mValue-self.gridM.fSet[i])
        self.gridM.fSet = mSet
        self.gridM.fsecSet = []
        # The next lines are used to update the variables of convergence.
        self.convState[0] = convDegree
        self.convState[1]+=1
        if self.convState[0] < self.epsilon:
            self.convState[2]+=1
            if self.convState[2] > 2:
                self.converged = True
        print('Degree of convergence : ' + str(convDegree))
    
    def Saver(self,fileName=None):
        """ 
        Input:
            fileName : a string or None.
                This will be used as name for the created file. If None is 
                given, the name will be "Result_i_[insert the date]".
        Output:
            None but it saves the current values for Z2 as well as all the
            points A(p) and M(p) in a txt file. The file also includes the date
            and the label of the iteration, a recap of the physical parameters,
            a recap of the integration parameters and the actual degree of 
            convergence.
        """
        today = date.today()
        if fileName == None:
            time = today.strftime("%y-%m-%d")
            fileName='Result_' + str(self.convState[1])+'_' + time + '.txt'
        file=open(fileName,'w')
        abrev = 'th'
        if self.convState[1]!=11 and self.convState[1]%10==1:
            abrev = 'st'
        elif self.convState[1]!=12 and self.convState[1]%10==2:
            abrev = 'nd'
        file.write('Results from the ' + str(self.convState[1]) + abrev + ' iteration     date:' + today.strftime("%d/%m/%y") + '\n')
        file.write('-'*72 + '\n')
        file.write('Parameters of the run are : \n')
        file.write('m='+str(self.bareMass) + ' mu='+str(self.mu) + ' lambda='+str(self.lambda_) + ' lambdaQCD=' + str(self.lambdaQCD) + ' gamma=' + str(self.gamma) + ' eta='+str(self.eta)+'\n'+'\n')
        file.write('Parameters for the integartions are : \n')
        file.write('nCheby=' + str(self.nCheby) + ' nLeg=' + str(self.nLeg) + ' pMin=' + str(self.pMin) + ' pMax=' + str(self.pMax) + '\n \n')
        file.write('Degree of convergence : ' + str(self.convState[0]) + '\n \n')
        file.write('Current value for Z2 : ')
        file.write(str(self.Z2) + '\n'+'\n')
        file.write('Graph for A(p) and M(p) :\n')
        file.write("{:<25} {:<25} {:<25}".format('p', 'A(p)', 'M(p)')+'\n')
        for i in range(len(self.gridSigmaA.xSet)):
            file.write("{:<25} {:<25} {:<25}".format(str(self.gridA.xSet[i]),str(self.gridA.fSet[i]),str(self.gridM.fSet[i]))+'\n')
        file.close()


#----------------- Functions for the analysis of the results -----------------#


def MakeGraph(fileNameList,title):
    """
    Input:
        fileNameList : a list of string.
            Each string must be the name of a file created with the method
            "Saver".
        title : a string.
            This will be the title of the graph.
    Output:
        None but it displays a graph of A(p) and M(p) based on the extractions 
        of the datas inside each file of fileNameList.
    """
    fig = plt.figure()
    fig.suptitle(title, fontsize=16)
    Axes1 = fig.add_subplot(2,1,1)
    Axes2 = fig.add_subplot(2,1,2)
    Axes1.set_xscale("log")
    Axes1.set_yscale("log")
    Axes1.set_ylabel('A(p) (DN)')
    Axes2.set_xscale("log")
    Axes2.set_yscale("log")
    Axes2.set_ylabel('M(p) (in GeV)')
    Axes2.set_xlabel('p (in GeV)')
    for fileName in fileNameList:
        pList=[]
        aList=[]
        mList=[]
        bool_=False
        file=open(fileName,'r')
        for line in file: 
            splittedResults=line.split()
            if bool_ and line!='':
               pList.append(float(splittedResults[0]))
               aList.append(float(splittedResults[1]))
               mList.append(float(splittedResults[2]))
            if splittedResults==['p','A(p)','M(p)']:
                   bool_=True
        file.close()
        graphName=fileName.split('_')[1]
        Axes1.scatter(pList,aList,label='Iteration '+graphName)
        Axes2.scatter(pList,mList,label='Iteration '+graphName)
    Axes1.legend()
    Axes2.legend()
    plt.show() 

def MakeIntGraph(fileNameList,pRange,title):
    """
    Input:
        fileNameList : a list of string.
            Each string must be the name of a file created with the method
            "Saver".
        pRange : a list (or array-like) of floats.
            A point will be drawn on the graph for each p in pRange.
        title : a string.
            This will be the title of the graph.
    Output:
        None but it displays an interpolate graph (cubic spline) of A(p) and 
        M(p) based on the extractions of the datas inside each file of 
        fileNameList. 
    """
    fig = plt.figure()
    fig.suptitle(title, fontsize=16)
    Axes1 = fig.add_subplot(2,1,1)
    Axes2 = fig.add_subplot(2,1,2)
    Axes1.set_xscale("log")
    Axes1.set_yscale("log")
    Axes1.set_ylabel('A(p) (DN)')
    Axes2.set_xscale("log")
    Axes2.set_yscale("log")
    Axes2.set_ylabel('M(p) (in GeV)')
    Axes2.set_xlabel('p (in GeV)')
    Axes2.set_ylim([1e-3,1e1])
    for fileName in fileNameList:
        pList,aList,mList=[],[],[]
        bool_=False
        file=open(fileName,'r')
        for line in file: 
            splittedResults=line.split()
            if bool_ and line!='':
               pList.append(float(splittedResults[0]))
               aList.append(float(splittedResults[1]))
               mList.append(float(splittedResults[2]))
            if splittedResults==['p','A(p)','M(p)']:
                   bool_=True
        file.close()
        gridA=Grid(3)
        gridM=Grid(3)
        gridA.xSet=pList
        gridM.xSet=pList
        gridA.fSet=aList
        gridM.fSet=mList
        aList,mList=[],[]
        for p in pRange:
            aList.append(SplineCubInt(p,gridA))
            mList.append(SplineCubInt(p,gridM))
        graphName=fileName.split('_')[1]
        Axes1.scatter(pRange,aList,label='Iteration '+graphName,s=2)
        Axes2.scatter(pRange,mList,label='Iteration '+graphName,s=2)
    Axes1.legend()
    Axes2.legend()
    plt.show()

def MakeIntAnim(fileNameList,pRange,title,animName='AnimatedPlot.gif'):
    """
    Input:
        fileNameList : a list of string.
            Each string must be the name of a file created with the method
            "Saver".
        pRange : a list (or array-like) of floats.
            A point will be drawn on the graph for each p in pRange.
        title : a string.
            This will be the title of the graph.
        animName : a string.
            This will be the name of the generated .gif.
    Output:
        None but it displays an animation of the convergence of A(p) and 
        M(p) based on the extractions of the datas inside each file of 
        fileNameList. The curves A(p) and M(p) are drawn thanks to a cubic
        spline interpolation. The animation is also saved.
    """
    dataSet=[]
    for fileName in fileNameList:
        pList,aList,mList=[],[],[]
        bool_=False
        file=open(fileName,'r')
        for line in file: 
            splittedResults=line.split()
            if bool_ and line!='':
               pList.append(float(splittedResults[0]))
               aList.append(float(splittedResults[1]))
               mList.append(float(splittedResults[2]))
            if splittedResults==['p','A(p)','M(p)']:
                   bool_=True
        file.close()
        gridA=Grid(3)
        gridM=Grid(3)
        gridA.xSet=pList
        gridM.xSet=pList
        gridA.fSet=aList
        gridM.fSet=mList
        aList,mList=[],[]
        for p in pRange:
            aList.append(SplineCubInt(p,gridA))
            mList.append(SplineCubInt(p,gridM))
        dataSet.append([pRange,aList,mList])
    fig = plt.figure()
    fig.suptitle(title, fontsize=16)
    colors = plt.get_cmap('copper', len(fileNameList))
    Axes1 = fig.add_subplot(2,1,1)
    Axes1.set_xscale("log")
    Axes1.set_yscale("log")
    Axes1.set_ylabel('A(p) (DN)')
    Axes1.set_xlim([1e-3,1e3])
    Axes1.set_ylim([6e-1,2e0])
    Axes2 = fig.add_subplot(2,1,2)
    Axes2.set_xscale("log")
    Axes2.set_yscale("log")
    Axes2.set_ylabel('M(p) (in GeV)')
    Axes2.set_xlabel('p (in GeV)')
    Axes2.set_xlim([1e-3,1e3])
    Axes2.set_ylim([1e-1,1e2])
    plt.subplots_adjust(left=0.15, bottom=0.15)
    T=range(1,len(fileNameList)+1)
    f_A, = Axes1.plot([], [], linewidth=2.5)
    f_M, = Axes2.plot([], [], linewidth=2.5)
    tempA = Axes1.text(5e1, 1.6e0, '', fontsize=12)
    tempM = Axes2.text(5e1, 3e-1, '', fontsize=12)
    def animate(i):
        p = dataSet[i-1][0]
        A = dataSet[i-1][1]
        M = dataSet[i-1][2]
        f_A.set_data(p, A)
        f_A.set_color(colors(i))
        f_M.set_data(p, M)
        f_M.set_color(colors(i))
        tempA.set_text('Iteration ' + str(int(T[i-1])))
        tempA.set_color(colors(i))
        tempM.set_text('Iteration ' + str(T[i-1]))
        tempM.set_color(colors(i))
        return f_M
    ani = FuncAnimation(fig=fig, func=animate, frames=T, interval=100, repeat=True)
    ani.save(animName, fps=6)

#--------------------- Instructions for the resolution -----------------------#


""" Example : 
sol=Solution(50, pMin=1e-3, pMax=1e3, m=0.004, mu=19, nLeg=1000, nCheby=50)
while not(sol.converged) and sol.convState[1]<=500:
    sol.updateSigmas()
    sol.updateAM()
    sol.updateZ2()
    sol.updateG()
    print('End of an iteration :',sol.convState[1])
    sol.Saver()
"""

""" 
fileNameList=[]
for i in range(11,12):
    name='Result_'+str(i)+'_22-05-16.txt'
    fileNameList.append(name)
pRange=numpy.logspace(-3,3,3000)
MakeIntGraph(fileNameList,pRange,'Dressing Function (\u03BC=19 GeV, m=4.18 GeV)')
"""

fileNameList=['Result_10_22-05-17.txt','Result_11_22-05-16.txt','Result_18_22-05-15.txt','Result_30_22-05-14.txt','Result_37_22-05-17.txt']
pRange=numpy.logspace(-2,2,3000)
for element in fileNameList:
    if element=='Result_10_22-05-17.txt':
        Title = 'Charm Mass Function'
    if element=='Result_11_22-05-16.txt':
        Title = 'Bottom Mass Function'
    if element=='Result_18_22-05-15.txt':
        Title = 'Strange Mass Function'
    if element=='Result_30_22-05-14.txt':
        Title = 'Up Mass Function'
    if element=='Result_37_22-05-17.txt':
        Title = 'Chiral Mass Function'
    MakeIntGraph([element],pRange,Title)

0