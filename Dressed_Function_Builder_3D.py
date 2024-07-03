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
from datetime import date
from matplotlib.animation import FuncAnimation
from dressed_funct_build import *
import cmath

#----------------------------- Solution object -------------------------------#


class ComplexSolution(Solution):
    """ This class contains an approximated solution for the Dyson-Schwinger
    equation. This approximated solution can the be improved with an iterative
    process.
    """
    def __init__(self,N,pMin=1e-3,pMax=1e3,m=4e-3,mu=19,nLeg=1000,nCheby=50):
        """ 
        Input:
            See __init__ of the class Solution.
        Output:
            It creates an object of class ComplexSolution. This class inherits
            all the methods and the attributes of the class Solution. 
            Nevertheless, there are a few differences :
                the attribute self.alpha is now generalized for complex values 
                of k2.
                Six new attributes have been added:
                    complexSigmaA is (for now) an empty list,
                        this list will be filled with doublets whose first 
                        element is a the complex value for p and whose second 
                        element is the corresponding value for the function 
                        Sigma_A (see equation (19) in the project description);
                    complexSigmaM is (for now) an empty list,
                        this list will be filled with doublets whose first 
                        element is a the complex value for p and whose second 
                        element is the corresponding value for the function 
                        Sigma_M (see equation (19) in the project description);
                    sigmaAmu is (for now) None,
                        this will be replaced bythe evaluation of Sigma_A for 
                        p=self.mu;
                    sigmaMmu is (for now) None,
                        this will be replaced bythe evaluation of Sigma_M for 
                        p=self.mu;
                    complexSigmaV is (for now) an empty list,
                        this list is similar to self.complexSigmaA but do not
                        contains the values of Sigma_M but of sigma_v (this  
                        list is build thanks to equation (2) and (25) in the 
                        project description);
                    complexSigmaS is (for now) an empty list,
                        this list is similar to self.complexSigmaA but do not
                        contains the values of Sigma_A but of sigma_s (this  
                        list is build thanks to equation (2) and (25) in the 
                        project description);
        """
        super().__init__(N,pMin=pMin,pMax=pMax,m=m,mu=mu,nLeg=nLeg,nCheby=nCheby)
        self.alpha = lambda k2 : math.pi * self.eta**7 * (k2/self.lambda_**2)**2 * cmath.exp(-self.eta**2*k2/self.lambda_**2) + (2*math.pi*self.gamma*(1-cmath.exp(-k2)))/cmath.log(math.e**2 - 1 + (1 + k2/self.lambdaQCD**2)**2)
        self.complexSigmaA=[] # Maybe I could use some objects Grid here but it
        self.complexSigmaM=[] # should require some modifications on it.
        self.sigmaAmu=None
        self.sigmaMmu=None
        self.complexSigmaV=[]
        self.complexSigmaS=[]
        
    def UpLoader(self,fileName):
        """
        Input:
            fileName : a string.
                This string must be the name of a file created with the method
                "Saver".
        Output:
            None but the attributes gridA and gridM of the object self are
            fulled with the datas inside fileName.
        """
        fileName = 'Result_30_22-05-14.txt'
        pList,aList,mList = [],[],[]
        bool_ = False
        file = open(fileName,'r')
        for line in file: 
            splittedResults=line.split()
            if bool_ and line != '':
                pList.append(float(splittedResults[0]))
                aList.append(float(splittedResults[1]))
                mList.append(float(splittedResults[2]))
            if splittedResults == ['p','A(p)','M(p)']:
                bool_ = True
        file.close()
        self.gridA.xSet = pList
        self.gridM.xSet = pList
        self.gridA.fSet = aList
        self.gridM.fSet = mList
        return
    
    def ComplexSigmasCalc(self,complexRange):
        """
        Input:
            complexRange : a list (array-like) of complex numbers
        Output:
            None but some of the new attributes of ComplexSolution are filled:
                complexSigmaA is filled with doublets,
                    the first element of the doublet is one of the complex 
                    numbers inside complexRange and the second element of the 
                    doublet is the corresponding value for the function Sigma_A
                    (see equation (19) in the project description);
                complexSigmaM is filled with doublets,
                    the first element of the doublet is one of the complex 
                    numbers inside complexRange and the second element of the 
                    doublet is the corresponding value for the function Sigma_M
                    (see equation (19) in the project description);
                self.sigmaAmu replaced by a float,
                    this float is the evaluation of Sigma_A for p=self.mu;
                self.sigmaMmu replaced by a float,
                    this float is the evaluation of Sigma_M for p=self.mu;
        """
        k2 = lambda q,z,p : p**2+q**2-2*p*q*z
        funcToInt_M = lambda q,z,p : 1/(2*math.pi)**3 * 2*q**3 * 3 * self.G(k2(q,z,p)) * (SplineCubInt(q,self.gridM)/(SplineCubInt(q,self.gridA)*(q**2+SplineCubInt(q,self.gridM)**2)))
        funcToInt_A = lambda q,z,p : 1/(2*math.pi)**3 * 2*q**3 * self.G(k2(q,z,p)) * (1/(SplineCubInt(q,self.gridA)*(q**2+SplineCubInt(q,self.gridM)**2))) * self.F(q,z,p)
        title='roots_weights_'+str(self.nLeg)+'.txt'
        funcIntegrated_M = lambda p : GaussQuad2D_var(funcToInt_M,p,title,self.nCheby,a_q=self.pMin,b_q=self.pMax,transf_q='log')
        funcIntegrated_A = lambda p : GaussQuad2D_var(funcToInt_A,p,title,self.nCheby,a_q=self.pMin,b_q=self.pMax,transf_q='log')
        complexSigmaA=[]
        complexSigmaM=[]
        for complexP in complexRange:
            if complexP==complex(0,0):
                pass
            complexSigmaM.append([complexP,funcIntegrated_M(complexP)])
            complexSigmaA.append([complexP,funcIntegrated_A(complexP)])
            print(str(complexP**2))
        sigmaAmu=funcIntegrated_A(self.mu)
        sigmaMmu=funcIntegrated_M(self.mu)
        self.complexSigmaA=complexSigmaA
        self.complexSigmaM=complexSigmaM
        self.sigmaAmu=sigmaAmu
        self.sigmaMmu=sigmaMmu
    
    def ComplexSmallSigmasCalc(self):
        """
        Input:
            Nothing.
        Output:
            None but some of the new attributes of ComplexSolution are filled:
                self.complexSigmaV is filled with doublets,
                    the first element of the doublet is one of the complex 
                    numbers inside complexRange and the second element of the 
                    doublet is the corresponding value for the function sigma_v
                    (see equation (2) and (25) in the project description);
                self.complexSigmaS is filled with doublets,
                    the first element of the doublet is one of the complex 
                    numbers inside complexRange and the second element of the 
                    doublet is the corresponding value for the function sigma_s
                    (see equation (2) and (25) in the project description);
        """
        complexA,complexM = [],[]
        for i in range(len(self.complexSigmaA)):
            complexA.append([self.complexSigmaA[i][0], 1 + self.complexSigmaA[i][1] - self.sigmaAmu])
        for j in range(len(self.complexSigmaM)):  
            complexM.append([self.complexSigmaM[j][0], 1/complexA[j][1]*(sol.bareMass + self.complexSigmaM[j][1] - self.sigmaMmu)])
        complexSigmaV,complexSigmaS = [],[]
        for k in range(len(complexA)):
            complexSigmaV.append([complexA[k][0],1/(complexA[k][1]*(complexA[k][0]**2 + complexM[k][1]**2))])
            complexSigmaS.append([complexA[k][0],complexM[k][1]/(complexA[k][1]*(complexA[k][0]**2 + complexM[k][1]**2))])
        self.complexSigmaV = complexSigmaV
        self.complexSigmaS = complexSigmaS
        
    def ComplexSaver(self,fileName=None):
        """ 
        Input:
            fileName : a string or None.
                This will be used as name for the created file. If None is 
                given, the name will be "Result_i_[insert the date]".
        Output:
            None but it saves the current values for Z2 as well as all the 
            complex points sigma_v(p) and sigma_s(p) in a txt file. The file 
            also includes the date and the label of the iteration, a recap of 
            the physical parameters, a recap of the integration parameters and
            the actual degree of convergence.
        """
        today = date.today()
        if fileName == None:
            time = today.strftime("%y-%m-%d")
            fileName='ComplexResult_' + time + '.txt'
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
        file.write('Datas for sigma_v(p) and sigma_s(p) :\n')
        file.write("{:<50} {:<50} {:<50}".format('p', 'sigma_v(p)', 'sigma_s(p)')+'\n')
        for i in range(len(self.complexSigmaV)):
            file.write("{:<50} {:<50} {:<50}".format(str(self.complexSigmaV[i][0]),str(self.complexSigmaV[i][1]),str(self.complexSigmaS[i][1]))+'\n')
        file.close()

def MakeGraph(fileName,title):
    """
    Input:
        fileName : a string.
            This string must be the name of a file created with the method
            "ComplexSaver".
        title : a string.
            This will be the title of the graph.
    Output:
        None but it displays a complex graph of |sigma_v(p2)| and |sigma_s(p2)|
        based on the extractions of the datas inside fileName.
    """
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig1.suptitle(title, fontsize=16)
    fig2.suptitle(title, fontsize=16)
    Axes1 = fig1.add_subplot(projection='3d')
    Axes2 = fig2.add_subplot(projection='3d')
    Axes1.set_xlabel('Re(p^2)')
    Axes1.set_ylabel('Im(p^2)')
    Axes1.set_zlabel('Re(sigma_v(p^2))')
    Axes1.set_zlim([-7.5,7.5])
    Axes2.set_xlabel('Re(p^2)')
    Axes2.set_ylabel('Im(p^2)')
    Axes2.set_zlabel('Re(sigma_s(p^2))')
    Axes2.set_zlim([-7.5,7.5])
    ReP2List=[]
    ImP2List=[]
    sigmavList=[]
    sigmasList=[]
    bool_=False
    file=open(fileName,'r')
    for line in file: 
        splittedResults=line.split()
        if bool_ and line!='':
            p=complex(splittedResults[0])
            p2=p**2
            ReP2List.append(p2.real)
            ImP2List.append(p2.imag)
            sigmavList.append((complex(splittedResults[1])).real)
            sigmasList.append((complex(splittedResults[2])).real)
        if splittedResults==['p','sigma_v(p)','sigma_s(p)']:
            bool_=True
    file.close()
    Axes1.scatter(ReP2List,ImP2List,sigmavList)
    Axes2.scatter(ReP2List,ImP2List,sigmasList)
    plt.show() 


""" 
sol=ComplexSolution(50, pMin=1e-3, pMax=1e3, m=0.004, mu=19, nLeg=500, nCheby=30)
sol.UpLoader('Result_30_22-05-14.txt')
ReP2Range=numpy.arange(-1,1,0.05)
ImP2Range=numpy.arange(-1,1,0.05)
ComplexRange=[]
# These lines are added to build a nice ComplexRange that will at the end 
# produce a plot on the [-1,1]*[-1,1] region of the p^2 complex plane. 
for Re in ReP2Range:
    for Im in ImP2Range:
        p=cmath.sqrt(complex(Re,Im))
        ComplexRange.append(p)
sol.ComplexSigmasCalc(ComplexRange)
sol.ComplexSmallSigmasCalc()
sol.ComplexSaver()
"""

MakeGraph("ComplexResult_22-05-17.txt",'Complex plot')

