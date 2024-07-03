##########################################################################################
#####################                    Authors                     #####################
################     Chevalier Cyrille, Cimino Lorenzo, Trotti Enrico     ################
##########################################################################################

##########################################################################################
#####################           Solution of the quark DSE            #####################
##########################################################################################

import math
import numpy
import time
import matplotlib.pyplot as plt

#------------------------- Legendre polynomial -------------------------------#

class LegendrePoly():
    """
    This class allows calculations with Legendre polynomials. DISCLAIMER : this
    class is essentially build to perform Gauss-Legendre integration. To really
    deal with Legendre Polynomial, please use "Module_Integration_Sym.py".
    """
    def __init__(self,order):
        """
        Input:
            order : an integer which refers to the order of the wanted Legendre
            polynomial.
        Output:
            It creates an object of class LegendrePoly. This object has two 
            attributes:
                order which is the inputed integer order,
                    this is the order of the polynomial;
                previous which is an object of class LegendrePoly,
                    this is the Legendre polynomial of order self.order-1.
            Actually, this is simply a tower of nested objects LegendrePoly
            wich only have one attribute: their order.
        """
        if order==0:
            self.order=0
            self.previous=None
        else:
            self.order=order
            prevLegPoly=LegendrePoly(order-1)
            self.previous=prevLegPoly
            

    def Eval(self,x):
        """
        Input:
            x : a float.
        Output:
            It returns a list of two elements. These elements are the evalution
            of the Legendre polynomial of order self.order and self.order-1 at 
            x. DISCLAIMER : once more, this method is optimized to perform 
            Gauss-Legendre integration and not simply to evaluate the 
            polynomial.
        """
        # Evaluation of Legendre polynomial of order 0 gives 1 for all x.
        # There is no polynomial with order -1 so the second element of the
        # returned list is None.
        if self.order == 0:
            return [1,None]
        # Evaluation of Legendre polynomial of order 1 gives x for all x.
        # Evaluation of Legendre polynomial of order 1-1=0 gives 1 for all x.
        elif self.order == 1:
            return [x,1]
        # Evaluation of Legendre polynomial of other order is given by a 
        # recurrence formula. That gives the first element of the list.
        # The recurrence formula needs the evaluation of the previous Legendre 
        # polynomial. So the second element of the list has already been 
        # computed and then can easily be returned.
        else:
            [previous,secPrevious]=self.previous.Eval(x)
            solve = (2*self.order-1)/self.order*x*previous-(self.order-1)/self.order*secPrevious
            return [solve,previous] 
    
    def Deriv(self,x,legPolyEval=None):
        """
        Input:
            x : a float.
            legPolyEval : a list given by the method Eval or None.
                It must be the list given by the evaluation at x of the
                Legendre polynomial that we want to differentiate. If None is 
                given, the method will takes some time to perform this
                evaluation. 
        Output:
            It returns the evalution of the derivative of the Legendre 
            polynomial self at x. DISCLAIMER : once more, this method is 
            optimized to perform Gauss-Legendre integration and not simply to 
            evaluate the polynomial.
        """
        # If no legPolyEval was given, the first step is to perform its 
        # evaluation (thanks to the method Eval). If not, this calculation is 
        # avoided and time is saved.
        if legPolyEval==None:
            legPolyEval=self.Eval(x)
        # The calculation of the derivative uses a recurrence formula. This
        # formula needs the evaluation of the Legendre polynomial of order 
        # self.order+1 but this evaluation can be obtained from the recurrence
        # formula used in the method Eval (no other loop is needed, every 
        # necessary evaluations are already computed). 
        n = self.order
        [current,previous]=legPolyEval
        nextP = (2*n+1)/(n+1)*x*current-(n)/(n+1)*previous
        solve = ((n+1)*x*current-(n+1)*nextP)/(1-x**2)
        return solve
    
    def FindRoot(self,start,precision=10**(-12)):
        """
        Input:
            start : a float (a starting guess for the root).
            precision : a (small) float (the precision for the determination).
        Output:
            It returns a root of the Legendre polynomial self around start. 
        """
        oldX = start
        # Reminder : p.Eval(x) returns the evaluation at x of both the Legendre
        # polynomial p and the previous one.
        [current,previous]=self.Eval(oldX)
        # As the Legender polynomial has already been evaluated, it is given in
        # the arguments of the method Deriv.
        newX = oldX - current/self.Deriv(oldX,legPolyEval=[current,previous])
        while abs(oldX-newX) > precision:
           oldX = newX
           [current,previous]=self.Eval(oldX)
           newX = oldX - current/self.Deriv(oldX,legPolyEval=[current,previous])
        return newX
    
    def ComputeRootsAndWeights(self,precision=10**(-12)):
        """
        Input:
            precision : it is a float that describes how precise will be the
            root determination.
        Output:
            It returns a list of two lists, the first contains all the roots 
            the Legendre polynomial self while the second contains all the 
            weights of the Legendre polynomial self. It also creates a txt file 
            that includes all theses roots (first element of each line) and the all these 
            weights (second element of each line).
        """
        solveRoots=[]
        solveWeights=[]
        # The next line creates the name of the txt file and open it.
        title='roots_weights_'+str(self.order)+'.txt'
        f=open(title,'w')
        for index in range(1,self.order+1):
            # The starting point for the root is choosen in order to ensure 
            # that all the roots will be reached.
            start=numpy.cos(numpy.pi*(index - 0.25)/(self.order + 0.5))
            root=self.FindRoot(start,precision=precision)
            solveRoots.append(root)
            # The weight is obtained thanks to a well-known formula.
            weight=2/((1-root**2)*self.Deriv(root)**2)
            solveWeights.append(weight)
            # Notice that every time a root and a weigth are computed the file 
            # is completed. So if the run is interrupted, the computed roots 
            # and weights are saved (I hope...).
            f.write(str(root)+' '+str(weight)+'\n')
        f.close()
        return [solveRoots,solveWeights]

#------------------------ Gauss-Legendre quadrature --------------------------#


def LinTranslator(a,b,x):
    """
    Input:
        a : a float.
        b : a float smaller than a.
        x : a float between -1 and 1.
    Output:
        A float which is the linear translation and dilatation of x in the 
        interval [a,b]
    """
    if x>1 or x<-1:
        raise Exception('x must be in [-1,1]')
        return
    solve=((b-a)*x+a+b)/2
    return solve

def LogTranslator(a,b,x):
    """
    Input:
        a : a float.
        b : a float smaller than a.
        x : a float between -1 and 1.
    Output:
        A float which is the logarithmic translation and dilatation of x in the 
        interval [a,b].
    """
    if x>1 or x<-1:
        raise Exception('x must be in [-1,1]')
        return
    A=-math.log(a*b)/math.log(b/a)
    B=2/math.log(b/a)
    solve=numpy.exp((x-A)/B)
    return solve

def LegQuadrature(func,fileName,a=-1,b=1,transf='lin'):
    """
    Input:
        func : a function of one float variable (defined on [a,b]).
        fileName : a string.
            This must be the name of the txt document that contains the roots
            and the weigths of the Legendre Polynomial.
        a : a float.
        b : a float bigger than a.
        transf : can be 'lin' or 'log'.
            It indicates which type of interval translator will be performed 
            if [a,b] is not [-1,1].
    Output:
        A float which is a value for the integral of the function func(x) on
        the interval [a,b]. This routine uses Gauss-Legendre integration.
    """
    
    # The next lines open the file called "fileName", read it and build a list 
    # that contains all the roots as well as a list that contains all the 
    # weights from the reading.

    f=open(fileName, 'r')
    listOfRoots=[]
    listOfWeights=[]
    for line in f: 
        numbers=line.split()
        listOfRoots.append(float(numbers[0]))
        listOfWeights.append(float(numbers[1]))
    f.close()
    n=len(listOfRoots)
    solve=0
    if a==-1 and b==1:
        for i in range(n):
            solve+=listOfWeights[i]*func(listOfRoots[i])
        return solve
    else :
        if transf=='lin':
            for i in range(n):
                solve+=listOfWeights[i]*func(LinTranslator(a,b,listOfRoots[i]))
            return (b-a)/2*solve
        if transf=='log':
            A=-math.log(a*b)/math.log(b/a)
            B=2/math.log(b/a)
            for i in range(n):
                solve+=listOfWeights[i]*1/B*math.exp((listOfRoots[i]-A)/B)*func(LogTranslator(a,b,listOfRoots[i]))
            return solve

""" Example : 
for i in range(500,1000,100): # Over 1400 I get an error :
    starting=time.time()       # maximum recursion depth exceeded
    P=LegendrePoly(i)
    P.ComputeRootsAndWeights()
    print(i, ' done') 
    print(time.time()-starting,'\n') """

""" Example : 
func = lambda x : (1/x)**2
for i in range(500,1000,100):
    title='roots_weights_'+str(i)+'.txt'
    starting=time.time()
    print(i,LegQuadrature(func,title,a=1,b=10**6,transf='log')-1)
    print(time.time()-starting)"""

#------------------------- Chebyshev polynomials -----------------------------#


class ChebyshevPoly:
    """
    This class is symbolizing a Chebyshev polynomial. It does NOT inherit the
    methods of the class Polynomial.
    """
    def __init__(self,order):
        """
        Input:
            order : an integer which refers to the order of the wanted 
            Chebyshev polynomial.
        Output:
            It creates an object of class ChebyshevPoly. This object has three 
            attributes:
                listOfRoots which is an empty list,
                    this is a memory place to store the computed roots of the
                    polynomial ;
                listOfWeights which is an empty list,
                    this is a memory place to store the weights associated to
                    the Legendre polynomial (see Gauss-Legendre integration);
                order which is the inputed integer order,
                    this is the order of the polynomial;
        """
        self.order = order
        self.listOfRoots=[]
        self.listOfWeights=[]
    
    def FindAllRoots(self):
        """
        Input:
            Nothing.
        Output:
            It returns a list of all the roots of the Chebyshev polynomial. It
            also replaces the attribute listOfRoots of the ChebyshevPoly by 
            this list (previous value for listOfRoots is forgotten).
        """
        solve = []
        for i in range(self.order):
            x_i = numpy.cos(numpy.pi*(i+1)/(self.order+1))
            solve.append(x_i)
        self.listOfRoots = solve
        return solve
    
    def ComputeWeights(self):
        """
        Input:
            Nothing.
        Output:
            It returns a list of all the weights associated to the Chebyshev 
            polynomial. It also replaces the attribute listOfWeights of the
            ChebyshevPoly by this list (previous value for listOfWeights is 
            forgotten).
        """
        solve = []
        for i in range(self.order):
            w_i = numpy.pi/(self.order+1)*numpy.sin(numpy.pi*(i+1)/(self.order+1))**2
            solve.append(w_i)
        self.listOfWeights = solve
        return solve
    
    def Eval(self,x):
        """
        Input:
            x : a float.
        Output:
            It returns the evaluation of the Chebyshev polynomial on the point 
            x.
        """
        if x==1:
            return self.order + 1
        elif x ==-1:
            if self.order % 2 == 0:
                return self.order +1
            else:
                return -(self.order + 1)
        else:
            theta = numpy.arccos(x)
            solve = numpy.sin((self.order+1)*theta)/numpy.sin(theta)
        return solve


#---------------------- Gauss-Chebyshev quadrature ---------------------------#


def ChebQuadrature(func,n,a=-1,b=1,transf='lin'):
    """
    Input:
        func : a function of one float variable (defined on [a,b]).
        n : an integer 
            This refers to the order of the Gauss-Chebyshev integration, 
            basically the higher is n the more precise is the integration.
        a : a float.
        b : a float bigger than a.
        transf : can be 'lin' or 'log'.
            It indicates which type of interval translator will be performed 
            if [a,b] is not [-1,1].
    Output:
        A float which is a value for the integral of the function 
        \sqrt(1-x^2)*func(x) on the interval [a,b]. This routine uses 
        Gauss-Chebyshev integration.
    """
    ChebPoly=ChebyshevPoly(n)
    ChebPoly.FindAllRoots()
    ChebPoly.ComputeWeights()
    solve=0
    if a==-1 and b==1:
        for i in range(n):
            solve+=ChebPoly.listOfWeights[i]*func(ChebPoly.listOfRoots[i])
        return solve
    else :
        if transf=='lin':
            for i in range(n):
                solve+=ChebPoly.listOfWeights[i]*func(LinTranslator(a,b,ChebPoly.listOfRoots[i]))
            return (b-a)/2*solve
        if transf=='log':
            A=-math.log(a*b)/math.log(b/a)
            B=2/math.log(b/a)
            for i in range(n):
                solve+=ChebPoly.listOfWeights[i]*1/B*math.exp((ChebPoly.listOfRoots[i]-A)/B)*func(LogTranslator(a,b,ChebPoly.listOfRoots[i]))
            return solve
        
""" Example :
func = lambda x : 1
print(ChebQuadrature(func,5)-math.pi/2)
"""


#----------------------- 2D Gauss-like quadrature ---------------------------#


def GaussQuad2D(func,n_p,n_z,a_p=0,b_p=1,a_z=-1,b_z=1,transf_p='lin',transf_z='lin',precision=10**(-12)):
    """
    Input: 
        func : a function of two float variables (p and z).
            This function must be define on the rectangle [a_p,b_p]*[a_z,b_z].
        n_p : an integer 
            This refers to the order of the Gauss-Legendre integration, 
            basically the higher is n_p the more precise is the integration.
        n_z : an integer 
            This refers to the order of the Gauss-Chebyshev integration, 
            basically the higher is n_z the more precise is the integration.
        a_p : a float.
        b_p : a float bigger than a_p.
        a_z : a float.
        b_z : a float bigger tah a_z.
        transf_p : can be 'lin' or 'log'.
            It indicates which type of interval translator will be performed 
            to reach [a_p,b_p] from [-1,1]. 
        transf_z : can be 'lin' or 'log'.
            It indicates which type of interval translator will be performed 
            to reach [a_z,b_z] from [-1,1].
        precision : a (small) float that define how precise will be the
        determination of the root for Gauss-Legendre integration.
    Output:
        A float which is a value for the integral of the function 
        \sqrt(1-z^2)*func(p,z) on the rectangle [a_p,b_p]*[a_z,b_z]. This 
        routine uses a combination of Gauss-Legendre and Gauss-Chebyshev 
        integration.
    """
    LegPoly=LegendrePoly(n_p)
    LegPoly.FindAllRoots(precision=precision)
    LegPoly.ComputeWeights()
    ChebPoly=ChebyshevPoly(n_z)
    ChebPoly.FindAllRoots()
    ChebPoly.ComputeWeights()
    solve=0
    if transf_p=='lin' and transf_z=='lin':
        for i in range(n_p):
            for j in range(n_z):
                solve+=ChebPoly.listOfWeights[j]*LegPoly.listOfWeights[i]*func(LinTranslator(a_p,b_p,LegPoly.listOfRoots[i]),LinTranslator(a_z,b_z,ChebPoly.listOfRoots[j]))
        solve*=(b_z-a_z)/2*(b_p-a_p)/2
    if transf_p=='log' and transf_z=='lin':
        A=-math.log(a_p*b_p)/math.log(b_p/a_p)
        B=2/math.log(b_p/a_p)
        for i in range(n_p):
            for j in range(n_z):
                solve+=ChebPoly.listOfWeights[j]*LegPoly.listOfWeights[i]*1/B*math.exp((LegPoly.listOfRoots[i]-A)/B)*func(LogTranslator(a_p,b_p,LegPoly.listOfRoots[i]),LinTranslator(a_z,b_z,ChebPoly.listOfRoots[j]))
    return solve

def GaussQuad2D_var(func,p,fileName,n_z,a_q=0,b_q=1,a_z=-1,b_z=1,transf_q='lin',transf_z='lin',precision=10**(-12)):
    """
    Input: 
        func : a function of three float variables (q, z and p).
            This function must be define on the rectangle [a_p,b_p]*[a_z,b_z].
        p : a float (the fixed value of p in func)
        n_q : an integer 
            This refers to the order of the Gauss-Legendre integration, 
            basically the higher is n_p the more precise is the integration.
        n_z : an integer 
            This refers to the order of the Gauss-Chebyshev integration, 
            basically the higher is n_z the more precise is the integration.
        a_q : a float.
        b_q : a float bigger than a_q.
        a_z : a float.
        b_z : a float bigger tah a_z.
        transf_q : can be 'lin' or 'log'.
            It indicates which type of interval translator will be performed 
            to reach [a_q,b_q] from [-1,1]. 
        transf_z : can be 'lin' or 'log'.
            It indicates which type of interval translator will be performed 
            to reach [a_z,b_z] from [-1,1].
        precision : a (small) float that define how precise will be the
        determination of the root for Gauss-Legendre integration.
    Output:
        A float which is a value for the integral of the function 
        \sqrt(1-z^2)*func(q,z) on the rectangle [a_q,b_q]*[a_z,b_z]. This 
        routine uses a combination of Gauss-Legendre and Gauss-Chebyshev 
        integration.
    """
    f=open(fileName, 'r')
    listOfRoots=[]
    listOfWeights=[]
    for line in f: 
        numbers=line.split()
        listOfRoots.append(float(numbers[0]))
        listOfWeights.append(float(numbers[1]))
    f.close()
    n_q=len(listOfRoots)
    ChebPoly=ChebyshevPoly(n_z)
    ChebPoly.FindAllRoots()
    ChebPoly.ComputeWeights()
    solve=0
    if transf_q=='lin' and transf_z=='lin':
        for i in range(n_q):
            for j in range(n_z):
                solve+=ChebPoly.listOfWeights[j]*listOfWeights[i]*func(LinTranslator(a_q,b_q,listOfRoots[i]),LinTranslator(a_z,b_z,ChebPoly.listOfRoots[j]),p)
        solve*=(b_z-a_z)/2*(b_q-a_q)/2
    if transf_q=='log' and transf_z=='lin':
        A=-math.log(a_q*b_q)/math.log(b_q/a_q)
        B=2/math.log(b_q/a_q)
        for i in range(n_q):
            for j in range(n_z):
                solve+=ChebPoly.listOfWeights[j]*listOfWeights[i]*1/B*math.exp((listOfRoots[i]-A)/B)*func(LogTranslator(a_q,b_q,listOfRoots[i]),LinTranslator(a_z,b_z,ChebPoly.listOfRoots[j]),p)
    return solve
        
    