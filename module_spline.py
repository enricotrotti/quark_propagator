##########################################################################################
#####################                    Authors                     #####################
################     Chevalier Cyrille, Cimino Lorenzo, Trotti Enrico     ################
##########################################################################################

##########################################################################################
#####################           Solution of the quark DSE            #####################
##########################################################################################


#----------General functions

import math
import numpy
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def TridiagSolver(triMatrix,R):
    """
    Input:
        R : a list of float.
            Let us call L its length
        triMatrix : a list of three lists.
            These three list must be three list of floats. The length of the 
            second list must be N. The length of the first and the third must 
            be N-1.
            TriMatrix can also be a list of lists representing a tridiagonal 
            matrix (with the convention that the sublists contain the lines of
            the matrix).
    Output:
        The list of float [z0,z1,z2,...] such that :
         |B0  C0  0   0  ...| |z0 | = |R0 |
         |A0  B1  C1  0  ...| |z1 |   |R1 |
         |0   A1  B2  C2 ...| |z2 |   |R2 |
         |... ... ... ...   | |...|   |...|
    """
    try:
        [A,B,C]=triMatrix
        if len(A)!=len(B)-1 or len(C)!=len(B)-1:
            raise Exception()
        N=len(B)
        beta=[B[0]]
        rho=[R[0]]
        for i in range(1,N):
            beta_i=B[i]-A[i-1]/beta[i-1]*C[i-1]
            beta.append(beta_i)
            rho_i=R[i]-A[i-1]/beta[i-1]*rho[i-1]
            rho.append(rho_i)
        solve=(N-1)*[0]
        solve.append(rho[-1]/beta[-1])
        for i in range(1,N):
            z=(rho[N-1-i]-C[N-1-i]*solve[N-1-i+1])/beta[N-1-i]
            solve[N-1-i]=z
        return solve
    except:
        A=[]
        B=[triMatrix[0][0]]
        C=[]
        for i in range(1,len(triMatrix)):
            B.append(triMatrix[i][i])
            A.append(triMatrix[i][i-1])
            C.append(triMatrix[i-1][i])
        return TridiagSolver([A,B,C],R)
    
""" Example :
triMatrix=[[3,4,0],[6,10,1],[0,10,9]]
R=[11,29,47]
print(TridiagSolver(triMatrix, R))"""


#------------------------ Make a grid of datas -------------------------------#
    
class Grid:

    """
    This class is build to contain all the information about the function that
    we want to interpolate (basically, the set {(x_i,f(x_i))}_i). Let us call 
    func this function.
    """
    def __init__(self,N):
        """
        Input:
            N : a non-zero integer.
        Output:
            It creates an object of class Grid. This object has four
            attributes:
                xSet which is (for now) a list of N+1 ordered floats 
                homogeneously spread along the interval [-1,1],
                    this list will be modified to contain the x points where 
                    func is known;
                initGrid which the same list of N+1 ordered floats homogeneously  
                spread along the interval [-1,1],
                    this list will be used to adapt xSet and must never be 
                    changed;
                fSet which is (for now) an empty list,
                    this list will be fulled with the evalution of func on the 
                    points of self.xSet.
                fsecSet which is (for now) an empty list,
                    this list will be fulled with the imposed second derivative
                    of the interpolated function on the points of self.xSet 
                    (only usefull for cubic spline).
        """
        solve=[]
        index = 0
        while index <= N:
            z = -1 + 2*index/N
            solve.append(z)
            index += 1
        self.xSet = solve
        self.initGrid = solve
        self.fSet = []
        self.fsecSet = []
    
    def __str__(self):
        """ 
        Input:
            Nothing.
        Output:
            It displays the grid in the console. Depending on if self.fSet is 
            empty or not, only self.xSet is displayed or both self.xSet and 
            self.fSet are displayed (with the structure (x_i,f(x_i))).
        """
        solve=''
        if self.fSet==[]:
            solve+=str(self.xSet)+'\n'
            solve+='- fSet is still empty -'
        else :
            for i in range(len(self.xSet)):
                solve+='('+str(round(self.xSet[i],4))+','+str(round(self.fSet[i],4))+') '
        return solve
    
    def mapping_0(self,a,b):
        """
        Input:
            a a float.
            b : a float bigger than a.
        Output:
            It maps self.initGrid from [-1,1] to [a,b] in a linear way and puts 
            the result inside self.xSet (previous value for self.xSet is 
            forgotten).This new value for self.xSet is also returned.
        """
        solve=[]
        for x in self.initGrid:
            solve.append(((b-a)*x+a+b)/2)
        self.xSet=solve
        return solve
    
    def mapping_1(self):
        """
        Input:
            Nothing.
        Output:
            It maps self.initGrid from [-1,1] to [0,+inf[ and puts the result 
            inside self.xSet (previous value for self.xSet is forgotten).This
            new value for self.xSet is also returned.
        """
        solve=[]
        for x in self.initGrid:
            if x!=1:
                solve.append((1+x)/(1-x))
            if x==1:
                solve.append((1+x)/(1-x+0.001))
        self.xSet=solve
        return solve
    
    def mapping_2(self,a,b):
        """
        Input:
            Nothing.
        Output:
            It maps self.initGrid from [-1,1] to [a,b] and puts the result 
            inside self.xSet (previous value for self.xSet is forgotten).This
            new value for self.xSet is also returned.
        """
        solve=[]
        A=-math.log(a*b)/math.log(b/a)
        B=2/math.log(b/a)
        for x in self.initGrid:
            solve.append(math.exp((x-A)/B))
        self.xSet=solve
        return solve
    
    def AddData(self,func):
        """ 
        Input:
            func : a function of one float variable.
                This function is the one that we want to interpolate.
        Output:
            It replaces self.fSet by another set obtained by the evaluation of 
            func on each element inside self.xSet. In addition, self.fsecSet is
        """
        solve=[]
        for x in self.xSet:
            f=func(x)
            solve.append(f)
            # print("{:<15} {:<15}".format('p='+str(numpy.round(x,5)),'f='+str(numpy.round(f,5))))
        self.fSet=solve
        self.fsecSet=[]
        return solve
    
    def Localise(self,x):
        """
        Input:
            x : a float.
        Output:
            It returns an integer that describe the position of x inside the list 
            self.xSet. If x is between the ith and the i+1th element of self.xSet,
            it returns i+1 (i=0 means that x is before each element of self.xSet, 
            i=len(self.xSet) means that x is over each element of self.xSet).
        """
        i=0
        while i < len(self.xSet) and self.xSet[i]<x :
            i+=1
        return i
    
    def ComputeDeriv(self):
        """
        Input:
            Nothing
        Output:
            It replaces self.fsecSet by a list that contains the second 
            derivative that we will require to the futur interpolated function
            (there is a condition at each points of self.xSet). This 
            new value for self.fsecSet is also returned (previous value for 
            self.fsecSet is forgotten).
            This method is only usefull for cubic spline.
        """
        A=[]
        B=[1]
        C=[0]
        R=[0]
        for i in range(1,len(self.xSet)-1):
            A.append((self.xSet[i]-self.xSet[i-1])/6)
            B.append((self.xSet[i+1]-self.xSet[i-1])/3)
            C.append((self.xSet[i+1]-self.xSet[i])/6)
            R.append((self.fSet[i+1]-self.fSet[i])/(self.xSet[i+1]-self.xSet[i])-(self.fSet[i]-self.fSet[i-1])/(self.xSet[i]-self.xSet[i-1]))
        A.append(0)
        B.append(1)
        R.append(0)
        fsec=TridiagSolver([A,B,C], R)
        self.fsecSet=fsec
        return fsec

"""Example : 
grid=Grid(200)
grid.mapping_2(10**(-3), 10**3)
print(grid.xSet)"""

#--------------------- Linear spline interpolation ---------------------------#


def LinearInterpolation(x,point1,point2):
    """ 
    Input:
        x : a float.
        point1 and point2 : two different doublets of floats that refers to
        points on the plane.
    Output:
        It returns the evaluation at x of the linear function that goes through  
        the points point1 and point2.
    """
    (x1,f1)=point1
    (x2,f2)=point2
    a=(f1-f2)/(x1-x2)
    b=f2-a*x2
    return a*x + b

def SplineLinInt(x,grid):
    """
    Input:
        x : a float
        grid : an object of class Grid.
    Output:
        It returns the evaluation at x of the linear sline interpolation built 
        from grid.
    """
    i=grid.Localise(x)
    if i == len(grid.xSet) or i==0:
        return None
    x1=grid.xSet[i-1]
    x2=grid.xSet[i]
    f1=grid.fSet[i-1]
    f2=grid.fSet[i]
    solve=LinearInterpolation(x,(x1,f1),(x2,f2))
    return solve

def LinearSpline_Graph(xrange,func,N):
    """ 
    Input:
        xrange : a list-like of floats.
            It is on this list we will plot the function and its interpolation.
        func : a function of one float variable.
            This is the function that we will interploate.
        N : an integer.
            It corresponds to  the number of points used to interpolate func.
    Output:
        None but it displays a supperposed graph of func and its linear
        interpolation.
    """
    # Setting of the interpolation.
    grid = Grid(N)
    grid.mapping_1()
    grid.AddData(func)
    interFunc = lambda x : SplineLinInt(x,grid)
    # Defining datas for the plot.
    funcPlot=[]
    intFuncPlot=[]
    for x in xrange:
        funcPlot.append(func(x))
        intFuncPlot.append(interFunc(x))
    # Building of the plot.
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    function, = plt.plot(xrange,funcPlot)
    intFunction, = plt.plot(xrange,intFuncPlot)
    # Showing the graph.
    plt.show()

""" Example : 
func = lambda x : 0.2/(x+0.5)+0.002
xrange = numpy.arange(0,10,0.0001)
LinearSpline_Graph(xrange,func,10)"""


#--------------------- Cubic spline interpolation ---------------------------#


def CubicInterpolation(x,point1,point2):
    """
    Input:
        x: a float
        point1 and point2 : two different triplet of floats.
            Each of these triplets specify a point of the plane where the cubic 
            function must go through and specify its seconde derivative at this 
            point.
    Output:
        It returns the evaluation at x of the cubic function that satisfies the
        conditions imposed by point1 and point2.
    """
    (x1,f1,fsec1)=point1
    (x2,f2,fsec2)=point2
    A=(x2-x)/(x2-x1)
    B=1-A
    C=1/6*(A**3-A)*(x2-x1)**2
    D=1/6*(B**3-B)*(x2-x1)**2
    return A*f1+B*f2+C*fsec1+D*fsec2

def SplineCubInt(x,grid):
    """
    Input:
        x : a float
        grid : an object of class Grid.
    Output:
        It returns the evaluation at x of the cubic sline interpolation built 
        from grid.
    """
    try :
        if x==grid.xSet[0]:
            return grid.fSet[0]
        i=grid.Localise(x)
        if i==0 or i == len(grid.xSet):
            return None
        fsec = grid.fsecSet
        x1,x2 = grid.xSet[i-1],grid.xSet[i]
        f1,f2 = grid.fSet[i-1],grid.fSet[i]
        fsec1,fsec2 = fsec[i-1],fsec[i]
        solve=CubicInterpolation(x,(x1,f1,fsec1),(x2,f2,fsec2))
        return solve
    except :
        grid.ComputeDeriv()
        return SplineCubInt(x,grid)

def CubicSlpine_Graph(xrange,func,N):
    """ 
    Input:
        xrange : a list-like of floats.
            It is on this list we will plot the function and its interpolation.
        func : a function of one float variable.
            This is the function that we will interploate.
        N : an integer.
            It corresponds to  the number of points used to interpolate func.
    Output:
        None but it displays a supperposed graph of func and its cubic spline
        interpolation.
    """
    # Setting of the interpolation.
    grid = Grid(N)
    grid.mapping_1()
    grid.AddData(func)
    interFunc = lambda x : SplineCubInt(x,grid)
    # Defining datas for the plot.
    funcPlot=[]
    intFuncPlot=[]
    for x in xrange:
        funcPlot.append(func(x))
        intFuncPlot.append(interFunc(x))
    # Building of the plot.
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    function, = plt.plot(xrange,funcPlot)
    intFunction, = plt.plot(xrange,intFuncPlot)
    # Showing the graph.
    plt.show()

""" Example : 
func = lambda x : 0.2/(x+0.5)+0.002
xrange = numpy.arange(0,10,0.1)
CubicSlpine_Graph(xrange,func,10)"""


#------------------------ 2D linear interpolation ----------------------------#


class Grid2D:
    """ This is a generalisation of the class Grid for two dimensional 
    functions. Let us call func this 2D function.
    """
    def __init__(self,Nx,Ny):
        """ 
        Input:
            Nx : an integer.
            Ny : an integer.
        Output:
            It creates object of class Grid2D. This object has two attributes:
                xGrid is the grid used in the x direction;
                yGrid is the grid used in the y direction;
                fSet which is (for now) an empty list,
                    this list will be fulled with the evalution of func on the 
                    points of the two dimensional grid.
        """
        self.xGrid=Grid(Nx)
        self.yGrid=Grid(Ny)
        self.fSet=[]
    
    def Mapping(self, ax, bx, ay, by, transfx='lin',transfy='lin'):
        """ 
        Input : 
            
        Output :
            
        """
        if transfx=='lin':
            self.xGrid.mapping_0(ax, bx)
        if transfx=='log':
            self.xGrid.mapping_2(ax, bx)
        if transfy=='lin':
            self.yGrid.mapping_0(ay, by)
        if transfy=='log':
            self.yGrid.mapping_2(ay, by)
            
    def AddData(self,func):
        """ 
        Input : 
            
        Output :
            
        """
        totalSolve=[]
        for x in self.xGrid.xSet :
            solve=[]
            for y in self.yGrid.xSet :
                solve.append(func(x,y))
            totalSolve.append(solve)
        self.fSet=totalSolve
    
    def Localise(self,x,y):
        """ 
        Input : 
            
        Output :
            
        """
        i=self.xGrid.Localise(x)
        j=self.yGrid.Localise(y)
        return (i,j)
    

def SplineLinInt2D(grid,x,y):
    """ 
    Input : 
        
    Output :
        
    """
    (i,j) = grid.Localise(x,y)
    if i == len(grid.xGrid.xSet) or i == 0:
        return 0
    if j == len(grid.yGrid.xSet) or j == 0:
        return 0
    xi = grid.xGrid.xSet[i-1]
    xi1 = grid.xGrid.xSet[i]
    yj = grid.yGrid.xSet[j-1]
    yj1 = grid.yGrid.xSet[j]
    fij = grid.fSet[i-1][j-1]
    fi1j = grid.fSet[i][j-1]
    fij1 = grid.fSet[i-1][j]
    fi1j1 = grid.fSet[i][j]
    t = (x-xi)/(xi1-xi)
    u = (y-yj)/(yj1-yj)
    solve = (1-t)*(1-u)*fij + (t)*(1-u)*fi1j + (t)*(u)*fi1j1 + (1-t)*(u)*fij1
    return solve


def LinSlpine2D_Graph(xrange,yrange,func,Nx,Ny):
    """ 
    Input:
        xrange : a list-like of floats.
            It is on this list we will plot the function and its interpolation.
        func : a function of one float variable.
            This is the function that we will interploate.
        N : an integer.
            It corresponds to  the number of points used to interpolate func.
    Output:
        None but it displays a supperposed graph of func and its cubic spline
        interpolation.
    """
    # Setting of the interpolation.
    grid2D = Grid2D(Nx,Ny)
    grid2D.AddData(func)
    interFunc = lambda x,y : SplineLinInt2D(grid2D,x,y)
    # Defining datas for the plot.
    intFuncPlot = []
    xSet = []
    ySet = []
    for x in xrange:
        for y in yrange:
            intFuncPlot.append(interFunc(x,y))
            xSet.append(x)
            ySet.append(y)
    # Building of the plot.
    fig = plt.figure()
    Axes3D = fig.add_subplot(projection='3d')
    Axes3D.scatter(xSet,ySet,intFuncPlot)
    # Showing the graph.
    plt.show()


""" Example : 
func = lambda x,y : - 1/2*16*x**2 - 2*16*x*y + 1/2*16*y**2 + 1/4*4**4*x**4
xrange = numpy.arange(-0.99,0.99,0.1)
yrange = numpy.arange(-0.99,0.99,0.1)
LinSlpine2D_Graph(xrange,yrange,func,10,10)"""

#------------------ Interpolation : graphs with slider -----------------------#


"""
# Defining parameters of the interpolations.
N=15
func = lambda x : math.sin(x)
xrange = numpy.arange(-2,12,0.0001)
# Setting of the interpolations.
grid = Grid(N)
grid.mapping_0(0,10)
grid.AddData(func)
interFuncLin = lambda x : SplineLinInt(x,grid)
interFuncCub = lambda x : SplineCubInt(x,grid)
# Defining datas for the plot.
funcPlot=[]
linIntFuncPlot=[]
cubIntFuncPlot=[]
for x in xrange:
    funcPlot.append(func(x))
    linIntFuncPlot.append(interFuncLin(x))
    cubIntFuncPlot.append(interFuncCub(x))
# Building of the plot.
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
function, = plt.plot(xrange,funcPlot)
linIntFunction, = plt.plot(xrange,linIntFuncPlot)
cubIntFunction, =  plt.plot(xrange,cubIntFuncPlot)
ax.set_yticklabels([])
axecc = plt.axes([0.25, 0.1, 0.65, 0.03])
# Building the slider.
N_slider = Slider(ax=axecc,label='N',valmin=3,valmax=N,valstep=1,valinit=N)
def updateplot(val):
    grid = Grid(N_slider.val)
    grid.mapping_0(0,10)
    grid.AddData(func)
    interFuncLin = lambda x : SplineLinInt(x,grid)
    interFuncCub = lambda x : SplineCubInt(x,grid)
    funcPlot=[]
    linIntFuncPlot=[]
    cubIntFuncPlot=[]
    for x in xrange:
        funcPlot.append(func(x))
        linIntFuncPlot.append(interFuncLin(x))
        cubIntFuncPlot.append(interFuncCub(x))
    function.set_ydata(funcPlot)
    linIntFunction.set_ydata(linIntFuncPlot)
    cubIntFunction.set_ydata(cubIntFuncPlot)
    fig.canvas.draw_idle()
N_slider.on_changed(updateplot)
# Showing the graph.
plt.show()
"""