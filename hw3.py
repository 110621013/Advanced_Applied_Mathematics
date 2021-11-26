# 要求f=ax3+bx2+cx+d的四項系數，n-1個區間n個點(ni, i=0~n-1)的話，4(n-1)個要推的未知數：
# 1.兩段函數值在中間節點處相等(2n-4)(aixi3+bixi2+cixi+di = aixi+13+bixi+12+cixi+1+di, i=range(n-1))
# 2.頭尾函數通過端點(2)
# 3.中間節點處一階導數相同(n-2)
# 4.中間節點處二階導數相同(n-2)
# 5.端點處二階導數=0(2)(可看情況換)
# 看樹熙課本的蘇都扣
'''
用自然邊界條件
https://zhuanlan.zhihu.com/p/62860859
1，计算步长 hi = xi+1 -xi
2，将数据节点和指定的首位端点条件带入矩阵方程
3，解矩阵方程，求得二次微分值mi 。该矩阵为三对角矩阵，常见解法为高斯消元法，可以对系数矩阵进行LU分解，分解为单位下三角矩阵和上三角矩阵。即 B=Ax=(LU)x=L(Ux)=Ly
4，计算样条曲线的系数：
ai = (mi+1-mi)6/hi
bi = mi/2
ci = (yi+1-yi)/hi - hi*mi/2 - hi(mi+1-mi)/6
di = yi
5，在每个区间中，创建方程
gi(x) = di + ci(x-xi) + bi(x-xi)**2 + ai(x-xi)**3
'''

import numpy as np

# give arr -> LU, check ok
def LU_decomposition(arr):
    assert len(arr.shape)==2
    assert arr.shape[0]==arr.shape[1]
    n = arr.shape[0]

    L = np.full_like(arr, 0.0, dtype=np.double)
    for i in range(n):
        L[i][i] = 1.0

    for col in range(n-1):
        for row in range(col+1, n):
            L[row][col] = arr[row][col] / arr[col][col]
            arr[row] = arr[row] - L[row][col]*arr[col]
    return L, arr
# use LU_decomposition to solve linear eq, check ok
def LU_solve_eq(A, B):
    n = B.shape[0]
    L, U = LU_decomposition(A)
    D, m = np.zeros_like(B), np.zeros_like(B)

    # LD = B
    for i in range(n):
        D[i] = B[i]
        for j in range(i+1, n):
            B[j] -= L[j][i] * D[i]
    
    # Um = D
    for i in range(n-1, -1, -1):
        m[i] = D[i] / U[i][i]
        for j in range(i-1, -1, -1):
            D[j] -= U[j][i] * m[i]
    return m


def get_cubic_spline_coefficient(x, y):
    n = len(x)-1 # intervel nums
    h = [x[i+1]-x[i] for i in range(n)]
    A = np.zeros((n+1, n+1))
    B = np.zeros((n+1))

    # natural condition
    A[0][0] = 1
    A[n][n] = 1
    for i in range(1, n):
        A[i][i] = 2*(h[i-1]+h[i])
        A[i][i-1] = h[i-1]
        A[i][i+1] = h[i]
        B[i] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
    m = LU_solve_eq(A, B)

    coefficient = []
    for i in range(n):
        coef = []
        coef.append((m[i+1]-m[i])/6/h[i])
        coef.append(m[i]/2)
        coef.append((y[i+1]-y[i])/h[i] - h[i]*m[i]/2 - h[i]*(m[i+1]-m[i])/6)
        coef.append(y[i])
        coefficient.append(coef)
    return coefficient

def cubic_spline_plot(x, y, coefficient):
    import matplotlib.pyplot as plt
    n = len(x)-1 # intervel nums
    
    plt.scatter(x, y)
    for i in range(n):
        plot_x = np.linspace(x[i], x[i+1], 10)
        plot_x_to_y = np.linspace(0.0, x[i+1]-x[i], 10)
        plot_y = coefficient[i][0]*plot_x_to_y**3 + coefficient[i][1]*plot_x_to_y**2 + coefficient[i][2]*plot_x_to_y**1 + coefficient[i][3]*plot_x_to_y**0
        plt.plot(plot_x, plot_y, label='{}~{}'.format(str(x[i]), str(x[i+1])))
    plt.legend()
    plt.title('cubic_spline')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plt.close()

def parsing_vs_approximate():
    import matplotlib.pyplot as plt
    x = np.linspace(-10.0, 10.0, 100)
    parsing = np.sin(x)/np.sin(1) - x
    approximate = (np.power(x, 2)-np.power(x, 3))/(np.power(x, 2)-x+2)
    plt.plot(x, parsing, label='parsing')
    plt.plot(x, approximate, label='approximate')
    plt.legend()
    plt.title('parsing vs approximate')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()
    plt.close()

    delta = parsing-approximate
    plt.plot(x, delta, label='delta')
    plt.legend()
    plt.title('parsing - approximate')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()
    plt.close()

    x = np.linspace(-2.0, 2.0, 100)
    parsing = np.sin(x)/np.sin(1) - x
    approximate = (np.power(x, 2)-np.power(x, 3))/(np.power(x, 2)-x+2)
    plt.plot(x, parsing, label='parsing')
    plt.plot(x, approximate, label='approximate')
    plt.legend()
    plt.title('parsing vs approximate')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()
    plt.close()

    delta = parsing-approximate
    plt.plot(x, delta, label='delta')
    plt.legend()
    plt.title('parsing - approximate')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()
    plt.close()

    x = np.array([0.25, 0.5, 0.75])
    parsing = np.sin(x)/np.sin(1) - x
    approximate = (np.power(x, 2)-np.power(x, 3))/(np.power(x, 2)-x+2)
    print(parsing, approximate)


if __name__ == '__main__':
    '''
    print('gogo')
    x = [0, 1, 2, 3, 4]
    y = [-7, -5, 2, 18, 40]
    coefficient = get_cubic_spline_coefficient(x, y)
    print('coefficient:', coefficient)
    cubic_spline_plot(x, y, coefficient)

    x = [0, 1, 3, 6, 10]
    y = [20, -8, 10, 2, -7]
    coefficient = get_cubic_spline_coefficient(x, y)
    print('coefficient:', coefficient)
    cubic_spline_plot(x, y, coefficient)
    '''
    parsing_vs_approximate()