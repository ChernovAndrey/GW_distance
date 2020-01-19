"""
изоморфные графы; два треугольнка
"""
import numpy as np
from scipy.optimize import linprog


def se(a, b):
    return (a - b) ** 2


class GW_discrepancy(object):
    def __init__(self, Dx, Dy, Tx, Ty, T_init, loss=se, eps=1e-5, flag_print=False):
        """
        :param Dx: distance matrix for first graph
        :param Dy: distance matrix for second graph
        :param T_init: init value for measure X \times Y
        :param Tx: measure on X space
        :param Ty: measure on Y space
        :param loss: loss
        :param eps: in func optimize upper bound for difference between prev GW distance  and current
        :param flag_print: print in console log information
        """

        self.flag_print = flag_print
        self.eps = eps
        self.Tx = Tx
        self.Ty = Ty
        self.loss = loss
        self.Dy = Dy
        self.Dx = Dx
        self.T = T_init
        self.size_x = Dx.shape[0]
        self.size_y = Dy.shape[0]

        self.U = T_init.flatten() # vectorized T
        self.G = self._calculate_G()

        self.Sx = self._calculate_eccentricity(self.Tx, self.Dx)  # eccentricity for X space
        self.Sy = self._calculate_eccentricity(self.Ty, self.Dy)  # eccentricity for Y space

        self.A_eq, self.b_eq = self._bound_equations()


    def _calculate_G(self):
        k = 0
        G = np.zeros(shape=(self.size_x * self.size_y, self.size_x * self.size_y))
        for i0 in range(self.size_x):
            for i in range(self.size_y):
                G[k] = self.loss(np.tile(self.Dy[i], self.size_x), np.repeat(self.Dx[i0], self.size_y))
                k += 1
        return 0.5 * np.sqrt(G)

    def _calculate_eccentricity(self, T, D):
        """
        :param T: measure on space
        :param D: distance in spave
        :return: eccentricity for space
        """
        return T @ (D.T ** 2) / 2  # transpose necessary for oriented graphs

    def _bound_equations(self):
        """

        :return: A_eq - matrix boundary and b_eq - vector
        """
        A_eq = np.zeros(shape=(self.size_x + self.size_y, self.size_x * self.size_y))
        b_eq = np.zeros(self.size_x + self.size_y)
        for i in range(self.size_x):
            A_eq[i, np.arange(self.size_y * i, self.size_y * (i + 1))] = 1  # boundaries by string in T
            b_eq[i] = self.Tx[i]

        for i in range(self.size_y):
            if self.flag_print:
                print(np.arange(i, self.size_x * self.size_y, self.size_y))
            A_eq[i + self.size_x, np.arange(i, self.size_x *
                                            self.size_y, self.size_y)] = 1  # boundaries by columns in T
            b_eq[i + self.size_x] = self.Ty[i]

        if self.flag_print:
            print('A_eq:')
            print(A_eq)  # low rank matrix
            print('rank matrix A:', A_eq)
            print('b_eq:')
            print(b_eq)
        return A_eq, b_eq

    def _flb(self):
        c = np.abs(np.tile(self.Sy, self.size_x) - np.repeat(self.Sx, self.size_y)) / 2.0  # cartesian product
        self.U = np.array(list(linprog(c, A_eq=self.A_eq, b_eq=self.b_eq).values())[0])  # bounds >= 0 ?
        self.T = self.U.reshape(self.size_x, self.size_y)
        if self.flag_print:
            print('T after flb: ', self.U)

    # def calculate_slow(self): # deprecated
    #     res = 0.0
    #     # L = self.loss(self.Dx, self.Dy)
    #     for i0 in range(self.size_x):
    #         for j0 in range(self.size_x):
    #             for i1 in range(self.size_y):
    #                 for j1 in range(self.size_y):
    #                     res += self.loss(self.Dx[i0, j0], self.Dy[i1, j1]) * self.T[i0, i1] * self.T[j0, j1]
    #     return res

    def optimize(self):
        self._flb()
        cur_GW = self.calculate()
        prev_GW = self.calculate() - 2*self.eps
        while np.abs(cur_GW - prev_GW) > self.eps:
            self.U = np.array(list(linprog(self.G @ self.U, A_eq=self.A_eq, b_eq=self.b_eq).values())[0])
            self.T = self.U.reshape(self.size_x, self.size_y)
            if self.flag_print:
                print('T:')
                print(self.T)
                print('GW distance: ', cur_GW)
            prev_GW = cur_GW
            cur_GW = self.calculate()

    def calculate(self): # calculate GW distance¬
        return self.U.T @ self.G @ self.U


def ex1():  # two  triangle graph
    n = 3
    k = n - 1
    Dx = np.array([[0, 1 / 2, 1 / 2], [1 / 2, 0, 1 / 2], [1 / 2, 1 / 2, 0]])
    Dy = np.array([[0, 1], [1, 0]])
    # Dy = Dx
    # T = np.array([[0.5, 0.5, 0], [0, 0.5, 0.5]]).T
    # Tx = np.ones(n) / n
    Tx = np.array([1, 0, 0])
    # Ty = np.ones(k) / k
    Ty = np.array([0, 1])
    T = np.array([[0.5, 0.5], [0, 0.5], [0.5, 0.0]])
    # T = np.ones(shape=(n, k)) / (n * k)
    print('T:')
    print(T)
    gw = GW_discrepancy(Dx, Dy, Tx, Ty, T, se)
    print('GW distance before opt:', gw.calculate())
    print(gw.optimize())
    print('GW distance after opt:', gw.calculate())


ex1()
