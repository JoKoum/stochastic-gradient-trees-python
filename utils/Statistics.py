import numpy as np

class Statistics:
    def __init__(self):
        self.MACHEP = 1.11022302462515654042E-16
        self.MAXLOG = 7.09782712893383996732E2
        self.MINLOG = -7.451332191019412076235E2
        self.MAXGAM = 171.624376956302725
        self.SQTPI = 2.50662827463100050242E0
        self.SQRTH = 7.07106781186547524401E-1
        self.LOGPI = 1.14472988584940017414

        self.big = 4.503599627370496E15
        self.biginv = 2.22044604925031308085E-16

        self.P0 = np.array([-5.99633501014107895267E1, 9.80010754185999661536E1, 
        -5.66762857469070293439E1, 1.39312609387279679503E1, 
        -1.23916583867381258016E0])
        self.Q0 = np.array([1.95448858338141759834E0, 4.67627912898881538453E0, 8.63602421390890590575E1, 
        -2.25462687854119370527E2, 2.00260212380060660359E2,
        -8.20372256168333339912E1, 1.59056225126211695515E1, 
        -1.18331621121330003142E0])

        self.P1 = np.array([4.05544892305962419923E0, 3.15251094599893866154E1, 5.71628192246421288162E1, 
        4.40805073893200834700E1, 1.46849561928858024014E1, 
        2.18663306850790267539E0, -1.40256079171354495875E-1,
        -3.50424626827848203418E-2, -8.57456785154685413611E-4])
        self.Q1 = np.array([1.57799883256466749731E1, 4.53907635128879210584E1, 4.13172038254672030440E1, 
        1.50425385692907503408E1, 2.50464946208309415979E0,
        -1.42182922854787788574E-1, -3.80806407691578277194E-2, 
        -9.33259480895457427372E-4])

        self.P2 = np.array([3.23774891776946035970E0, 
        6.91522889068984211695E0, 3.93881025292474443415E0,
        1.33303460815807542389E0, 2.01485389549179081538E-1,
        1.23716634817820021358E-2, 3.01581553508235416007E-4,
        2.65806974686737550832E-6, 6.23974539184983293730E-9])
        self.Q2 = np.array([6.02427039364742014255E0, 3.67983563856160859403E0, 
        1.37702099489081330271E0, 2.16236993594496635890E-1, 
        1.34204006088543189037E-2, 3.28014464682127739104E-4, 
        2.89247864745380683936E-6, 6.79019408009981274425E-9])
    
    @staticmethod
    def binomialStandardError(p, n):
        """
        Computes standard error for observed values of a binomial random variable.
        p the probability of success
        n the size of the sample
        Return the standard error
        """
        if n == 0:
            return 0
        
        return np.sqrt((p * (1 - p)) / n)
    
    @staticmethod
    def chiSquaredProbability(x, v):
        raise NotImplementedError

    def FProbability(self, F, df1, df2):
        return self.incompleteBeta(df2/2.0, df1/2.0, df2/(df2+df1*F))
    
    def incompleteBeta(self, aa, bb, xx):
        """
        Returns the Incomplete Beta Function evaluated from zero to xx.
        aa the alpha parameter of the beta distribution.
        bb the beta parameter of the beta distribution.
        xx the integration end point.
        """
        
        if (aa <= 0.0) or (bb <= 0.0):
            raise ArithmeticError("ibeta: Domain error!")
        
        if (xx <= 0.0) or (xx >= 1.0):
            if xx == 0.0:
                return 0.0
            if xx == 1.0:
                return 1.0
            raise ArithmeticError("ibeta: Domain error!")
        
        flag = False
        if (bb * xx <= 1.0) or (xx <= 0.95):
            t = self.powerSeries(aa, bb, xx)
            return t
        
        w = 1.0 - xx

        # Reverse a and b if x is greater than the mean.
        if xx > (aa / (aa + bb)):
            flag = True
            a = bb
            b = aa
            xc = xx
            x = w
        else:
            a = aa
            b = bb
            xc = w
            x = xx
        
        if flag and (b*x <= 1.0) and (x <= 0.95):
            t = self.powerSeries(a, b, x)
            if t <= self.MACHEP:
                t = 1.0 - self.MACHEP
            else:
                t = 1.0 - t
            
            return t
        
        # Choose expansion for better convergence.
        y = x * (a + b - 2.0) - (a - 1.0)
        if y < 0.0:
            w = self.incompleteBetaFraction1(a, b, x)
        else:
            w = self.incompleteBetaFraction2(a, b, x) / xc
        
        # Multiply w by the factor a b _ _ _ x (1-x) | (a+b) / ( a | (a) | (b) ) .

        y = a * np.log(x)
        t = b * np.log(xc)
        if (a+b < self.MAXGAM) and (np.abs(y) < self.MAXLOG) and (np.abs(t) < self.MAXLOG):
            t = np.power(xc, b)
            t *= np.power(x, a)
            t /= a
            t *= w
            t *= self.gamma(a + b) / (self.gamma(a) * self.gamma(b))
            if flag:
                if t <= self.MACHEP:
                    t = 1.0 - self.MACHEP
                else:
                    t = 1.0 - t

            return t
        
        # Resort to logarithms.
        y += t + self.lnGamma(a + b) - self.lnGamma(a) - self.lnGamma(b)
        y += np.log(w/a)
        if y < self.MINLOG:
            t = 0.0
        else:
            t = np.exp(y)
        
        if flag:
            if t <= self.MACHEP:
                t = 1.0 - self.MACHEP
            else:
                t = 1.0 - t
        return t
    
    def incompleteBetaFraction1(self, a, b, x):
        """
        Continued fraction expansion #1 for incomplete beta integral.
        """
        k1 = a
        k2 = a + b
        k3 = a
        k4 = a + 1.0
        k5 = 1.0
        k6 = b - 1.0
        k7 = k4
        k8 = a + 2.0

        pkm2 = 0.0
        qkm2 = 1.0
        pkm1 = 1.0
        qkm1 = 1.0
        ans = 1.0
        r = 1.0
        n = 0
        thresh = 3.0 * self.MACHEP

        while True:
            xk = -(x * k1 * k2) / (k3 * k4)
            pk = pkm1 + pkm2 * xk
            qk = qkm1 + qkm2 * xk

            pkm2 = pkm1
            pkm1 = pk
            qkm2 = qkm1
            qkm1 = qk
            
            xk = (x * k5 * k6) / (k7 * k8)
            pk = pkm1 + pkm2 * xk
            qk = qkm1 + qkm2 * xk
            pkm2 = pkm1
            pkm1 = pk
            qkm2 = qkm1
            qkm1 = qk
            
            if (qk != 0):
                r = pk / qk
            
            if (r != 0):
                t = np.abs((ans - r) / r)
                ans = r
            else:
                t = 1.0
            
            if (t < thresh):
                return ans
                
            k1 += 1.0
            k2 += 1.0
            k3 += 2.0
            k4 += 2.0
            k5 += 1.0
            k6 -= 1.0
            k7 += 2.0
            k8 += 2.0
            
            if ((np.abs(qk) + np.abs(pk)) > self.big):
                pkm2 *= self.biginv
                pkm1 *= self.biginv
                qkm2 *= self.biginv
                qkm1 *= self.biginv
            
            if ((np.abs(qk) < self.biginv) or (np.abs(pk) < self.biginv)):
                pkm2 *= self.big
                pkm1 *= self.big
                qkm2 *= self.big
                qkm1 *= self.big
      

            if n < 300:
                break

            n += 1
        
        return ans

    
    def incompleteBetaFraction2(self, a, b, x):
        """
        Continued fraction expansion #2 for incomplete beta integral.
        """
        k1 = a
        k2 = b - 1.0
        k3 = a
        k4 = a + 1.0
        k5 = 1.0
        k6 = a + b
        k7 = a + 1.0
        k8 = a + 2.0
        
        pkm2 = 0.0
        qkm2 = 1.0
        pkm1 = 1.0
        qkm1 = 1.0
        z = x / (1.0 - x)
        ans = 1.0
        r = 1.0
        n = 0
        thresh = 3.0 * self.MACHEP

        while True:
            xk = -(z * k1 * k2) / (k3 * k4)
            pk = pkm1 + pkm2 * xk
            qk = qkm1 + qkm2 * xk
            pkm2 = pkm1
            pkm1 = pk
            qkm2 = qkm1
            qkm1 = qk
            
            xk = (z * k5 * k6) / (k7 * k8)
            pk = pkm1 + pkm2 * xk
            qk = qkm1 + qkm2 * xk
            pkm2 = pkm1
            pkm1 = pk
            qkm2 = qkm1
            qkm1 = qk
            
            if (qk != 0):
                r = pk / qk
            
            if (r != 0):
                t = np.abs((ans - r) / r)
                ans = r
            else:
                t = 1.0
            
            if (t < thresh):
                return ans
            
            k1 += 1.0
            k2 -= 1.0
            k3 += 2.0
            k4 += 2.0
            k5 += 1.0
            k6 += 1.0
            k7 += 2.0
            k8 += 2.0
            
            if ((np.abs(qk) + np.abs(pk)) > self.big):
                pkm2 *= self.biginv
                pkm1 *= self.biginv
                qkm2 *= self.biginv
                qkm1 *= self.biginv
            
            if ((np.abs(qk) < self.biginv) or (np.abs(pk) < self.biginv)):
                pkm2 *= self.big
                pkm1 *= self.big
                qkm2 *= self.big
                qkm1 *= self.big
            
            

            if n  < 300:
                break
            
            n += 1
            
        return ans
        
    def gamma(self, x):
        P = np.array([1.60119522476751861407E-4, 1.19135147006586384913E-3,
        1.04213797561761569935E-2, 4.76367800457137231464E-2,
        2.07448227648435975150E-1, 4.94214826801497100753E-1,
        9.99999999999999996796E-1])
        Q = np.array([-2.31581873324120129819E-5, 5.39605580493303397842E-4,
        -4.45641913851797240494E-3, 1.18139785222060435552E-2,
        3.58236398605498653373E-2, -2.34591795718243348568E-1,
        7.14304917030273074085E-2, 1.00000000000000000320E0])

        q = np.abs(x)

        if q > 33.0:
            if x < 0.0:
                p = np.floor(q)
                if p == q:
                    raise ArithmeticError("gamma: overflow")
                z = q - p
                if z > 0.5:
                    p += 1.0
                    z = q - p
                z = q * np.sin(np.pi * z)
                if z == 0.0:
                    raise ArithmeticError("gamma: overflow")
                z = np.abs(z)
                z = np.pi / (z * self.stirlingFormula(q))

                return -z
            else:
                return self.stirlingFormula(x)
            
        z = 1.0
        while x >= 3.0:
            x -= 1.0
            z *= x
        
        while x < 0.0:
            if x == 0.0:
                raise ArithmeticError("gamma: singular")
            elif x > -1.E-9:
                return (z / ((1.0 + 0.5772156649015329 * x) * x))
            z /= x
            x += 1.0
        
        while x < 2.0:
            if x == 0.0:
                raise ArithmeticError("gamma: singular")
            elif x < 1.e-9:
                return (z / ((1.0 + 0.5772156649015329 * x) * x))
            z /= x
            x += 1.0
        
        if (x == 2.0) or (x == 3.0):
            return z
        
        x -= 2.0
        p = self.polevl(x, P, 6)
        q = self.polevl(x, Q, 7)
        return z * p / q
    
    @staticmethod
    def polevl(x, coef, N):
        """
        Evaluates the given polynomial of degree <tt>N</tt> at <tt>x</tt>. 
        
        y  =  C  + C x + C x  +...+ C x
              0    1     2          N
        
        Coefficients are stored in reverse order:
        
        coef[0] = C  , ..., coef[N] = C  .
                   N                   0
        
        In the interest of speed, there are no checks for out of bounds arithmetic.
        
        x argument to the polynomial.
        coef the coefficients of the polynomial.
        N the degree of the polynomial.
        """
        ans = coef[0]

        for i in range(N):
            ans = ans * x + coef[i]
        
        return ans


    def stirlingFormula(self, x):
        """
        Returns the Gamma function computed by Stirling's formula. The polynomial
        STIR is valid for 33 <= x <= 172.
        """
        STIR = np.array([7.87311395793093628397E-4, -2.29549961613378126380E-4,
        -2.68132617805781232825E-3, 3.47222221605458667310E-3,
        8.33333333333482257126E-2])
        MAXSTIR = 143.01608

        w = 1.0 / x
        y = np.exp(x)

        w = 1.0 + w * self.polevl(w, STIR, 4)

        if x > MAXSTIR:
            # Avoid overflow in math.pow()
            v = np.power(x, 0.5 * x - 0.25)
            y = v * (v / y)
        else:
            y = np.power(x, x - 0.5) / y
        
        y = self.SQTPI * y * w
        return y
    
    def lnGamma(self, x):
        """
        Returns natural logarithm of gamma function.
        x the value
        return natural logarithm of gamma function
        """
        A = np.array([8.11614167470508450300E-4, -5.95061904284301438324E-4,
        7.93650340457716943945E-4, -2.77777777730099687205E-3,
        8.33333333333331927722E-2])
        B = np.array([-1.37825152569120859100E3, -3.88016315134637840924E4,
        -3.31612992738871184744E5, -1.16237097492762307383E6,
        -1.72173700820839662146E6, -8.53555664245765465627E5])
        C = np.array([3.51815701436523470549E2, -1.70642106651881159223E4,
        -2.20528590553854454839E5, -1.13933444367982507207E6,
        -2.53252307177582951285E6, -2.01889141433532773231E6])

        if x < -34.0:
            q = -x
            w = self.lnGamma(q)
            p = np.floor(q)
            if (p == q):
                raise ArithmeticError("lnGamma: Overflow")
            
            z = q - p
            if (z > 0.5):
                p += 1.0
                z = p - q
            
            z = q * np.sin(np.pi * z)
            if (z == 0.0):
                raise ArithmeticError("lnGamma: Overflow")
            z = self.LOGPI - np.log(z) - w
            return z
        
        if (x < 13.0):
            z = 1.0
            while (x >= 3.0):
                x -= 1.0
                z *= x
            while (x < 2.0):
                if (x == 0.0):
                    raise ArithmeticError("lnGamma: Overflow")
                z /= x
                x += 1.0
            
            if (z < 0.0):
                z = -z
            if (x == 2.0):
                return np.log(z)
            x -= 2.0
            p = x * self.polevl(x, B, 5) / self.p1evl(x, C, 6)
            return (np.log(z) + p)
        
        if (x > 2.556348e305):
            raise ArithmeticError("lnGamma: Overflow")
        
        q = (x - 0.5) * np.log(x) - x + 0.91893853320467274178
        
        if (x > 1.0e8):
            return q
        
        p = 1.0 / (x * x)
        
        if (x >= 1000.0):
            q += ((7.9365079365079365079365e-4 * p - 2.7777777777777777777778e-3) * p + 0.0833333333333333333333) / x
        else:
            q += self.polevl(p, A, 4) / x
        
        return q
    
    @staticmethod
    def p1evl(x, coef, N):
        """
        The function <tt>p1evl()</tt> assumes that <tt>coef[N] = 1.0</tt> and is
        omitted from the array. Its calling arguments are otherwise the same as
        <tt>polevl()</tt>.
        
        In the interest of speed, there are no checks for out of bounds arithmetic.
        
        x argument to the polynomial.
        coef the coefficients of the polynomial.
        N the degree of the polynomial.
        """

        ans = x + coef[0]

        for i in range(N):
            ans = ans * x + coef[i]
        
        return ans

    def powerSeries(self, a, b, x):
        """
        Power series for incomplete beta integral. Use when b*x is small and x not
        too close to 1.
        """

        ai = 1.0 / a
        u = (1.0 - b) * x
        v = u / (a + 1.0)
        t1 = v
        t = u
        n = 2.0
        s = 0.0
        z = self.MACHEP * ai
        while np.abs(v) > z:
            u = (n - b) * x / n
            t *= u
            v = t / (a + n)
            s += v
            n += 1.0
        s += t1
        s += ai

        u = a * np.log(x)
        if ((a + b) < self.MAXGAM) and (np.abs(u) < self.MAXLOG):
            t = self.gamma(a + b) / (self.gamma(a) * self.gamma(b))
            s = s * t * np.power(x, a)
        else:
            t = self.lnGamma(a + b) - self.lnGamma(a) - self.lnGamma(b) + u + np.log(s)
            if (t < self.MINLOG):
                s = 0.0
            else:
                s = np.exp(t)
        return s

if __name__ == "__main__":
    print('Binomial standard error (0.5, 100): {}'.format(Statistics().binomialStandardError(0.5, 100)))
    print('lnGamma(6): {}'.format(Statistics().lnGamma(6)))
    print('F probability (5.1922, 4, 5): {}'.format(Statistics().FProbability(5.1922, 4, 5)))