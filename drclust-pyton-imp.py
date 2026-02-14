import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

class DrClust:
    """Python implementation of drclust library algorithms"""
    
    def __init__(self):
        pass
    
    # ==================== PREPROCESSING FUNCTIONS ====================
    
    def zscore(self, X):
        """Z-score standardization"""
        n = X.shape[0]
        S = np.cov(X.T, bias=True)
        mean_X = np.mean(X, axis=0)
        std_diag = np.sqrt(np.diag(S))
        Xs = (X - mean_X) / std_diag
        return Xs
    
    def minmax(self, X):
        """Min-max normalization"""
        n = X.shape[0]
        min_X = np.min(X, axis=0)
        max_X = np.max(X, axis=0)
        Xs = (X - min_X) / (max_X - min_X)
        return Xs
    
    def preproc(self, X, prep):
        """Main preprocessing function"""
        if prep == 0:
            return X
        elif prep == 1:
            return self.zscore(X)
        elif prep == 2:
            return self.minmax(X)
        else:
            raise ValueError("prep must be 0, 1, or 2")
    
    # ==================== AUXILIARY FUNCTIONS ====================
    
    def randPU(self, n, c):
        """Generate random partition matrix"""
        U = np.zeros((n, c))
        U[:c, :] = np.eye(c)
        U[c:, 0] = 1
        
        # Shuffle rows from c onwards
        for i in range(c, n):
            U[i] = np.random.permutation(U[i])
        
        # Shuffle all rows
        U = np.random.permutation(U)
        return U
    
    def km(self, X, K, Rndstart=10):
        """K-means clustering implementation"""
        n, J = X.shape
        
        S2x = np.tile(np.sum(X**2, axis=1, keepdims=True), (1, K))
        U = self.randPU(n, K)
        su = np.sum(U, axis=0)
        M = (U.T @ X) / su[:, np.newaxis]
        s2m = np.sum(M**2, axis=1)
        
        ssbold = np.sum(s2m * su)
        it = 0
        dif = np.inf
        
        while dif > 1e-8:
            it += 1
            Dist = S2x + s2m[np.newaxis, :] - 2 * X @ M.T
            U = (Dist == np.min(Dist, axis=1, keepdims=True)).astype(float)
            
            # Handle non-unique assignments
            while np.sum(np.sum(U, axis=1) > 1) > 0:
                surows = np.sum(U, axis=1)
                sumax = np.argmax(surows > 1)
                posmax = np.argmax(U[sumax])
                U[sumax] = 0
                U[sumax, posmax] = 1
            
            su = np.sum(U, axis=0)
            
            # Handle empty clusters
            while np.sum(su == 0) > 0:
                sumin = np.argmin(su)
                sumax = np.argmax(su)
                o = np.argmax(U[:, sumax])
                U[o, sumin] = 1
                U[o, sumax] = 0
                su = np.sum(U, axis=0)
            
            # Update centroids
            M = (U.T @ X) / su[:, np.newaxis]
            s2m = np.sum(M**2, axis=1)
            
            # Check convergence
            ssb = np.sum(s2m * su)
            dif = ssb - ssbold
            ssbold = ssb
        
        return U
    
    def split_maxwd(self, su, wd, U, Xs, K):
        """Split cluster with maximum within-cluster distance"""
        sumin = np.argmin(su)
        wdmax = np.argmax(wd)
        splitk = [sumin, wdmax]
        isplit = np.where(U[:, wdmax] == 1)[0]
        U[isplit] = 0
        
        if len(isplit) > 2:
            U[np.ix_(isplit, splitk)] = self.km(Xs[isplit], 2, 10)
        elif len(isplit) == 2:
            U[np.ix_(isplit, splitk)] = np.eye(2)
        
        return U
    
    def assign_ssed(self, S2x, M, Xs, K, n):
        """Assignment based on sum of squared Euclidean distances"""
        s2m = np.sum(M**2, axis=1)
        Dist = S2x + s2m[np.newaxis, :] - 2 * Xs @ M.T
        U = (Dist == np.min(Dist, axis=1, keepdims=True)).astype(float)
        
        # Handle non-unique assignments
        while np.sum(np.sum(U, axis=1) > 1) > 0:
            surows = np.sum(U, axis=1)
            sumax = np.argmax(surows > 1)
            posmax = np.argmax(U[sumax])
            U[sumax] = 0
            U[sumax, posmax] = 1
        
        return U
    
    def varimax(self, A):
        """Varimax rotation"""
        conv = 1e-6
        m, r = A.shape
        T = np.eye(r)
        B = A.copy()
        mones = np.ones((m, 1))
        
        f = np.sum((B**2 - mones @ np.sum(B**2, axis=0, keepdims=True) / m)**2)
        fold = f - 2 * conv * f
        if f == 0:
            fold = -conv
        
        while f - fold > f * conv:
            fold = f
            for i in range(r):
                for j in range(i + 1, r):
                    x = B[:, i]
                    y = B[:, j]
                    xx = T[:, i]
                    yy = T[:, j]
                    
                    u = x**2 - y**2
                    v = 2 * x * y
                    u = u - np.sum(u) / m
                    v = v - np.sum(v) / m
                    
                    a = 2 * np.sum(u * v)
                    b = np.sum(u**2) - np.sum(v**2)
                    c = np.sqrt(a**2 + b**2)
                    
                    sign = 1 if a >= 0 else -1
                    
                    if c < 1e-11:
                        cos = 1
                        sin = 0
                    else:
                        vvv = -sign * np.sqrt((b + c) / (2 * c))
                        sin = np.sqrt(0.5 - 0.5 * vvv)
                        cos = np.sqrt(0.5 + 0.5 * vvv)
                    
                    v_new = cos * x - sin * y
                    w = cos * y + sin * x
                    vv = cos * xx - sin * yy
                    ww = cos * yy + sin * xx
                    
                    if vvv >= 0:
                        B[:, i] = v_new
                        B[:, j] = w
                        T[:, i] = vv
                        T[:, j] = ww
                    else:
                        B[:, j] = v_new
                        B[:, i] = w
                        T[:, j] = vv
                        T[:, i] = ww
            
            f = np.sum((B**2 - mones @ np.sum(B**2, axis=0, keepdims=True) / m)**2)
        
        return B
    
    # ==================== MAIN ALGORITHMS ====================
    
    def redkm(self, X, K, Q, Rndstart=20, verbose=0, maxiter=100, tol=1e-6, rot=0, prep=1, print_stats=0):
        """
        Reduced K-Means (RKM) Algorithm
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data matrix
        K : int
            Number of clusters for units
        Q : int
            Number of principal components
        Rndstart : int, default=20
            Number of random starts
        verbose : int, default=0
            Verbosity level
        maxiter : int, default=100
            Maximum iterations
        tol : float, default=1e-6
            Tolerance for convergence
        rot : int, default=0
            Apply varimax rotation (1) or not (0)
        prep : int, default=1
            Preprocessing: 0=none, 1=zscore, 2=minmax
        print_stats : int, default=0
            Print statistics
        
        Returns:
        --------
        dict : Results dictionary containing U, A, centers, etc.
        """
        
        Xs = self.preproc(X, prep)
        n, J = Xs.shape
        
        Xs2 = Xs**2
        st = np.sum(Xs2)
        S = np.cov(Xs.T, bias=True)
        S2x = np.tile(np.sum(Xs**2, axis=1, keepdims=True), (1, K))
        
        fbest = 0
        
        for loop in range(Rndstart):
            # Initialization
            U = self.randPU(n, K)
            su = np.sum(U, axis=0)
            Xmean = (U.T @ Xs) / su[:, np.newaxis]
            
            # Update A
            XX = Xs.T @ U @ np.diag(1.0/su) @ U.T @ Xs
            eigenvals, A = eigh(XX)
            
            # Sort eigenvalues and eigenvectors in descending order
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            A = A[:, idx]
            A = A[:, :Q]
            
            # Update Ymean
            Ymean = Xmean @ A
            f0 = np.trace(Ymean.T @ U.T @ U @ Ymean)
            fdif = 2 * tol
            it = 0
            
            # Iteration phase
            while fdif > tol and it < maxiter:
                it += 1
                
                # Update U
                U = self.assign_ssed(S2x, Ymean @ A.T, Xs, K, n)
                su = np.sum(U, axis=0)
                
                # Handle empty clusters
                while np.sum(su == 0) > 0:
                    Xmean = (U.T @ Xs) / su[:, np.newaxis]
                    Xmean2 = (U.T @ Xs2) / su[:, np.newaxis]
                    wd = np.sum(np.diag(su) @ (Xmean2 - (Xmean @ A @ A.T)**2), axis=1)
                    wd[su <= 1] = 0
                    U = self.split_maxwd(su, wd, U, Xs, K)
                    su = np.sum(U, axis=0)
                
                Xmean = (U.T @ Xs) / su[:, np.newaxis]
                
                # Update A
                XX = Xs.T @ U @ np.diag(1.0/su) @ U.T @ Xs
                eigenvals, A = eigh(XX)
                idx = np.argsort(eigenvals)[::-1]
                A = A[:, idx[:Q]]
                
                # Update Ymean and Y
                Ymean = Xmean @ A
                Y = Xs @ A
                
                # Within sum of squares
                Xmean2 = (U.T @ Xs2) / su[:, np.newaxis]
                wd = np.sum(np.diag(su) @ (Xmean2 - (Xmean @ A @ A.T)**2), axis=1)
                
                f = np.trace(Ymean.T @ U.T @ U @ Ymean)
                fdif = f - f0
                
                if fdif > tol:
                    f0 = f
                    A0 = A.copy()
                else:
                    break
            
            if verbose:
                print(f"RKM: Loop = {loop + 1}; Explained variance (%) = {(f/st)*100:.2f}; iter = {it+1}; fdif = {fdif}")
            
            # Store best results
            if loop == 0 or f > fbest:
                Ubest = U.copy()
                Abest = A.copy()
                Ybest = Xs @ Abest
                Xmeanbest = Xmean.copy()
                fbest = f
                loopbest = loop + 1
                itbest = it + 1
                fdifbest = fdif
                wdbest = wd.copy()
        
        # Sort components by variance
        varY = np.var(Ybest, axis=0)
        ic = np.argsort(varY)[::-1]
        Abest = Abest[:, ic]
        
        if rot:
            Abest = self.varimax(Abest)
        
        Ybest = Ybest[:, ic]
        Ymeanbest = Xmeanbest @ Abest
        
        # Sort clusters by size
        cluster_sizes = np.sum(Ubest, axis=0)
        iicc = np.argsort(cluster_sizes)[::-1]
        Ubest = Ubest[:, iicc]
        wdbest = wdbest[iicc]
        
        pseudoF = (fbest/(K-1)) / ((st-fbest)/(n-K))
        
        if verbose:
            print(f"RKM (Final): Explained Variance (%) = {(fbest/st)*100:.2f}; loop = {loopbest}; iter = {itbest}; fdif = {fdifbest}")
        
        return {
            'U': Ubest,
            'A': Abest,
            'centers': Ymeanbest,
            'withinss': wdbest,
            'betweenss': fbest,
            'totss': st,
            'size': np.sum(Ubest, axis=0),
            'pseudoF': pseudoF,
            'loop': loopbest,
            'it': itbest
        }
    
    def factkm(self, X, K, Q, Rndstart=20, verbose=0, maxiter=100, tol=1e-6, rot=0, prep=1, print_stats=0):
        """
        Factorial K-Means (FKM) Algorithm
        
        Parameters and Returns similar to redkm
        """
        
        Xs = self.preproc(X, prep)
        n, J = Xs.shape
        
        Xs2 = Xs**2
        st = np.sum(Xs2)
        S = np.cov(Xs.T, bias=True)
        S2x = np.tile(np.sum(Xs**2, axis=1, keepdims=True), (1, K))
        
        fbest = 0
        
        for loop in range(Rndstart):
            # Initialization
            U = self.randPU(n, K)
            su = np.sum(U, axis=0)
            Xmean = (U.T @ Xs) / su[:, np.newaxis]
            
            # Update A
            XX = Xs.T @ U @ np.diag(1.0/su) @ U.T @ Xs
            eigenvals, A = eigh(XX)
            
            # Sort eigenvalues and eigenvectors in descending order
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            A = A[:, idx]
            A = A[:, :Q]
            
            # Project Xs, Xmean on A
            Ymean = Xmean @ A
            Y = Xs @ A
            
            f0 = np.trace(Ymean.T @ U.T @ U @ Ymean)
            fdif = 2 * tol
            it = 0
            
            # Iteration phase
            while fdif > tol and it < maxiter:
                it += 1
                
                # Update U using projected data
                S2x = np.tile(np.sum(Y**2, axis=1, keepdims=True), (1, K))
                U = self.assign_ssed(S2x, Ymean, Y, K, n)
                su = np.sum(U, axis=0)
                
                # Handle empty clusters
                while np.sum(su == 0) > 0:
                    Xmean = (U.T @ Xs) / su[:, np.newaxis]
                    Xmean2 = (U.T @ Xs2) / su[:, np.newaxis]
                    Ymean2 = (U.T @ (Xs @ A)**2) / su[:, np.newaxis]
                    wd = np.sum(np.diag(su) @ (Ymean2 - (Xmean @ A)**2), axis=1)
                    wd[su <= 1] = 0
                    U = self.split_maxwd(su, wd, U, Xs @ A, K)
                    su = np.sum(U, axis=0)
                
                # Update centroids
                Xmean = (U.T @ Xs) / su[:, np.newaxis]
                
                # Update A
                XX = Xs.T @ U @ np.diag(1.0/su) @ U.T @ Xs
                eigenvals, A = eigh(XX)
                idx = np.argsort(eigenvals)[::-1]
                A = A[:, idx[:Q]]
                
                # Project Xs and Xmean on A
                Ymean = Xmean @ A
                Y = Xs @ A
                
                # Within sum of squares
                Ymean2 = (U.T @ (Xs @ A)**2) / su[:, np.newaxis]
                Xmean2 = (U.T @ Xs2) / su[:, np.newaxis]
                wd = np.sum(np.diag(su) @ (Ymean2 - (Xmean @ A)**2), axis=1)
                
                f = np.trace(Ymean.T @ U.T @ U @ Ymean)
                fdif = f - f0
                
                if fdif > tol:
                    f0 = f
                    A0 = A.copy()
                else:
                    break
            
            if verbose:
                print(f"FKM: Loop = {loop + 1}; Explained variance (%) = {(f/st)*100:.2f}; iter = {it}; fdif = {fdif}")
            
            # Store best results
            if loop == 0 or f > fbest:
                Ubest = U.copy()
                Abest = A.copy()
                Ybest = Xs @ Abest
                Ymbest = Ymean.copy()
                fbest = f
                loopbest = loop + 1
                itbest = it + 1
                fdifbest = fdif
                wdbest = wd.copy()
        
        # Sort components by variance and rotate factors
        varY = np.var(Ybest, axis=0)
        ic = np.argsort(varY)[::-1]
        varY = varY[ic]
        Abest = Abest[:, ic]
        Ybest = Ybest[:, ic]
        
        if rot:
            Abest = self.varimax(Abest)
        
        # Sort clusters by size
        cluster_sizes = np.sum(Ubest, axis=0)
        iicc = np.argsort(cluster_sizes)[::-1]
        Ubest = Ubest[:, iicc]
        wdbest = wdbest[iicc]
        
        pseudoF = (fbest/(K-1)) / ((st-fbest)/(n-K))
        
        if verbose:
            print(f"FKM (Final): Explained variance (%) = {(fbest/st)*100:.2f}; loop = {loopbest}; iter = {itbest}; fdif = {fdifbest}")
        
        return {
            'U': Ubest,
            'A': Abest,
            'centers': Ymbest,
            'withinss': wdbest,
            'betweenss': fbest,
            'totss': st,
            'size': np.sum(Ubest, axis=0),
            'pseudoF': pseudoF,
            'loop': loopbest,
            'it': itbest
        }
    
    def dpcakm(self, X, K, Q, Rndstart=20, verbose=0, maxiter=100, tol=1e-6, constr=None, prep=1, print_stats=0):
        """
        Clustering with Disjoint Principal Components Analysis (CDPCA)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data matrix
        K : int
            Number of clusters for units
        Q : int
            Number of principal components
        Rndstart : int, default=20
            Number of random starts
        verbose : int, default=0
            Verbosity level
        maxiter : int, default=100
            Maximum iterations
        tol : float, default=1e-6
            Tolerance for convergence
        constr : array-like, optional
            Constraint vector for variables
        prep : int, default=1
            Preprocessing: 0=none, 1=zscore, 2=minmax
        print_stats : int, default=0
            Print statistics
        
        Returns:
        --------
        dict : Results dictionary containing U, A, V, centers, etc.
        """
        
        Xs = self.preproc(X, prep)
        n, J = Xs.shape
        
        if constr is None:
            constr = np.zeros(J)
        
        Xs = Xs * (n / (n - 1))  # Bias correction
        Xs2 = Xs**2
        st = np.sum(Xs2)
        
        S = np.cov(Xs.T, bias=True)
        S2x = np.tile(np.sum(Xs**2, axis=1, keepdims=True), (1, K))
        
        JJ = np.arange(J)
        VC = np.eye(Q)
        fbest = 0
        
        for loop in range(Rndstart):
            # Initialization with constraints
            flg = 1
            while flg > 0:
                V = self.randPU(J, Q)
                for j in range(J):
                    if constr[j] > 0:
                        V[j] = VC[int(constr[j]) - 1]
                flg = np.sum(np.sum(V, axis=0) == 0)
            
            U = self.randPU(n, K)
            su = np.sum(U, axis=0)
            A = np.zeros((J, Q))
            
            # Update centroid matrix
            Xmean = (U.T @ Xs) / su[:, np.newaxis]
            
            # Initialize A for each component
            for g in range(Q):
                ibCg = V[:, g]
                JCg = JJ[ibCg == 1]
                S = Xs.T @ U @ np.diag(1.0/su) @ U.T @ Xs
                
                if np.sum(ibCg) > 1:
                    Sg = S[np.ix_(JCg, JCg)]
                    eigenvals, Av = eigh(Sg)
                    a = np.argmax(np.abs(eigenvals))
                    A[JCg, g] = Av[:, a]
                else:
                    A[JCg, g] = 1
            
            Ymean = Xmean @ A
            f0 = np.trace(Ymean.T @ U.T @ U @ Ymean)
            fmax = 0
            fdif = 2 * tol
            
            # Iteration phase
            for it in range(maxiter):
                # Update U
                Y = Xs @ A
                S2x = np.tile(np.sum(Y**2, axis=1, keepdims=True), (1, K))
                U = self.assign_ssed(S2x, Ymean, Y, K, n)
                su = np.sum(U, axis=0)
                
                # Handle empty clusters
                while np.sum(su == 0) > 0:
                    Xmean = (U.T @ Xs) / su[:, np.newaxis]
                    Xmean2 = (U.T @ Xs2) / su[:, np.newaxis]
                    Ymean2 = (U.T @ (Xs @ A)**2) / su[:, np.newaxis]
                    wd = np.sum(np.diag(su) @ (Ymean2 - (Xmean @ A)**2), axis=1)
                    wd[su <= 1] = 0
                    U = self.split_maxwd(su, wd, U, Xs @ A, K)
                    su = np.sum(U, axis=0)
                
                # Update centroids
                Xmean = (U.T @ Xs) / su[:, np.newaxis]
                
                # Update V and A
                S = Xs.T @ U @ np.diag(1.0/su) @ U.T @ Xs
                A0 = A.copy()
                
                for j in range(J):
                    if constr[j] == 0:  # Only update unconstrained variables
                        posmax = np.argmax(V[j] == 1)
                        
                        for g in range(Q):
                            V_temp = V.copy()
                            V_temp[j] = VC[g]
                            
                            if np.sum(V_temp[:, posmax]) > 0:
                                # Update A for new assignment
                                ibCg = V_temp[:, g]
                                ibCpm = V_temp[:, posmax]
                                JCg = JJ[ibCg == 1]
                                JCpm = JJ[ibCpm == 1]
                                
                                A_temp = A.copy()
                                A_temp[:, g] = 0
                                A_temp[:, posmax] = 0
                                
                                # Update component g
                                if np.sum(ibCg) > 1:
                                    Sg = S[np.ix_(JCg, JCg)]
                                    eigenvals, Av = eigh(Sg)
                                    a = np.argmax(np.abs(eigenvals))
                                    if np.sum(Av[:, a]) < 0:
                                        Av[:, a] = -Av[:, a]
                                    A_temp[JCg, g] = Av[:, a]
                                else:
                                    A_temp[JCg, g] = 1
                                
                                # Update component posmax
                                if np.sum(ibCpm) > 1:
                                    Sg = S[np.ix_(JCpm, JCpm)]
                                    eigenvals, AAv = eigh(Sg)
                                    aa = np.argmax(np.abs(eigenvals))
                                    if np.sum(AAv[:, aa]) < 0:
                                        AAv[:, aa] = -AAv[:, aa]
                                    A_temp[JCpm, posmax] = AAv[:, aa]
                                else:
                                    A_temp[JCpm, posmax] = 1
                                
                                Ymean_temp = Xmean @ A_temp
                                f_temp = np.trace(Ymean_temp.T @ U.T @ U @ Ymean_temp)
                                
                                if f_temp > fmax:
                                    fmax = f_temp
                                    posmax = g
                                    A0 = A_temp.copy()
                                else:
                                    A = A0.copy()
                        
                        V[j] = VC[posmax]
                
                A = A0.copy()
                Y = Xs @ A
                
                # Within sum of squares
                Ymean = Xmean @ A
                Ymean2 = (U.T @ (Xs @ A)**2) / su[:, np.newaxis]
                Xmean2 = (U.T @ Xs2) / su[:, np.newaxis]
                wd = np.sum(np.diag(su) @ (Ymean2 - (Xmean @ A)**2), axis=1)
                
                f = np.trace(Ymean.T @ U.T @ U @ Ymean)
                fdif = f - f0
                
                if fdif > tol:
                    f0 = f
                    fmax = f0
                    A0 = A.copy()
                else:
                    break
            
            Ymean = Xmean @ A
            f = np.trace(Ymean.T @ U.T @ U @ Ymean)
            fdif = f - f0
            
            if verbose:
                print(f"CDPCA: Loop = {loop + 1}; Explained Variance (%) = {(f/st)*100:.2f}; iter = {it+1}; fdif = {fdif}")
            
            # Store best results
            if loop == 0 or f > fbest:
                Vbest = V.copy()
                Ubest = U.copy()
                Abest = A.copy()
                Ybest = Xs @ Abest
                fbest = f
                Ymbest = Ymean.copy()
                loopbest = loop + 1
                itbest = it + 1
                fdifbest = fdif
                wdbest = wd.copy()
        
        Ybest = Xs @ Abest
        
        # Sort clusters of variables by variance
        varY = np.var(Ybest, axis=0)
        ic = np.argsort(varY)[::-1]
        Abest = Abest[:, ic]
        Vbest = Vbest[:, ic]
        Ybest = Ybest[:, ic]
        
        # Sort clusters of objects by size
        cluster_sizes = np.sum(Ubest, axis=0)
        ic = np.argsort(cluster_sizes)[::-1]
        Ubest = Ubest[:, ic]
        wdbest = wdbest[ic]
        
        pseudoF = (fbest/(K-1)) / ((st-fbest)/(n-K))
        
        if verbose:
            print(f"CDPCA (Final): Percentage of Explained Variance (%) = {(fbest/st)*100:.2f}; loop = {loopbest}; iter = {itbest}; fdif = {fdifbest}")
        
        return {
            'U': Ubest,
            'A': Abest,
            'V': Vbest,
            'centers': Ymbest,
            'withinss': wdbest,
            'betweenss': fbest,
            'totss': st,
            'K-size': np.sum(Ubest, axis=0),
            'Q-size': np.sum(Vbest, axis=0),
            'pseudoF': pseudoF,
            'loop': loopbest,
            'it': itbest
        }

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    # Initialize the class
    drclust = DrClust()
    
    # Run RKM
    print("Running Reduced K-Means...")
    rkm_results = drclust.redkm(X, K=3, Q=2, Rndstart=5, verbose=1)
    
    # Run FKM
    print("\nRunning Factorial K-Means...")
    fkm_results = drclust.factkm(X, K=3, Q=2, Rndstart=5, verbose=1)
    
    # Run CDPCA
    print("\nRunning CDPCA...")
    cdpca_results = drclust.dpcakm(X, K=3, Q=2, Rndstart=5, verbose=1)