import numpy as np
from sklearn.base import BaseEstimator
from scipy.linalg import eigh, svd
from sklearn.cluster import KMeans


class SimulatenousClustering(BaseEstimator):
    """
    Base class for Simultaneous Clustering Methods.
    
    Parameters
    ----------
    K : int
        Number of groups/clusters to partion into
    L : int
        Number of factors in reduced space
    Rndstart : int, default=20
        Number of random starts
    maxiter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-6
        Tolerance for convergence
    Rndstate : int or None, default=None
        Seed for random number generation
    """
    def __init__(self, K, L, Rndstart=20, maxiter=100, tol=1e-6, Rndstate=None):
            self.K = K
            self.L = L
            self.Rndstart = Rndstart
            self.maxiter = maxiter
            self.tol = tol
            self.Rndstate = Rndstate

    def zscore(self, X):
        """Standardize data: (X - mean) / std with zero-division safety."""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1.0 
        return (X - mean) / std

    def whiten(self, X, epsilon=1e-10):
        """
        ZCA Whitening (Sphering)
        """
        # Center
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        eigenvals, eigenvecs = eigh(np.cov(X_centered, rowvar=False, bias=True))
        
        # Whitening Matrix W = V * S^(-1/2) * V.T
        inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(eigenvals, epsilon)))
        W = eigenvecs @ inv_sqrt @ eigenvecs.T
        
        # Transform
        X_white = X_centered @ W
        return X_white, mean, W

    def initialise_U(self, N):
        """Generate random partition matrix U"""
        rng = np.random.default_rng(self.Rndstate)
        labels = rng.integers(0, self.K, size=N)
        U = np.zeros((N, self.K))
        U[np.arange(N), labels] = 1
        return U
    
    def compute_centroids(self, X, U, weighted=False):
        """
        Compute cluster centroids/means
        (U^T U)^{-1} U^T X
        """
        # Compute sum of points in each cluster
        cluster_sums = U.T @ X
        # Find number of points in each cluster and remove zeros
        cluster_sizes = U.sum(axis=0)
        cluster_sizes[cluster_sizes==0] = 1

        if weighted:
            # Weighted output for RKM (sqrt(n_k)*mu_k)
            return np.diag(np.sqrt(1/cluster_sizes)) @ cluster_sums
        # General output
        return np.diag(1/cluster_sizes) @ cluster_sums

    def assign_clusters(self, X, centroids):
        """Assigns objects to the nearest centroid (K-means step)"""
        N = X.shape[0]
        X_sq = np.sum(X**2, axis=1)[:, np.newaxis] # (N, 1)
        C_sq = np.sum(centroids**2, axis=1) # (K, )
        dist = X_sq + C_sq - 2 * np.dot(X, centroids.T)

        # Find index of nearest of K centroid
        labels = np.argmin(dist, axis=1)
        U = np.zeros((N, self.K))
        U[np.arange(N), labels] = 1
        return U

    def split_clusters(self, X, U, centroids):
        """Split cluster with maximum within-cluster distance when there are empty clusters"""
         
        # Check if no empty clusters
        if np.all(U.sum(axis=0)>0):
            return U
        
        # Find all empty clusters and current cluster assignments
        empty_clusters = np.where(U.sum(axis=0) == 0)[0]
        labels = np.argmax(U, axis=1)

        for cluster_idx in empty_clusters:
            # Compute distances between points assigned cluster centroid
            dist_to_centroid = np.sum((X-centroids[labels])**2, axis=1)
            # Find cluster with largest within-cluster distance
            wc_dist = np.bincount(labels, weights=dist_to_centroid, minlength=self.K)

            # Set empty clusters to -1 so they aren't picked
            wc_dist[empty_clusters] = -1

            # Choose cluster to split
            split_idx = np.argmax(wc_dist)
            split_points = np.where(labels == split_idx)[0]

            # Need at least 2 points in cluster to split
            if len(split_points) < 2:
                continue

            # Split the cluster into two using standard K-Means
            sub_km = KMeans(n_clusters=2, random_state=self.Rndstate)
            sub_labels = sub_km.fit_predict(X[split_points])
            
            # Identify which points have been moved
            moved = split_points[sub_labels == 1]
            if len(moved) > 0:
                # Update U
                U[moved, split_idx] = 0
                U[moved, cluster_idx] = 1

                # Update labels and centroids
                labels[moved] = cluster_idx
                centroids[split_idx] = np.mean(X[labels == split_idx], axis=0)
                centroids[cluster_idx] = np.mean(X[labels == cluster_idx], axis=0)

        return U
    
    def varimax(self, A, tolerance=1e-6, max_iter=100):
        """Varimax rotation using SVD"""
        P, L = A.shape
        # Initialize rotation matrix R as identity
        R = np.eye(L)
        # Calculate current variance
        d = 0
        
        for i in range(max_iter):
            d_old = d
            
            # Project loadings
            Lambda = np.dot(A, R)
            # Gradient of the Varimax objective function
            alpha = Lambda**3 - Lambda * (np.sum(Lambda**2, axis=0) / P)
            # Use SVD to find update that maximises variance
            U, S, Vt = np.linalg.svd(np.dot(A.T, alpha))
            
            # Update rotation matrix, calculate new variance
            R = np.dot(U, Vt)
            d = np.sum(S)
            
            # Check convergence
            if d_old != 0 and d / d_old < 1 + tolerance:
                break
        # Return rotated loadings
        return np.dot(A, R)



class RKM(SimulatenousClustering):
    def update_A(self, X, U):
        """Update subspace A by maximising Between-Cluster Variance"""
        # Obtain weighted centroids matrix (U^T U)^{1/2} C
        weighted_centroids_X = self.compute_centroids(X, U, weighted=True)
        
        # Find L leading right singular vectors
        eigenvals, eigenvectors = eigh(weighted_centroids_X.T @ weighted_centroids_X)
        A = eigenvectors[:, :-self.L-1:-1]
        return A

    def fit(self, X):
        """Run RKM algorithm"""
        # Normalise data
        X_std = self.zscore(X)
        N, P = X_std.shape

        # compute 
        total_var = np.sum(X_std**2)
        best_loss = np.inf

        for run in range(self.Rndstart):
            U = self.initialise_U(N)
            A = self.update_A(X_std, U)
            loss_old = np.inf

            for it in range(self.maxiter):
                # U step
                Y = X_std @ A
                centroids_Y = self.compute_centroids(Y, U)
                U = self.assign_clusters(Y, centroids_Y)
                U = self.split_clusters(Y, U, centroids_Y)

                # A step
                A = self.update_A(X_std, U)

                # Compute loss
                weighted_centroids_X = self.compute_centroids(X_std, U, weighted=True)
                explained_var = np.trace(A.T @ weighted_centroids_X.T @ weighted_centroids_X @ A)
                # ||X-P_U XAA^T||^2 = ||X||^2 - ||P_U XAA^T||^2
                loss = total_var - explained_var

                # Check convergence in loss
                if it > 0 and abs(loss_old - loss) < self.tol:
                    break
                loss_old = loss
            
            # Store best loss
            if loss < best_loss:
                best_loss = loss
                self.loss = loss
                self.U = U.copy()
                self.A = A.copy()
                self.centroids = self.compute_centroids(X_std@A, U)
        return self


class FKM(SimulatenousClustering):
    def update_A(self, X, U):
        """Update subspace A"""
        # Need eigenvectors of X^T (P_U - I_N) X
        # Equal to [X^T P_U X - X^T X]

        # Obtain X^T P_U X first
        weighted_centroids_X = self.compute_centroids(X, U, weighted=True)
        # Obtain target matrix and corresponding eigenvectors
        target = (X.T@X) - (weighted_centroids_X.T @ weighted_centroids_X)
        eigenvals, eigenvectors = eigh(target)
        # Set A as L leading eigenvectors of target
        A = eigenvectors[:, :self.L]
        return A

    def fit(self, X):
        """Run FKM algorithm"""
        # Normalise data
        X_white, mean, W = self.whiten(X)
        N, P = X_white.shape

        best_loss = np.inf

        for run in range(self.Rndstart):
            U = self.initialise_U(N)
            A = self.update_A(X_white, U)
            loss_old = np.inf

            for it in range(self.maxiter):
                # U step
                Y = X_white @ A
                centroids_Y = self.compute_centroids(Y, U)
                U = self.assign_clusters(Y, centroids_Y)
                U = self.split_clusters(Y, U, centroids_Y)

                # A step
                A = self.update_A(X_white, U)

                # Compute loss
                # ||XA - P_U XA||^2
                labels = np.argmax(U, axis=1)
                Y = X_white @ A
                centroids_Y = self.compute_centroids(Y, U)
                loss = np.sum((Y-centroids_Y[labels])**2)

                # Check convergence in loss
                if it > 0 and abs(loss_old - loss) < self.tol:
                    break
                loss_old = loss
            
            # Store best loss
            if loss < best_loss:
                best_loss = loss
                self.loss = loss
                self.U = U.copy()
                self.A = (W @ A).copy()
                self.centroids = self.compute_centroids((X-mean)@self.A, U)
        return self


class CDPCA(SimulatenousClustering):
    def update_A(self, X, U, V_labels, weighted_centroids_X=None):
        if weighted_centroids_X is None:
            weighted_centroids_X = self.compute_centroids(X, U, weighted=True)
        # Set up new A
        A = np.zeros((X.shape[1], self.L))

        # Iterate through factors
        for l in range(self.L):
            # Identify variables that laod onto this factor
            idx = np.where(V_labels == l)[0]
            
            if len(idx) > 0:
                # Sub-matrix of weighted centroids, for variables corresponding to factor l
                C_sub = weighted_centroids_X[:, idx]
                try:
                    # Compute SVD of C_sub
                    u, s, vt = svd(C_sub, full_matrices=False)
                    # Only need leading eigenvector
                    A[idx, l] = vt[0]

                except np.linalg.LinAlgError:
                    # Potential inversion/singular errors
                    A[idx, l] = 0
        return A
    
    def update_V(self, A, weighted_centroids_X):
        """
        Update Variable Partition V with 'Non-Empty Factor' Guarantee.
        Ensures every factor has at least one variable assigned.
        """
        # Naive/greedy update with potential zero columns
        target = weighted_centroids_X.T @ (weighted_centroids_X @ A)
        abs_target = np.abs(target)
        labels = np.argmax(abs_target, axis=1)
        
        # Find indices of factors that have no variables that load onto it
        counts = np.bincount(labels, minlength=self.L)
        empty_factors = np.where(counts == 0)[0]
        
        # Check if there are 'empty' factors
        if len(empty_factors) > 0:
            for ef in empty_factors:
                # For each empty factor; use coefficients to sort variables
                candidates = np.argsort(abs_target[:, ef])[::-1]
                
                # Iterate through sorted and steal best candidate
                for var_idx in candidates:
                    current_owner = labels[var_idx]
                    # Only steal if the current owner has > 1 variable
                    if counts[current_owner] > 1:
                        # Assign var to empty factor
                        labels[var_idx] = ef
                        # Update counts
                        counts[current_owner] -= 1
                        counts[ef] += 1
                        # Filled empty factor loading
                        break
        return labels

    def fit(self, X):
        """Run CDPCA ALS algorithm"""
        # Normalise data
        X_std = self.zscore(X)
        N, P = X_std.shape

        # compute 
        total_var = np.sum(X_std**2)
        best_loss = np.inf

        for run in range(self.Rndstart):
            U = self.initialise_U(N)
            # Initialise variable to factor partition:
            # Set up V with at least one variable corresponding to each factor
            V_labels = np.random.permutation(np.concatenate([np.arange(self.L), np.random.randint(0, self.L, size=P-self.L)]))
            A = self.update_A(X_std, U, V_labels)
            loss_old = np.inf

            for it in range(self.maxiter):
                # U step
                Y = X_std @ A
                centroids_Y = self.compute_centroids(Y, U)
                U = self.assign_clusters(Y, centroids_Y)
                U = self.split_clusters(Y, U, centroids_Y)

                # Matrix W where W.T @ W = X^T P_U X
                weighted_centroids_X = self.compute_centroids(X_std, U, weighted=True)

                # A step
                A = self.update_A(X_std, U, V_labels, weighted_centroids_X)
                # V step
                V_labels = self.update_V(A, weighted_centroids_X)

                # Compute loss
                explained_var = np.trace(A.T @ weighted_centroids_X.T @ weighted_centroids_X @ A)
                # ||X-P_U XAA^T||^2 = ||X||^2 - ||P_U XAA^T||^2
                loss = total_var - explained_var

                # Check convergence in loss
                if it > 0 and abs(loss_old - loss) < self.tol:
                    break
                loss_old = loss
            
            # Store best loss
            if loss < best_loss:
                best_loss = loss
                self.loss = loss
                self.U = U.copy()
                self.A = A.copy()
                self.V_labels = V_labels.copy()
                self.centroids = self.compute_centroids(X_std@A, U)
        return self


class GRC(SimulatenousClustering):
    def __init__(self, K, L, L_d, rho_1, rho_2, alpha=1e-4, Rndstart=20, maxiter=100, tol=1e-6, Rndstate=None):
        """
        GRC Initialisation
        """
        # Check Constraints
        if rho_1 <= rho_2:
            raise ValueError(f"Constraint violated: rho_1 ({rho_1}) must be strictly greater than rho_2 ({rho_2}).")
        super().__init__(K=K, L=L, Rndstart=Rndstart, maxiter=maxiter, tol=tol, Rndstate=Rndstate)

        # Store GRC-specific parameters
        self.L_d = L_d
        self.L_c = L - L_d
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.alpha = alpha
    
    def update_A(self, X, U, A, max_subiter=10):
        # Pre-compute fixed terms
        XtX = X.T @ X
        weighted_centroids_X = self.compute_centroids(X, U, weighted=True)
        Xt_PU_X = weighted_centroids_X.T @ weighted_centroids_X

        # Run set amount of projections via GP algorithm
        for sub_it in range(max_subiter):
            A_old = A.copy()

            # Split components of A
            Ac, Ad = A[:, :self.L_c], A[:, self.L_c:]

            # Compute gradient matrix
            G = np.zeros_like(A)
            # Compute G_c
            G[:, :self.L_c] = (-2*(1-self.rho_1) * XtX @ Ac) + (-2*(self.rho_1-self.rho_2) * Xt_PU_X @ Ac)
            # Compute G_d
            G[:, self.L_c:] = -2 * XtX @ Ad

            # Move A against gradient
            # POTENTIALLY ADD: Exact line search
            A_target = A - self.alpha * G
            # Project A onto orthogonal manifold
            u, s, vt = svd(A_target, full_matrices=False)
            A = u @ vt
        return A

    def compute_loss(self, X, U, A):
        """Compute GRC loss function"""
        # Pre-compute fixed terms
        Ac = A[:, :self.L_c]
        XtX = X.T @ X
        weighted_centroids_X = self.compute_centroids(X, U, weighted=True)
        Xt_PU_X = weighted_centroids_X.T @ weighted_centroids_X

        # Term 1: ||X - XAA^T||^2
        term_1 = np.sum(X**2) - np.trace(A.T @ XtX @ A)

        # Term 2: ||X A_c - P_U X A_c||^2
        term_2 = np.trace(Ac.T @ XtX @ Ac) - np.trace(Ac.T @ Xt_PU_X @ Ac)

        # Term 3: ||P_U X A_c||^2
        term_3 = np.trace(Ac.T @ Xt_PU_X @ Ac)

        return term_1 + self.rho_1 * term_2 + self.rho_2 * term_3
    
    def fit(self, X):
        """Run GRC algorithm"""
        # Normalise data
        X_white, mean, W = self.whiten(X)
        N, P = X_white.shape

        best_loss = np.inf

        for run in range(self.Rndstart):
            # Initialise U and A
            U = self.initialise_U(N)
            A = np.linalg.qr(np.random.randn(P, self.L))[0]
            loss_old = np.inf

            for it in range(self.maxiter):
                # Update A
                A = self.update_A(X_white, U, A)
                # Clustering relevant sub-space
                Ac = A[:, :self.L_c]
            
                # Update U
                Yc = X_white @ Ac
                centroids_Yc = self.compute_centroids(Yc, U)
                U = self.assign_clusters(Yc, centroids_Yc)
                U = self.split_clusters(Yc, U, centroids_Yc)

                # Compute loss
                loss = self.compute_loss(X_white, U, A)

                # Check convergence in loss
                if it > 0 and abs(loss_old - loss) < self.tol:
                    break

                loss_old = loss

            # Store best loss
            if loss < best_loss:
                best_loss = loss
                self.loss = loss
                self.U = U.copy()
                self.A = (W @ A).copy()
                self.Ac = self.A[:, :self.L_c]
                self.centroids = self.compute_centroids((X-mean)@self.Ac, U)
        return self
