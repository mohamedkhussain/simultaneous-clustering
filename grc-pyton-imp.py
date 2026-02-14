import numpy as np
from scipy.linalg import eigh, svd
from sklearn.cluster import KMeans
import warnings

class GRC:
    """
    Generalized Reduced Clustering (GRC) Algorithm
    Python implementation of the C code
    """
    
    def __init__(self):
        pass
    
    def identity_matrix(self, N):
        """Create identity matrix"""
        return np.eye(N)
    
    def kmeans_lloyd(self, X, centers, max_iter=1000):
        """
        Lloyd's K-means algorithm implementation
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data matrix
        centers : array-like, shape (n_clusters, n_features)
            Initial cluster centers
        max_iter : int
            Maximum iterations
            
        Returns:
        --------
        labels : array, cluster assignments
        centers : array, final centers
        wss : array, within-cluster sum of squares
        """
        n, p = X.shape
        k = centers.shape[0]
        labels = np.zeros(n, dtype=int)
        
        for iteration in range(max_iter):
            old_labels = labels.copy()
            
            # Assign points to nearest centers
            for i in range(n):
                distances = np.sum((X[i] - centers)**2, axis=1)
                labels[i] = np.argmin(distances)
            
            # Check for convergence
            if np.array_equal(labels, old_labels):
                break
            
            # Update centers
            new_centers = np.zeros_like(centers)
            counts = np.zeros(k)
            
            for i in range(n):
                cluster = labels[i]
                new_centers[cluster] += X[i]
                counts[cluster] += 1
            
            # Avoid division by zero
            for j in range(k):
                if counts[j] > 0:
                    new_centers[j] /= counts[j]
                else:
                    new_centers[j] = centers[j]  # Keep old center
            
            centers = new_centers
        
        # Calculate within-cluster sum of squares
        wss = np.zeros(k)
        for i in range(n):
            cluster = labels[i]
            wss[cluster] += np.sum((X[i] - centers[cluster])**2)
        
        return labels, centers, wss
    
    def loss_function(self, X, U, A, N_comp1, N_comp2, rho1, rho2):
        """
        Calculate the GRC loss function
        
        Parameters:
        -----------
        X : array, shape (N_sub, N_var)
            Data matrix
        U : array, shape (N_sub, N_clust)
            Cluster membership matrix
        A : array, shape (N_var, N_comp)
            Loading matrix
        N_comp1, N_comp2 : int
            Number of components in each part
        rho1, rho2 : float
            Regularization parameters
            
        Returns:
        --------
        loss : float
            Loss function value
        """
        N_sub, N_var = X.shape
        N_clust = U.shape[1]
        N_comp = A.shape[1]
        
        # Extract A1 (first N_comp1 components)
        A1 = A[:, :N_comp1]
        
        # Identity matrix
        I_N = np.eye(N_sub)
        
        # Calculate projection matrix P_U = U(U^T U)^{-1} U^T
        UtU = U.T @ U
        UtU_inv = np.linalg.inv(UtU + 1e-10 * np.eye(N_clust))  # Add small regularization
        P_U = U @ UtU_inv @ U.T
        
        # Calculate terms
        # Z1 = X - X A A^T
        AAt = A @ A.T
        Z1 = X - X @ AAt
        
        # Z2 = X A1 - P_U X A1
        XA1 = X @ A1
        Z2 = XA1 - P_U @ XA1
        
        # Z3 = P_U X A1
        Z3 = P_U @ XA1
        
        # Loss function terms
        term1 = np.trace(Z1.T @ Z1)
        term2 = np.trace(Z2.T @ Z2)
        term3 = np.trace(Z3.T @ Z3)
        
        loss = term1 + rho1 * term2 + rho2 * term3
        
        return loss
    
    def gradient(self, X, U, A, N_comp1, N_comp2, rho1, rho2):
        """
        Calculate gradient of the loss function with respect to A
        
        Returns:
        --------
        G : array, shape (N_var, N_comp)
            Gradient matrix
        """
        N_sub, N_var = X.shape
        N_clust = U.shape[1]
        N_comp = A.shape[1]
        
        # Extract A1 and A2
        A1 = A[:, :N_comp1]
        A2 = A[:, N_comp1:] if N_comp2 > 0 else np.zeros((N_var, 0))
        
        # Identity matrix
        I_N = np.eye(N_sub)
        
        # Calculate projection matrix P_U
        UtU = U.T @ U
        UtU_inv = np.linalg.inv(UtU + 1e-10 * np.eye(N_clust))
        P_U = U @ UtU_inv @ U.T
        
        # Calculate gradients
        # G1 for A1 part
        coef1 = 1 - rho1
        coef2 = rho1 - rho2
        temp_matrix = coef1 * I_N + coef2 * P_U
        G1 = X.T @ temp_matrix @ X @ A1
        
        # G2 for A2 part
        if N_comp2 > 0:
            G2 = X.T @ X @ A2
        else:
            G2 = np.zeros((N_var, 0))
        
        # Combine gradients
        if N_comp2 > 0:
            G = np.hstack([G1, G2])
        else:
            G = G1
        
        # Scale by -2
        G = -2 * G
        
        return G
    
    def gp_algorithm(self, X, U, A, N_comp1, N_comp2, rho1, rho2, 
                     max_iter=100, eps=1e-5, alpha_ini=1.0, max_alpha_iter=15):
        """
        Gradient Projection Algorithm for optimizing A
        
        Parameters:
        -----------
        X : array, Data matrix
        U : array, Cluster membership matrix  
        A : array, Loading matrix (will be modified in-place)
        N_comp1, N_comp2 : int, Component counts
        rho1, rho2 : float, Regularization parameters
        max_iter : int, Maximum iterations
        eps : float, Convergence tolerance
        alpha_ini : float, Initial step size
        max_alpha_iter : int, Maximum line search iterations
        """
        N_var, N_comp = A.shape
        A_current = A.copy()
        
        for iteration in range(max_iter):
            # Calculate gradient
            G = self.gradient(X, U, A_current, N_comp1, N_comp2, rho1, rho2)
            
            # Line search for optimal alpha
            alpha = 2 * alpha_ini
            loss_current = self.loss_function(X, U, A_current, N_comp1, N_comp2, rho1, rho2)
            
            for alpha_iter in range(max_alpha_iter):
                alpha = alpha / 2
                
                # Update A with gradient step
                A_target = A_current - alpha * G
                
                # Project onto Stiefel manifold using SVD
                U_svd, s, Vt_svd = svd(A_target, full_matrices=False)
                A_new = U_svd @ Vt_svd
                
                # Calculate new loss
                loss_new = self.loss_function(X, U, A_new, N_comp1, N_comp2, rho1, rho2)
                
                if loss_new <= loss_current:
                    break
            
            # If no improvement found, set alpha = 0
            if alpha_iter == max_alpha_iter - 1 and loss_new > loss_current:
                alpha = 0
                A_new = A_current.copy()
                loss_new = loss_current
            
            # Check convergence
            diff_loss = loss_current - loss_new
            if diff_loss <= eps:
                break
            
            A_current = A_new.copy()
        
        # Update A in-place
        A[:] = A_current
    
    def optim_grc(self, X, K, N_comp1, N_comp2=0, rho1=0.1, rho2=0.01, 
                  n_random_kmeans=10, max_iter=100, eps=1e-6, verbose=False):
        """
        Main GRC optimization function
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data matrix
        K : int
            Number of clusters
        N_comp1 : int
            Number of components for clustering subspace
        N_comp2 : int, default=0
            Number of additional components
        rho1, rho2 : float
            Regularization parameters
        n_random_kmeans : int
            Number of random k-means initializations
        max_iter : int
            Maximum ALS iterations
        eps : float
            Convergence tolerance
        verbose : bool
            Print progress information
            
        Returns:
        --------
        result : dict
            Dictionary containing:
            - 'A': Loading matrix
            - 'U': Cluster membership matrix
            - 'n_iter': Number of iterations
            - 'loss': Final loss value
        """
        N_sub, N_var = X.shape
        N_comp = N_comp1 + N_comp2
        
        # Initialize A randomly (orthonormal)
        A = np.random.randn(N_var, N_comp)
        A, _ = np.linalg.qr(A)
        
        # Initialize U randomly
        U = np.zeros((N_sub, K))
        random_assignments = np.random.choice(K, N_sub)
        for i, k in enumerate(random_assignments):
            U[i, k] = 1.0
        
        loss = 1e10
        
        # ALS Algorithm
        for iteration in range(max_iter):
            loss_old = loss
            
            # Update U (clustering step)
            if N_comp == N_comp1:
                # Case: A2 is empty
                F1 = X @ A
            else:
                # Case: A2 exists
                A1 = A[:, :N_comp1]
                F1 = X @ A1
            
            # Multiple random k-means initializations
            best_wss = np.inf
            best_labels = None
            
            for random_run in range(n_random_kmeans):
                if iteration > 0 and random_run == 0:
                    # Use previous centroids for first run after iteration 1
                    UtU_inv = np.linalg.inv(U.T @ U + 1e-10 * np.eye(K))
                    initial_centers = UtU_inv @ U.T @ F1
                else:
                    # Random initialization
                    random_indices = np.random.choice(N_sub, K, replace=False)
                    initial_centers = F1[random_indices]
                
                # Run k-means
                labels, centers, wss = self.kmeans_lloyd(F1, initial_centers)
                total_wss = np.sum(wss)
                
                if total_wss < best_wss:
                    best_wss = total_wss
                    best_labels = labels
            
            # Update U from best clustering result
            U = np.zeros((N_sub, K))
            for i, k in enumerate(best_labels):
                U[i, k] = 1.0
            
            # Update A
            if N_comp == N_comp1:
                # Eigendecomposition approach for A2 = empty case
                UtU_inv = np.linalg.inv(U.T @ U + 1e-10 * np.eye(K))
                P_U = U @ UtU_inv @ U.T
                
                # Matrix for eigendecomposition
                coef1 = 1 - rho1
                coef2 = rho1 - rho2
                M = coef1 * np.eye(N_sub) + coef2 * P_U
                
                # Compute X^T M X and find eigenvectors
                XtMX = X.T @ M @ X
                eigenvals, eigenvecs = eigh(XtMX)
                
                # Take largest eigenvalues
                idx = np.argsort(eigenvals)[::-1]
                A = eigenvecs[:, idx[:N_comp]]
            else:
                # Use GP algorithm for A2 ≠ empty case
                self.gp_algorithm(X, U, A, N_comp1, N_comp2, rho1, rho2)
            
            # Calculate loss and check convergence
            loss = self.loss_function(X, U, A, N_comp1, N_comp2, rho1, rho2)
            diff_loss = abs(loss_old - loss)
            
            if verbose:
                print(f"Iteration {iteration + 1}: Loss = {loss:.6f}, Diff = {diff_loss:.6f}")
            
            if diff_loss < eps:
                break
        
        return {
            'A': A,
            'U': U,
            'n_iter': iteration + 1,
            'loss': loss
        }

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    
    # Initialize GRC
    grc = GRC()
    
    # Run GRC algorithm
    print("Running GRC algorithm...")
    result = grc.optim_grc(
        X=X, 
        K=3,           # 3 clusters
        N_comp1=2,     # 2 components for clustering
        N_comp2=1,     # 1 additional component
        rho1=0.1,      # Regularization parameter 1
        rho2=0.01,     # Regularization parameter 2
        n_random_kmeans=5,
        max_iter=50,
        verbose=True
    )
    
    print(f"\nResults:")
    print(f"Final loss: {result['loss']:.6f}")
    print(f"Iterations: {result['n_iter']}")
    print(f"Loading matrix A shape: {result['A'].shape}")
    print(f"Membership matrix U shape: {result['U'].shape}")
    print(f"Cluster sizes: {np.sum(result['U'], axis=0)}")