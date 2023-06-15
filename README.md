# Dynamic vine-GARCH process

Matlab implementation of the vine GARCH model (C-vine) based on the paper:

*Dynamic asset correlations based on vines*, by Benjamin Poignard and Jean-David Fermanian, 2019, Econometric Theory

Link: https://doi.org/10.1017/S026646661800004X

Alternative variance-covariance models are implemented: 
- the scalar DCC model: both full likelihood and composite-likelihood methods are implemented in the second-step objective function. The latter method is based on contiguous overlapping pairs, which builds upon C. Pakel, N. Shephard, K. Sheppard and R.F. Engle (2021) and should be used when the dimension is large (i.e., larger than 200, 300, 400). The DCC-GARCH code builds upon the MFE toolbox of K. Sheppard, https://www.kevinsheppard.com/code/matlab/mfe-toolbox/
- the average oracle estimator (fixed estimator) of C. Bongiorno, D. Challet and G. Loeper (2023).

# Overview

The file *simulations.m* illustrates the implemention of the C-vine GARCH model for a $N$-dimensional vector of observations with/without truncation and is estimated level-by-level in the C-vine tree. Parallel computation is employed in each level of the tree with respect to the corresponding edges, according to Section 3.3 of Poignard and Fermanian (2019).

*simulations.m* provide the a out-of-sample GMVP-based allocation performances for the different variance-covariance models and based on the data generating process (DGP):

$$z_{t} = H_{t}^{1/2} \eta_{t}, \; H_t = D_{t}^{1/2} C_{t} D_{t}^{1/2}, \forall k, \eta_{t,k} = \sqrt{\gamma-2}\nu_{t,k}/\sqrt{\gamma}, \nu_{t,k} \sim t(\gamma),$$

where $t()$ is a centered Student distribution with $\gamma$ degrees of freedom, $\gamma >2$, $D_t^{1/2} = \text{diag}(\sigma_{t,1},\cdots,\sigma_{t,N}) \in \mathbb{R}^N$ is a diagonal matrix containing the marginal conditional volatility processes that follow a GARCH(1,1), and $(C_t)$ denotes the correlation matrix process. Each model (vine GARCH model, scalar DCC, average oracle and sample variance covariance) are estimated for a given in-sample period; the GMVP weights are computed based on the model estimated in-sample; then, the out-of-sample portfolio returns are computed.

Three DGPs are implemented to generate the process $(C_t)$: 

**(A) One-factor model:**

The matrix correlation process is based on the factor model:

$$C_t=(\mathbf{1}(i=j)+\mathbf{1}(i\neq j)a_{it}a_{jt})_{1\leq i,j\leq N}.$$

There are many possibilities for the choice of the factor loadings $(a_{it})$. To generate patterns that are sensitive in financial terms, we propose the following specification:
- a few assets that are highly correlated with the factor, for instance:

    $$a_{it}=0.8-0.1 \sin(2\pi t/\beta), i\in \{1,\ldots,N_1\}.$$
    
- a majority of assets that are reasonably correlated with the factor, for instance:

    $$a_{it}=0.4-0.3 \sin(2\pi t/\beta), i\in\{N_1+1,\ldots,N_1+N_2\}.$$
    
- a minority of assets that are poorly correlated with the factor, for instance:

    $$a_{it}=0.5 \sin(2\pi t/\beta), i\in\{N_1+N_2+1,\ldots,N_1+N_2+N_3=N\}.$$
    
The code uses $N_1=3$ and split the universe into the two other types of assets with the rule $60\%-40\%$.

**(B) Two-factors model:**

The matrix correlation process is based on the factor model:

$$C_t=(\mathbf{1}(i=j)+\mathbf{1}(i\neq j)(a_{it}a_{jt}+b_{it}b_{jt}))_{1\leq i,j\leq N}.$$

It is possible to specify similar factor loadings, by imposing that the maxima of the factor loadings will be smaller than $1/\sqrt{2}$. The code sets:

$$a_{it}=\frac{1}{\sqrt{2}}(0.8-0.1 \sin (2\pi t/\beta_1)), i\in\{1,\ldots,N_1\},$$

$$b_{it}=\frac{1}{\sqrt{2}}(0.8-0.1 \sin (2\pi t/\beta_2)), i\in\{1,\ldots,N_1\},$$

with $\beta_1\neq \beta_2$ possibly (the two systemic factors possibly evolve with different cycles).

**(C) Deterministic varying correlations:** 

Random choice of $N(N-1)/2$ deterministic processes among the cosinus, sinus, modulo and constant functions, and then generate some series

$$a_1 + a_2 \cos(2 \pi t / \alpha), b_1 + b_2 \sin(2 \pi t / \beta), c_1 + c_2 \text{mod}(t / \mu), \, d_1 + d_2 \mathbf{1}_{t>d_3},$$

for every $t=1,\ldots,T$. The parameters $a_1,b_1,c_1,d_1$ (resp. $a_2,b_2,c_2,d_2$) are chosen randomly and independently following a uniform distribution $\mathcal{U}\left(0.01,0.06\right)$ (resp. in the uniform distribution in $(0.3,0.4)$), $d_3$ is uniformly sampled in $1,\ldots,T$, and $\alpha,\beta,\mu$ are randomly (equally) selected among the fixed subset $\{600,700,1000,1200,1400\}$. 

All these series constitute the components of a lower triangular matrix $K_t$ with ones on the main diagonal. Then, we generate symmetric and positive definite matrices 

$$R_t = K_t K^\top_t, C_t = R^{\star-1/2}_t R_t R^{\star-1/2}_t, R_t = (r_{ij,t}), R^{\star}_t = \text{diag}(r_{11,t},\ldots,r_{NN,t})$$

Initializing each of the GARCH processes randomly and given $z_1$, we simulate the successive values of a multivariate GARCH process with conditional covariance matrices $(H_t)$. Finally, we simulate each $\eta_{t,k}$ independently under a $t(3)$ distribution. 

# C-vine selection

The C-vine can be selected using the following three functions:

- *select_Cvine.m*: the method is based on the average conditional kendall's tau non-parametric estimator. The method is time-consuming and
may not be stable when the dimension of the data is larger then 5 or 6. The method is based on formula (3.5) in I. Gijbels, M. Omelka and N. Veraverbeke (2015).

- *select_Cvine_akt.m*: selection of the central node according to an average Kendall's tau measure (AKT): for each variable $z_{kt}, k =1,\cdots,N$,  compute its average Kendall's tau (AKT) with respect to all other variables, that is:

 $$\max\underset{1\leq j\leq N,j\neq k}{\sum}|\widehat\tau_{kj}|.$$ 
 
The variable with the highest AKT is selected as the central node of the first tree; the central node of the second tree is the variable with the second highest AKT; all other central nodes are set according to this criterion. The function should be used for the deterministic correlation process-based DGP.

- *select_Cvine_corrcoeff.m*: selection of the central node according to an average sample linear correlation measure (ALC): for each variable $z_{kt}, k =1,\cdots,N$, compute the average linear correlation coefficient (ALC), that is:

$$\max\underset{1\leq j\leq N,j\neq k}{\sum}|\widehat\rho_{kj}|,$$ 

with $\rho_{kj}$ the linear correlation coefficient. Then, the variable with the highest ALC is selected as the central node of the first tree; the central node of the second tree is the variable with the second highest ALC; all other central nodes are set according to this criterion. The function should be used for the factor-based DGP.

Before runing the main function for the C-vine GARCH model *dynamic_vine.m* on real/simulated data, the C-vine must be selected and the column indices of the data matrix of observations must be re-ordered accordingly (this is done by select_Cvine.m, select_Cvine_akt.m and select_Cvine_corrcoeff.m)

# C-vine GARCH estimation

- *dynamic_vine.m*: main function to estimate the C-vine GARCH model, where the C-vine has been specified by the user or selected by *select_Cvine.m*, *select_Cvine_akt.m* or *select_Cvine_corrcoeff.m*. If the user has specified the C-vine a priori, the ordering of the data matrix of observations is key to run the estimation algorithm: the first column will be the root of the first tree, the second column will be the root of the second tree given the variable in the first column, the third column will be the root in the third tree given the variables in the two first columns, etc.

Full estimation or estimation with truncation can be performed; the estimation is performed edge-by-edge in each tree level, which allows to employ paraellel computation for each tree vine level: see Section 3.3 of Poignard and Fermanian (2019). The truncation level is user-specified: for a given "level" set as input in *dynamic_vine.m*, the algorithm estimates the partial correlation processes up to "level". 

- *corr2partial_Cvine.m*: performs the mapping from the classic correlation matrix to the partial correlation matrix, where the partial correlation structure, i.e., the sets of conditioning and conditioned variables are given by the C-vine structure, which is deduced from the ordering given in the columns of the data matrix of observations
- *partial2corr_Cvine*: performs the mapping from the partial correlation matrix to the classic correlation matrix, where the partial correlation structure, i.e., the sets of conditioning and conditioned variables are given by the C-vine structure, which is deduced from the ordering given in the columns of the data matrix of observations
- *corr2pcorr.m*: performs the mapping from the classic correlation matrix to the partial correlation matrix, where the partial correlation structure, i.e., the sets of conditioning and conditioned variables are given by any arbitrary R-vine structure.
- *pcorr2corr.m*: performs the mapping from the partial correlation matrix to the classic correlation matrix, where the partial correlation structure, i.e., the sets of conditioning and conditioned variables are given by any arbitrary R-vine structure.

Both *corr2pcorr.m* and *pcorr2corr.m* require the so-called vine array, which gives the vine structure. It is a lower triangular matrix/triangular array, with non-zero elements below (including) the main diogonal. See the end of *simulations.m* for examples. See J. Dißmann, E.C. Brechmann, C. Czado and D. Kurowicka (2013), *Selecting and estimating regular vine copulae and application to financial returns*, CSDA, 59, 52-69, for more details on vine array

On the parameter constraints for each partial correlation processes: the user may want to modify *vine_constr_tree1.m* and *vine_contr.m*, which provide the constraints on the Vine GARCH model. In particular, the upper/lower bounds on the constant 'a' and autoregressive parameter 'b' may be modified, depending on the dataset.

# Software requirements

The version of the Matlab software on which the code is implemented is a follows: 9.12.0.1975300 (R2022a) Update 3.

The following toolboxes should be installed:

- Statistics and Machine Learning Toolbox, Version 12.3.
- Parallel Computing Toolbox, Version 7.6. Parallel Computing Toolbox is highly recommended to run the code to speed up the edge-by-edge estimation procedure detailed in Section 3.3 of Poignard and Fermanian (2019).

# References

- C. Bongiorno, D. Challet and G. Loeper (2023), *Filtering time-dependent covariance matrices using time-independent eigenvalues*, Journal of Statistical Mechanics: Theory and Experiment. DOI: 10.1088/1742-5468/acb7ed
- J. Dißmann, E.C. Brechmann, C. Czado and D. Kurowicka (2013), *Selecting and estimating regular vine copulae and application to financial returns*, Computational Statistics & Data Analysis. DOI: 10.1016/j.csda.2012.08.010
- I. Gijbels, M. Omelka and N. Veraverbeke (2015), *Partial and average copulas and association measures*, Electronic Journal of Statistics, DOI: 10.1214/15-EJS1077.
- C. Pakel, N. Shephard, K. Sheppard and R.F. Engle (2021), *Fitting vast dimensional time-varying covariance models*, Journal of Business & Economic Statistics. DOI: 10.1080/07350015.2020.1713795
- B. Poignard and J.D. Fermanian (2019), *Dynamic asset correlations based on vines*, Econometric Theory. DOI: 10.1017/S026646661800004X
