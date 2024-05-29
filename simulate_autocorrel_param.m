function [alpha,beta,zeta] = simulate_autocorrel_param(N)

dim = N*(N-1)/2;
alpha = 0.001+(0.05-0.001)*rand(dim,1);
cond = true; zeta = zeros(dim,1);
while cond
    beta = 0.8+(0.9-0.8)*rand(dim,1);
    for l=1:dim
        cond_dist=true;
        while cond_dist
            param = -0.1+(0.1-(-0.1))*rand(1);
            cond_dist=(param>-0.01 & param<0.01);
        end
        zeta(l) = param;
    end
    cond = (any(abs(beta+zeta) > 1));
end