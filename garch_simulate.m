function [ht] = garch_simulate(data,parameter) 

T = length(data);
ht = zeros(T,1);
ht(1) = var(data);

for t = 2:T+1
    ht(t) = parameter(1) + parameter(2)*data(t-1)^2 + parameter(3)*ht(t-1);
end

ht = sqrt(ht); ht = ht(2:end,:);