function [Matrix,oldOrder] = reorderRVineMatrix(Matrix)

oldOrder = diag(Matrix); d = size(Matrix,2); Check = vec(Matrix);

Object = zeros(d^2,d);
for k=1:d^2
    for l=1:d
        if (Check(k)==l)
            Object(k,l) = 1;
        else
            Object(k,l) = 0;
        end
    end
end
O = logical(Object);
for i=1:d
    Matrix(O(:,oldOrder(i))') = d-i+1;
end
