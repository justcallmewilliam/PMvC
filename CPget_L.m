function [L] = CPget_L(A,K)
    numOfSamples = size(A, 1);
    L = zeros(size(A));
    W = zeros(size(A));
    W = W + A;
    DN = diag( 1./sqrt(sum(A, 2)+eps) );
    LapN = speye(numOfSamples) - DN * A * DN;
    L = L + LapN;
end
