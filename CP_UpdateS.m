function [S] = CP_UpdateS(A,k,XX,Stg,alpha,X,lambda)
    [~,n] = size(A);
    view_num = size(XX,2);
    for i = 1 : view_num
        S{i} = zeros(n,n);
        S_{i} = Stg{i}\(XX{i}+alpha*A);
        S_{i} = S_{i} - diag(diag(S_{i}));
        for j = 1:n
            idx = find(S_{i}(j,:)>0);
            S{i}(j,idx) = EProjSimplex_new(S_{i}(j,idx));
        end 
    end
end
    

