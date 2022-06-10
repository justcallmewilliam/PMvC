function [A] = CP_UpdateA(X, F,alpha,beta,A,S,K)
    [n,~] = size(F);
    P = L2_distance(F',F');
    S_sum = 0;
    for i = 1 : size(X,2)
        S_sum = S_sum + S{i};
    end
    temp = 1/size(X,2)*(S_sum - beta/(2*alpha)*P);  
    for j = 1:n
        [~, idx0] = sort(temp(j,:),'descend');
        idx = idx0(1:K);
        A(j,idx) = EProjSimplex_new(temp(j,idx));
    end 
end
    


