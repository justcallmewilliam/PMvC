function [F] = PMvC(X, label, beta, lambda, gamma,alpha)

%dataset(v): d_v*N(label_m+unlabel_n)
    max_iter = 30;
    stop_condition = 1e-3;
    iter = 1;
    stop_flag = 0;    
    k = 10;
    view_num = size(X,2);
    N = size(X{1},2);
    labeled_N = length(label);
    unlabeled_N = N - length(label);
    %u = 1;
	u = 1e10;


    for i = 1 : view_num
       S{i} = constructW_PKN(X{i});
    end
    A = zeros(N);
    %initialize lambda
    SUM = zeros(N);
    distX_initial = cell(view_num,1);
    for i = 1:view_num
        distX_initial{i} =  L2_distance_1(X{i},X{i}) ;
        SUM = SUM + distX_initial{i};
    end
    distX = 1/view_num*SUM;
    [distXs, ~] = sort(distX,2);
    rr = zeros(N,1);
    for i = 1:N
        di = distXs(i,2:k+2);
        rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
    end
    lambda = mean(rr);
    alpha = mean(rr);

    cls = length(unique(label));
    %get pseudo label
    Y_tar_pseudo = ones(unlabeled_N,cls);
    for i = 1 : view_num
        knn_model = fitcknn(X{i}(:,1:labeled_N)',label,'NumNeighbors',1);
        pseudo_label{i} = knn_model.predict(X{i}(:,labeled_N+1 : end)');
        one_hot_pseu{i} = gen_label_matrix(pseudo_label{i},cls);
        Y_tar_pseudo = Y_tar_pseudo .* one_hot_pseu{i};
        XX{i} = X{i}'*X{i};
        Stg{i} = XX{i}+(lambda+alpha)*eye(N);
    end
    [id_pseudo, ~] = find (Y_tar_pseudo==1);
    U = zeros(N,1);
    U((id_pseudo+labeled_N))=gamma;
    U(1:labeled_N)=u;
    U = diag(U);
    Y= gen_label_matrix(label,cls);
    Y_hat = [Y; Y_tar_pseudo];
    F = rand(N,length(unique(label)));
    F = F*diag(sqrt(1./(diag(F'*F)+eps)));%normalize
    
    [L]= CPget_L(A, k);

    J(1) = compute_obj_fun(X, F, L, S,beta, lambda,alpha,A, U,Y_hat);
    while iter < max_iter %~stop_flag
        iter = iter + 1;
        J0 = J(iter-1);
        %disp('UpdateS');
        S = CP_UpdateS(A,k,XX,Stg,alpha,X,lambda);
        A = CP_UpdateA(X, F,alpha,beta,A,S,k);
        [L]= CPget_L(A,k);
        %disp('UpdateF');
        F = (beta*L+U)\(U*Y_hat);
        F = F*diag(sqrt(1./(diag(F'*F)+eps))); %normalize

        J(iter) = compute_obj_fun(X, F, L, S,beta, lambda,alpha,A, U,Y_hat);
        
        if abs(J(iter) - J0) <= stop_condition
            stop_flag = 1;
        elseif iter > max_iter
            stop_flag = 1;
        end
    end
end

function obj_value = compute_obj_fun(X, F, L, S,beta, lambda,alpha,A, U,Y_hat)
    for i = 1 : size(X,2)
        j1(i) = norm(X{i}-X{i}*S{i},'fro')^2;
        j3(i) = norm(A-S{i},'fro')^2;
        j5(i) = norm(S{i},'fro')^2;
    end
    J1 = sum(j1);
    J2 = beta * trace(F' * L * F);
    J3 = alpha *sum(j3);
    J4 = trace((F-Y_hat)'*U*(F-Y_hat));
    J5 = lambda*sum(j5);
    obj_value = J1 + J2 +J3 + J4+J5;
end

function [Y] = gen_label_matrix(label,cls)
    Y = zeros(length(label), cls);
    for i = 1 : length(label)
       Y(i, label(i)) = 1; 
    end
end

