function [L_hat,S_hat,obj_func_kp1]=admm_algo_OP_on_graphs(X,lambda,gamma,Phi)
%%%%%%%%%%%%
%%% This function implements 'Graph regularized Outlier Pursuit (GOP)' algorithm. 
%%% It is minimizing ||L||_{*} + lambda ||S||_{1,2} +  alpha trace(L Phi L') subject to M=L+S with respect to L and S.
%%% Inputs:
%%% X, is data matrix with dimension (m x n) m is the number of features and n is the number of samples, 
%%% lambda and gamma, are regularization parameters,
%%% Phi, is the Laplacian matrix with dimension (n x n).
%%%
%%% Outputs: 
%%% L_hat, is the low rank matrix at the last iteration. 
%%% S_hat, is the column sparse matrix at the last iteration.
%%% obj_func_kp1, is an array which holds the values of the objective function at each iteration.
%%%%%%%%%%%%

eta=0.000001;
[p , n]=size(X);
L_k=randn(p,n);
W_k=randn(p,n);
S_k=randn(p,n);
S_kmin1=randn(p,n);
W_kmin1=randn(p,n);

Z1_k=X-L_k-S_k;
Z2_k=W_k-L_k;
P1_k=norm_nuclear(L_k);
P2_k=lambda*sum(sqrt(sum(S_k.^2)));
P3_k=gamma*trace(L_k*Phi*L_k');
converged=0;
cnt=0;


r1_k=1;
r2_k=1;
cnt=0;
maxiter=1000;
while (~converged) 
 cnt=cnt+1;        
fprintf('GOP iteration = %d \n', cnt)
H_1=X-S_k+(Z1_k/r1_k);
H_2=W_k+(Z2_k/r2_k);
A=(r1_k*H_1 + r2_k*H_2)/(r1_k + r2_k);
r_k=(r1_k+r2_k)/2;    
    
L_kp1=prox_nuclear_norm(A,1/r_k);
S_kp1=colmn_thresh((X-L_kp1+(Z1_k/r1_k)),lambda/r1_k);
W_kp1=(L_kp1-(Z2_k/r2_k))*(gamma*Phi+r2_k*diag(ones(1,n)))^-1;
Z1_kp1=Z1_k+r1_k*(X-L_kp1-S_kp1);
Z2_kp1=Z2_k+r2_k*(W_kp1-L_kp1);

P1_kp1=norm_nuclear(L_kp1);
P2_kp1=lambda*sum(sqrt(sum(S_kp1.^2)));
P3_kp1=gamma*trace(L_kp1*Phi*L_kp1');

obj_func_kp1(cnt)=P1_kp1+P2_kp1+P3_kp1;

% checking convergence
rel_err_1= norm(P1_kp1-P1_k,'fro')^2/(norm(P1_k,'fro')^2);
rel_err_2=norm(P2_kp1-P2_k,'fro')^2/(norm(P2_k,'fro')^2);
rel_err_3=norm(P3_kp1-P3_k,'fro')^2/(norm(P3_k,'fro')^2); 
rel_err_z1=norm(Z1_kp1-Z1_k,'fro')^2/(norm(Z1_k,'fro')^2);
rel_err_z2=norm(Z2_kp1-Z2_k,'fro')^2/(norm(Z2_k,'fro')^2);

    if(rel_err_1<eta && rel_err_2<eta && rel_err_3<eta && rel_err_z1<eta && rel_err_z2<eta || cnt>=maxiter )
        converged=1;
    else
     
    % get ready for new iteration    
    S_k=S_kp1;
    W_k=W_kp1;
    Z1_k=Z1_kp1;
    Z2_k=Z2_kp1;
    P1_k=P1_kp1;
    P2_k=P2_kp1;
    P3_k=P3_kp1;
  
    end
     
end


L_hat=L_kp1;
S_hat=S_kp1;

end



