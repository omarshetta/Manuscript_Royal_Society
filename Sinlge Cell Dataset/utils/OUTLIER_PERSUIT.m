function [L_hat,C_hat,cnt] = OUTLIER_PERSUIT(M,lambda)
%%%%%%%%%%%%
%%% This function implements 'Outlier Pursuit (OP)' algorithm. 
%%% It is minimizing ||L||_{*} + lambda ||C||_{1,2} subject to M=L+C with respect to L and C.
%%% Inputs:
%%% X, is data matrix with dimension (m x n) m is the number of features and n is the number of samples, 
%%% lambda, is a regularization parameters.
%%%
%%% Outputs: 
%%% L_hat, is the low rank matrix at the last iteration. 
%%% C_hat, is the column sparse matrix at the last iteration.
%%% cnt, is an integer variable that counts the number of iterations. 
%%%%%%%%%%%%


[m,n]=size(M);
delta=10e-5;
neta=0.9;
u=0.99*norm(M);
u_dash=delta* u;

L_min0=zeros(m,n);
L_min1=zeros(m,n);
C_min0=zeros(m,n);
C_min1=zeros(m,n);
t_min0=1;
t_min1=1;

converged=0;
cnt=0;
tol=0.000001*norm(M,'fro');

while (~converged)

Y_L=L_min0+((t_min1-1)/t_min0)*(L_min0-L_min1);
Y_C=C_min0+((t_min1-1)/t_min0)*(C_min0-C_min1);
G_L=Y_L-0.5*(Y_L+Y_C-M);
G_C=Y_C-0.5*(Y_L+Y_C-M);

L_new=prox_nuclear_norm(G_L,u/2);
C_new=colmn_thresh(G_C,(lambda*u)/2);

t_new=(1+sqrt(4*t_min0^2+1))/2;
u_new=max(neta*u,u_dash);

% check if converged %
  S_L=2*(Y_L-L_new)+(L_new+C_new-Y_L-Y_C);
  S_C=2*(Y_C-C_new)+(L_new+C_new-Y_L-Y_C);

    if(norm(S_L,'fro')^2 + norm(S_C,'fro')^2 <= tol^2)
        converged=1;
    end
% getting ready for new iteration
L_min1=L_min0;
L_min0=L_new;
C_min1=C_min0;
C_min0=C_new;
t_min1=t_min0;
t_min0=t_new;
u=u_new;
cnt=cnt+1;
fprintf('OP iteration = %d \n', cnt)

end

L_hat=L_new;
C_hat=C_new;

end

