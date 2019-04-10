function [L_hat,S_hat,obj_func_kp1]=admm_algo_OP_on_graphs(X,lambda,gamma,Phi)

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
mu=10;
cnt=0;
maxiter=1000;
while (~converged) 
cnt=cnt+1                                                                   
H_1=X-S_k+(Z1_k/r1_k);
H_2=W_k+(Z2_k/r2_k);
A=(r1_k*H_1 + r2_k*H_2)/(r1_k + r2_k);
r_k=(r1_k+r2_k)/2;    
    
L_kp1=prox_nuclear_norm(A,1/r_k);
S_kp1=colmn_thresh((X-L_kp1+(Z1_k/r1_k)),lambda/r1_k);
% W_kp1=(L_kp1-(Z2_k/2))*(2*gamma*Phi+ones(n,n))^-1;
W_kp1=(L_kp1-(Z2_k/r2_k))*(gamma*Phi+r2_k*diag(ones(1,n)))^-1;
Z1_kp1=Z1_k+r1_k*(X-L_kp1-S_kp1);
Z2_kp1=Z2_k+r2_k*(W_kp1-L_kp1);

P1_kp1=norm_nuclear(L_kp1);
 P2_kp1=lambda*sum(sqrt(sum(S_kp1.^2)));
P3_kp1=gamma*trace(L_kp1*Phi*L_kp1');


% obj_func_k(cnt)=P1_k+P2_k+P3_k;
obj_func_kp1(cnt)=P1_kp1+P2_kp1+P3_kp1;


% R1_k=X-L_k-S_k;
% R2_k=W_k-L_k;
% S1_k=r1_k*(S_k-S_kmin1);
% S2_k=r2_k*(W_k-W_kmin1);
% 
% if(norm(R1_k,'fro')>mu*norm(S1_k,'fro'))
%   r1_kp1=2*r1_k;
%   Z1_kp1=Z1_kp1/2;
% else if(norm(S1_k,'fro')>mu*norm(R1_k,'fro'))
%         r1_kp1=r1_k/2;
%         Z1_kp1=2*Z1_kp1;
%     else
%         
%     r1_kp1=r1_k;    
%         
%     end
% end
%         
% 
% if(norm(R2_k,'fro')>mu*norm(S2_k,'fro'))
%   r2_kp1=2*r2_k;
%   Z2_kp1=Z2_kp1/2;
% else if(norm(S2_k,'fro')>mu*norm(R2_k,'fro'))
%         r2_kp1=r2_k/2;
%         Z2_kp1=2*Z2_kp1;
%     else
%         
%     r2_kp1=r2_k;    
%         
%     end
% end
% 


rel_err_1= norm(P1_kp1-P1_k,'fro')^2/(norm(P1_k,'fro')^2);
 rel_err_2=norm(P2_kp1-P2_k,'fro')^2/(norm(P2_k,'fro')^2);
 rel_err_3=norm(P3_kp1-P3_k,'fro')^2/(norm(P3_k,'fro')^2); 
 rel_err_z1=norm(Z1_kp1-Z1_k,'fro')^2/(norm(Z1_k,'fro')^2);
rel_err_z2=norm(Z2_kp1-Z2_k,'fro')^2/(norm(Z2_k,'fro')^2);
%  rel_err_1<eta && rel_err_2<eta && rel_err_3<eta && rel_err_z1<eta && rel_err_z2<eta 
% norm(R1_k,'fro')<=0.001 && norm(R2_k,'fro')<=0.001 && norm(S1_k,'fro')<=0.001 && norm(S2_k,'fro')<=0.001
 if(rel_err_1<eta && rel_err_2<eta && rel_err_3<eta && rel_err_z1<eta && rel_err_z2<eta || cnt>=maxiter )
     converged=1;
 else
  % get ready for new iteration    
  S_kmin1=S_k;
  S_k=S_kp1;
  L_k=L_kp1;
  W_kmin1=W_k;
  W_k=W_kp1;
%    r1_k=r1_kp1;
%    r2_k=r2_kp1;
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



