function [L_hat,C_hat,cnt] = OUTLIER_PERSUIT(M,lambda)
[m,n]=size(M);
 delta=10e-5;
% delta=0.00001;
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
[U,S,V]=svd(G_L,'econ');
shrinked=soft_thresh(S,u/2);
L_new=U*shrinked*V';
C_new=colmn_thresh(G_C,(lambda*u)/2);
t_new=(1+sqrt(4*t_min0^2+1))/2;
u_new=max(neta*u,u_dash);

% check if converged %
  S_L=2*(Y_L-L_new)+(L_new+C_new-Y_L-Y_C);
  S_C=2*(Y_C-C_new)+(L_new+C_new-Y_L-Y_C);

 if(norm(S_L,'fro')^2 + norm(S_C,'fro')^2 <= tol^2)
  converged=1;
 end

%  if(norm(L_new+C_new-M,'fro')^2 <= 0.05)
%   converged=1;
%  end


 L_min1=L_min0;
 L_min0=L_new;
 C_min1=C_min0;
 C_min0=C_new;
 t_min1=t_min0;
 t_min0=t_new;
 u=u_new;
 cnt=cnt+1

end

L_hat=L_new;
C_hat=C_new;

end

