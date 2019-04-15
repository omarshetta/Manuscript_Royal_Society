close all 
clear all
clc
addpath('./gspbox/');

gsp_start;

load('Test_1_mECS.mat')
X=in_X;

for i=1:max(unique(true_labs))
p(i)= length(find(true_labs==i));
end

load('ii_mat2.mat')
num_feat=[20,30,50,70,100,300,600];
c_OP=zeros(30,33);
c_OPG=zeros(30,33);
FP_c_OPG=zeros(30,1);
FP_c_OP=zeros(30,1);
FP_kernel=zeros(30,1);



%% mahal dist (Gaussian)  (59G1 and 6G2M ), used to find  most varaible genes  

% num_genes=[2,10,20,30,50];
num_genes=[2,20,30,50,60,70];
FP_mahal_most_var=zeros(30,numel(num_genes));
 mahal=zeros(65,30);
Pos=60:65;

for k=1:numel(num_genes)
for j=1:30

ind=find(true_labs==3);
iii=ind(ii_mat(j,:));
iii2=find(true_labs==1);

idx=[iii2;iii];
X=in_X(idx,:);
labs=true_labs(idx);



Xt=X';

% XX=in_X([ind;iii2],:);
v=var(X);
[b,ib]=sort(v,'descend');
 diff_genes=ib(1:num_genes(k));
 
%  X_train=X_pos(diff_genes,:);
 X_ts=Xt(diff_genes,:);

 
 
mu=mean(X_ts,2);
 
C=cov(X_ts');
C1=C+(1e-10)*eye(numel(diff_genes),numel(diff_genes));
invC1=inv(C1);
for i=1:65
    y=X_ts(:,i)-mu;
       mahal(i,j)=sqrt(y'*invC1*y);
%   mahal(i,j)=sqrt(y'*(C1\y));
end

[c,ic]=sort(mahal(:,j),'descend');
    h=find(c==min(mahal(Pos,j)));
   FP_mahal_most_var(j,k)=length(setdiff(ic(1:h),Pos));
    disp([j,num_genes(k)])
end
end
figure (3)
boxplot(FP_mahal_most_var)

figure (4)
plot(mahal(:,1),'bo')


%% OP GOP  (59G1 and 6G2M ), used to find  most varaible genes  


num_genes=[2,20,30,50,60,70];
FP_c_OPG2=zeros(30,numel(num_genes));
FP_OP2=zeros(30,numel(num_genes));
 r_OPG=zeros(30,numel(num_genes));
rop=zeros(30,numel(num_genes));
% lambda=[0.23,0.4,0.63,0.70,0.60];
Pos=60:65;
for k=1:numel(num_genes)
for j=1:30

ind=find(true_labs==3);
iii=ind(ii_mat(j,:));
iii2=find(true_labs==1);

idx=[iii2;iii];
X=in_X(idx,:);
labs=true_labs(idx);


v=var(X);
[b,ib]=sort(v,'descend');
 diff_genes=ib(1:num_genes(k));
 
 Xt=X';
 X_ts=Xt(diff_genes,:);

 param_graph.use_flann = 0; % to use the fast C++ library for construction of huge graphs.
param_graph.k = 3; % 4 for 200 genes 
param_graph.type='knn';

% M=log(X_var+1);
  M=X_ts;
% figure (j)
% plot(svd(M),'bo');

G1 = gsp_nn_graph(M',param_graph);

if(k==1)
    
lambda_opg=0.35;  

else
    lambda_opg=0.70; %0.75
end
[L,C,obj]=admm_algo_OP_on_graphs(M,lambda_opg,1,full(G1.L)); % lambda=0.8 , gamma=1
r_OPG(j,k)=rank(L);
c_OPG=sqrt(sum(C.^2));
 [c,ic]=sort(c_OPG,'descend');
 h=find(c==min(c_OPG(Pos)));
FP_c_OPG2(j,k)=length(setdiff(ic(1:h),Pos));


if(k==1)
    
lambda_op=0.16;%0.16  

else
    lambda_op=0.3;
end
  [L_hat,C_hat]=OUTLIER_PERSUIT(M,lambda_op);% 0.3
      rop(j,k)=rank(L_hat);
      c_OP=sqrt(sum(C_hat.^2));

     [c,ic]=sort(c_OP,'descend');
     h=find(c==min(c_OP(Pos)));
    FP_OP2(j,k)=length(setdiff(ic(1:h),Pos));



end
end
%%

figure (3)
boxplot([FP_c_OPG2(:,1),FP_OP2(:,1),FP_mahal_most_var(:,1),FP_c_OPG2(:,2),FP_OP2(:,2),FP_mahal_most_var(:,2),FP_c_OPG2(:,3),FP_OP2(:,3),FP_mahal_most_var(:,3),FP_c_OPG2(:,4),FP_OP2(:,4),FP_mahal_most_var(:,4),FP_c_OPG2(:,5),FP_OP2(:,5),FP_mahal_most_var(:,5),FP_c_OPG2(:,6),FP_OP2(:,6),FP_mahal_most_var(:,6)],'labels',{'GOP','OP','Mahal','GOP','OP','Mahal','GOP','OP','Mahal','GOP','OP','Mahal','GOP','OP','Mahal','GOP','OP','Mahal'},'labelorientation','inline');
% save FP_GOP_OP_mahal_most_var_genes_on_testset_sing_cell.mat FP_c_OPG2 FP_OP2 FP_mahal_most_var