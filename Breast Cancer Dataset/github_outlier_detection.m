% initiliaze 
clear all
addpath('./fast_kmeans/');
addpath('./gspbox/');
gsp_start;



[~,text2]=xlsread('breast_TCGA.xlsx');
load('text2.mat')
id_sequed=text2(1,2:end);
load('breast_TCGA.mat');
[num_clinical,t_clin]=xlsread('BRCA_clinical.xlsx');
 
 patient_ids=t_clin(:,1);
 ii=find(strcmp(t_clin(1,:),'ER_Status_nature2012'));
 i_posi=find(strcmp(t_clin(:,8),'Positive')); %% index of ER postive patients 
 i_neg=find(strcmp(t_clin(:,8),'Negative'));
 pos_patients=patient_ids(i_posi);
 neg_patients=patient_ids(i_neg);

 ind_pos=zeros(601,1);
 for j=1:601
 ind_pos(j)=find(strcmp(id_sequed,pos_patients(j)));
 end
 
 ind_neg=zeros(179,1);
 for j=1:179
 ind_neg(j)=find(strcmp(id_sequed,neg_patients(j)));
 end
 X=breast_TCGA;
 
 
 
load('ind_mat1')
 
 
 
 i_pos=300:399;

%% Mahalanobis distance for outlier detection (like fitting gaussian )
Pos=101:105;
mahal=zeros(105,30);
num_dim=[25,50,75,80,95,100,103,200,1000];
FP_mahal_test=zeros(30,numel(num_dim));
r=zeros(30,numel(num_dim));

tic
load('ii.mat')
for k=1:numel(num_dim)
for j=1:30
X_tst=[ X(:,ind_pos(i_pos)) , X(:,ind_neg(ind_mat1(j,:))) ]; %% X_tst is the same matrix  as D_tot   


%%% most varable gnees in test set
v=var(X_tst,0,2);
[b,ib]=sort(v,'descend');
 diff_genes=ib(1:num_dim(k)); 

 X_ts=X_tst(diff_genes,:); 
 
 mu=mean(X_ts,2);
%  r(j,k)=rank(X_ts);
C=cov(X_ts');
C1=C+(1e-9)*eye(numel(diff_genes),numel(diff_genes)); % need to slighlty regularize the covaraince matrix to overcome small numerical artefacts
invC1=inv(C1);
for i=1:105
    y=X_ts(:,i)-mu;
       mahal(i,j)=sqrt(y'*invC1*y);
end

[c,ic]=sort(mahal(:,j),'descend');
    h=find(c==min(mahal(Pos,j)));
    FP_mahal_test(j,k)=length(setdiff(ic(1:h),Pos));
    disp([j,num_dim(k)])
end
end
time=toc;
%%
figure(2)
str=string(num_dim);
boxplot(FP_mahal_test,'label',str)


%% OP and GOP choosing most varaible genes only on test set 

index=1:30;
num_genes=[25,50,80,95,100,200];
n=numel(num_genes);
Pos=101:105;
r_opg=zeros(30,n);
rop=zeros(30,n);
FP_OPG_test=zeros(30,n);
FP_OP_test=zeros(30,n);

 for k=1:n
     
    for j=1:30   
    X_tst=[ X(:,ind_pos(i_pos)) , X(:,ind_neg(ind_mat1(j,:))) ];
    %%% retain highest varaince genes across samples
    v=var(X_tst,0,2);
    [vv,iv]=sort(v,'descend');
    diff_genes=iv(1:num_genes(k));
    X_ts=X_tst(diff_genes,:);
    M=quantilenorm(X_ts);
    param_graph.use_flann = 0;
    param_graph.k = 5; 
    param_graph.type='knn';
    %%%  
    [G,sigma2] = gsp_nn_graph(M',param_graph);
    w=full(G.W);
    d=sum(w,2);
    D=diag(d);
    Lap_graph=D-w;
    min(eig(Lap_graph))

    
   [L_hat2,C_hat2]=admm_algo_OP_on_graphs(M,0.70,1,Lap_graph);
   r_opg(j,k)=rank(L_hat2);

   c_norms_OPG=sqrt(sum(C_hat2.^2));
   [c,ic]=sort(c_norms_OPG,'descend');
   h=find(c==min(c_norms_OPG(Pos)));
   FP_OPG_test(j,k)=length(setdiff(ic(1:h),Pos));
    
   lambda=0.38;
   [L_hat,C_hat]=OUTLIER_PERSUIT(M,lambda);% 0.3
   rop(j,k)=rank(L_hat);
   c_OP=sqrt(sum(C_hat.^2));


   [c,ic]=sort(c_OP,'descend');
   h=find(c==min(c_OP(Pos)));
   FP_OP_test(j,k)=length(setdiff(ic(1:h),Pos));

   disp([j, num_genes(k)])
   
    end 
    
 end


 %%

 figure (11)
 boxplot([FP_OPG_test(:,1),FP_OP_test(:,1),FP_mahal_test(:,1),FP_OPG_test(:,2),FP_OP_test(:,2),FP_mahal_test(:,2),FP_OPG_test(:,4),FP_OP_test(:,4),FP_mahal_test(:,4),FP_OPG_test(:,5),FP_OP_test(:,5),FP_mahal_test(:,5),FP_OPG_test(:,6),FP_OP_test(:,6),FP_mahal_test(:,6),FP_OPG_test(:,8),FP_OP_test(:,8),FP_mahal_test(:,8)],'labels',{'GOP','OP','Mahal','GOP','OP','Mahal','GOP','OP','Mahal','GOP','OP','Mahal','GOP','OP','Mahal','GOP','OP','Mahal'},'labelorientation','inline')
