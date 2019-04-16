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
r_opg=zeros(30,1);
r_op=zeros(30,1);
perf_OPG=zeros(30,2);
perf_OP=zeros(30,2);
perf_PCA=zeros(30,2);
perf_tsne=zeros(30,2);
mahalan_OPG=zeros(30,1);
mahalan_PCA=zeros(30,1);
mahalan_OP=zeros(30,1);
mahalan_tsne=zeros(30,1);


%%
for j=1:30

ind=find(true_labs==3);
iii=ind(ii_mat(j,:));
iii2=find(true_labs==1);

idx=[iii2;iii];
X=in_X(idx,:);
labs=true_labs(idx);

v=var(X);
[b,ib]=sort(v,'descend');
 M=X(:,ib(1:1000));


%%% build k-NN graph
param_graph.use_flann = 0;
param_graph.k = 5; % 20 
param_graph.type='knn';
G1 = gsp_nn_graph(M,param_graph);
%%%


%%% OPG 
[L,C,obj]=admm_algo_OP_on_graphs(M',1.6,2,full(G1.L)); % 1.5 ,5 
[U,~,~]=svd(L);
r_opg(j)=rank(L);
Z=U(:,1:r_opg(j))'*L;
[ cent_Z, pred_Z, ~,~,~ ] = kmeans_fast(Z',2,2,0);
out_Z=find(pred_Z==2);
[T_GRPCA,F_GRPCA]=outlier_det_perf(out_Z,60:65);
perf_OPG(j,:)=[T_GRPCA,F_GRPCA];

C=cov(Z');
a=(cent_Z(1,:)-cent_Z(2,:))';
mahalan_OPG(j)=sqrt(a'*inv(C)*a);


%%% PCA 
[coeff,scores]=pca(M,'NumComponents',r_opg(j));
[ cent_PCA, pred_PCA, ~,~,~ ] = kmeans_fast(scores,2,2,0);
out_PCA=find(pred_PCA==2);
[T_PCA,F_PCA]=outlier_det_perf(out_PCA,60:65);
perf_PCA(j,:)=[T_PCA,F_PCA];
 
 C=cov(scores);
 a=(cent_PCA(1,:)-cent_PCA(2,:))';
 mahalan_PCA(j)=sqrt(a'*inv(C)*a);



%%% Outlier pursuit
[L_hat,C_hat]=OUTLIER_PERSUIT(M',0.74);
r_op(j)=rank(L_hat);
[U,~,~]=svd(L_hat);
Z_op=U(:,1:r_op(j))'*M';
[ cent_Zop, pred_Zop, ~,~,~ ] = kmeans_fast(Z_op',2,2,0);
out_Zop=find(pred_Zop==2);
[T_OP,F_OP]=outlier_det_perf(out_Zop,60:65);
perf_OP(j,:)=[T_OP,F_OP];
 
C=cov(Z_op');
a=(cent_Zop(1,:)-cent_Zop(2,:))';
mahalan_OP(j)=sqrt(a'*inv(C)*a);



%%%% 
y=tsne(M,'NumDimensions',2);
[ cent_tsne, pred_tsne, ~,~,~ ] = kmeans_fast(y,2,2,0);
out_tsne=find(pred_tsne==2);
[T_tsne,F_tsne]=outlier_det_perf(out_tsne,60:65);
perf_tsne(j,:)=[T_tsne,F_tsne];

C=cov(y);
a=(cent_tsne(1,:)-cent_tsne(2,:))';
mahalan_tsne(j)=sqrt(a'*inv(C)*a);

    

end

%% F1 score 
F1_PCA=zeros(30,1);
for i=1:30
TP=perf_PCA(i,1);
FP=perf_PCA(i,2);
Rec=TP/5;
Prec=TP/(TP+FP);
F1_PCA(i)=2*(Rec * Prec)/( Rec + Prec);
end

F1_OPG=zeros(30,1);
for i=1:30
TP=perf_OPG(i,1);
FP=perf_OPG(i,2);
Rec=TP/5;
Prec=TP/(TP+FP);
F1_OPG(i)=2*(Rec * Prec)/( Rec + Prec);
end

F1_OP=zeros(30,1);
for i=1:30
TP=perf_OP(i,1);
FP=perf_OP(i,2);
Rec=TP/5;
Prec=TP/(TP+FP);
F1_OP(i)=2*(Rec * Prec)/( Rec + Prec);
end


F1_tsne=zeros(30,1);
for i=1:30
TP=perf_tsne(i,1);
FP=perf_tsne(i,2);
Rec=TP/5;
Prec=TP/(TP+FP);
F1_tsne(i)=2*(Rec * Prec)/( Rec + Prec);
end

figure (3)
boxplot([F1_tsne,F1_PCA,F1_OP,F1_OPG],'labels',{'tsne','PCA','OP','OPG'})



%% separability plot

figure (103)
boxplot([mahalan_OP, mahalan_OPG, mahalan_PCA, mahalan_tsne],  'labels' ,{'OP','GOP','PCA','t-SNE'} )
ylabel('Separability')
grid on 

 