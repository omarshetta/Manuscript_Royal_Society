% initiliaze 
clear all
addpath('./fast_kmeans/');
addpath('./gspbox/');
gsp_start;

[~,text2]=xlsread('breast_TCGA.xlsx'); 
load('text2.mat')
id_sequed=text2(1,2:end);
load('breast_TCGA.mat');
[num_clinical,t_clin]=xlsread('BRCA_clinical.xlsx');% need to download from TCGA repository and convert to excel spreadsheet
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


%% 

load('i_pos')
load('ind_outliers')
perf_OPG=zeros(30,2);
perf_PCA=zeros(30,2);
perf_tsne=zeros(30,2);
perf_OP=zeros(30,2);
r2=zeros(30,1);
r_OP=zeros(1,30);
mal_OPG=zeros(30,1);
mal_PCA=zeros(30,1);
mal_OP=zeros(30,1);
mahal_tsne=zeros(30,1);

F1_PCA=zeros(30,1);
F1_OPG=zeros(30,1);
F1_tsne=zeros(30,1);
F1_OP=zeros(30,1);


for j=1:30

X_tst=[ X(:,ind_pos(i_pos)) , X(:,ind_neg(ind_mat1(j,:))) ]; %% X_tst is the same matrix  as D_tot 
v=var(X_tst,0,2);
[b,ib]=sort(v,'descend');
diff_genes=ib(1:2000); 
X_ts=X_tst(diff_genes,:); 

Pos=101:105;

qq_var=quantilenorm(X_ts);


param_graph.use_flann = 0; % to use the fast C++ library for construction of huge graphs.
param_graph.k = 5; 
param_graph.type='knn';


G = gsp_nn_graph(qq_var',param_graph);
w=full(G.W);
d=sum(w,2);
D=diag(d);
Lap_graph=D-w;

[L_hat2,C_hat2,obj]=admm_algo_OP_on_graphs(qq_var,0.84,1,Lap_graph);% lambda_vec(25)
r2(j)=rank(L_hat2);
[U2,~,~]=svd(L_hat2);

Z_OPG=U2(:,1:r2(j))'*L_hat2;
[ cent_OPG, predOPG1, ~,~,~ ] = kmeans_fast(Z_OPG' ,2,2,0);
C=cov(Z_OPG');
y=(cent_OPG(1,:)-cent_OPG(2,:))';
mal_OPG(j)=sqrt(y'*inv(C)*y);

%%%% OPG 
out_OPG1=find(predOPG1==2);
[TP_OPG1,FP_OPG1]=outlier_det_perf(out_OPG1,Pos);
perf_OPG(j,:)=[TP_OPG1,FP_OPG1];
%%%%

%%% PCA 
for i=1:size(qq_var,1)
qq_var_cent(i,:)=qq_var(i,:)-mean(qq_var(i,:));    
end
[U,~,~]=svd(qq_var_cent);
z=U(:,1:r2(j))'*qq_var_cent;
[ cent_PCA, pred_pca, ~,~,~ ] = kmeans_fast(z' ,2,2,0);
out_pca=find(pred_pca==2);
[TP_pca,FP_pca]=outlier_det_perf(out_pca,Pos);
perf_PCA(j,:)=[TP_pca,FP_pca];
 %%% malhanobis distance for cent_PCA
 C=cov(z');
 y=(cent_PCA(1,:)-cent_PCA(2,:))';
 mal_PCA(j)=sqrt(y'*inv(C)*y);
 %%%
 
 
[L_hat,C_hat] = OUTLIER_PERSUIT(qq_var,0.38);
r_OP(j)=rank(L_hat);
[U,~,~]=svd(L_hat);

z_OP=U(:,1:r_OP(j))'*qq_var;
[ cent_OP, pred_OP, ~,~,~ ] = kmeans_fast(z_OP' ,2,2,0);
out_OP=find(pred_OP==2);
[TP_OP,FP_OP]=outlier_det_perf(out_OP,Pos);
perf_OP(j,:)=[TP_OP,FP_OP];
 %%% malhanobis distance for cent_PCA
 C=cov(z_OP');
 y=(cent_OP(1,:)-cent_OP(2,:))';
 mal_OP(j)=sqrt(y'*inv(C)*y);
 
 j
% % %  %%% tsne 
 y=tsne(qq_var','NumDimensions',r2(j));
 [ cent_tsne, pred_tsne, ~,~,~ ] = kmeans_fast(y,2,2,0);
 out_tsne=find(pred_tsne==2);
 [T_tsne,F_tsne]=outlier_det_perf(out_tsne,Pos);
 perf_tsne(j,:)=[T_tsne,F_tsne];
 %%%  mahal dist tsne
 C=cov(y);
 a=(cent_tsne(1,:)-cent_tsne(2,:))';
 mahal_tsne(j)=sqrt(a'*inv(C)*a);


%%%% F1 score
TP=perf_PCA(j,1);
FP=perf_PCA(j,2);
Rec=TP/5;
Prec=TP/(TP+FP);
F1_PCA(j)=2*(Rec * Prec)/( Rec + Prec);

TP=perf_OPG(j,1);
FP=perf_OPG(j,2);
Rec=TP/5;
Prec=TP/(TP+FP);
F1_OPG(j)=2*(Rec * Prec)/( Rec + Prec);


TP=perf_OP(j,1);
FP=perf_OP(j,2);
Rec=TP/5;
Prec=TP/(TP+FP);
F1_OP(j)=2*(Rec * Prec)/( Rec + Prec);

TP=perf_tsne(j,1);
FP=perf_tsne(j,2);
Rec=TP/5;
Prec=TP/(TP+FP);
F1_tsne(j)=2*(Rec * Prec)/( Rec + Prec);

end


figure (40)
boxplot([mal_OP,mal_OPG,mal_PCA, mahal_tsne],'labels',{'OP','GOP','PCA','t-SNE'})
grid on
figure (41)
boxplot([F1_OP,F1_OPG,F1_PCA,F1_tsne],'labels',{'OP','GOP','PCA','t-SNE'})
ylabel('F-score')
grid on



%% Visulization 
j=10;
X_tst=[ X(:,ind_pos(i_pos)) , X(:,ind_neg(ind_mat1(j,:))) ];

%%% retain highest varaince genes
v=var(X_tst,0,2);
[b,ib]=sort(v,'descend');
diff_genes=ib(1:2000); 
X_ts=X_tst(diff_genes,:); 
qq_var=quantilenorm(X_ts);

param_graph.use_flann = 0;
param_graph.k = 5; 
param_graph.type='knn';


Pos=101:105;

G = gsp_nn_graph(qq_var',param_graph);
w=full(G.W);
d=sum(w,2);
D=diag(d);
Lap_graph=D-w;

%%% OP
 [L_hat,C_hat,cnt] = OUTLIER_PERSUIT(qq_var,0.40);
[U,~,~]=svd(L_hat);
z_OP=U(:,1:2)'*qq_var;
figure (35)
plot(z_OP(1,1:100),z_OP(2,1:100),'ko')
hold on 
plot(z_OP(1,Pos),z_OP(2,Pos),'kd')
title('Outlier Pursuit')
xlabel('PC1')
ylabel('PC2')
legend('ER+','ER-')

%%% GOP
[L_hat2,C_hat2,obj]=admm_algo_OP_on_graphs(qq_var,0.84,1,Lap_graph);
r2=rank(L_hat2);
[U2,S2,V2]=svd(L_hat2);
Z_OPG=U2(:,1:r2)'*L_hat2;
figure (36)
plot(Z_OPG(1,1:99),Z_OPG(2,1:99),'ko')
hold on 
plot(Z_OPG(1,Pos),Z_OPG(2,Pos),'kd')
title('GOP')
xlabel('PC1')
ylabel('PC2')
legend('ER+','ER-')
c_norms_GOP=sqrt(sum(C_hat2.^2));
figure(101)
plot(c_norms_GOP,'bo')
hold on 
plot(Pos,c_norms_GOP(Pos),'ro')



%%% T-sne
figure (37)
y=tsne(qq_var');
plot(y(1:100,1),y(1:100,2),'ko')
hold on 
plot(y(101:105,1),y(101:105,2),'kd')
title('t-SNE')
legend('ER+','ER-')


%%%PCA
qq_var_cent=zeros(size(qq_var,1),size(qq_var,2));
for i=1:size(qq_var,1)
qq_var_cent(i,:)=qq_var(i,:)-mean(qq_var(i,:));    
end
[U,~,~]=svd(qq_var_cent);
z=U(:,1:r2)'*qq_var_cent;
figure (38)
plot(z(1,1:99),z(2,1:99),'ko')
hold on 
plot(z(1,Pos),z(2,Pos),'kd')
title('PCA')
xlabel('PC1')
ylabel('PC2')
legend('ER+','ER-')

