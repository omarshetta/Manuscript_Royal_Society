
% Add paths for functions used in this script
clc
clear all
cd ..
addpath('./fast_kmeans')
addpath('./utils')
cd Single_Cell_Dataset
% Load single cell dataset
disp(['Loading Single Cell Data... '])
load('Test_1_mECS.mat')
X = in_X;


for i = 1:max(unique(true_labs))
p(i) = length(find(true_labs==i));
end

load('ind_outliers.mat')% load indexes of the 6 sampled G2M cells for all 30 datasets.

% Initializing relevant output variables
r_gop = zeros(30,1);
r_op  = zeros(30,1);

perf_GOP  = zeros(30,2);
perf_OP   = zeros(30,2);
perf_PCA  = zeros(30,2);
perf_tsne = zeros(30,2);

separ_GOP  = zeros(30,1);
separ_PCA  = zeros(30,1);
separ_OP   = zeros(30,1);
separ_tsne = zeros(30,1);

F_PCA  = zeros(30,1);
F_GOP  = zeros(30,1);
F_OP   = zeros(30,1);
F_tsne = zeros(30,1);

%%
for j = 1:30
%%% Constructing dataset with 6 G2M cells and 59 G1 cells 
ind_G2M   = find(true_labs==3);
ind_G2M_6 = ind_G2M(ii_mat(j,:));
ind_G1    = find(true_labs==1);
idx       = [ind_G1;ind_G2M_6];
X         = in_X(idx,:);
%%%
%%% Retain most variable genes across samples
v      = var(X);
[b,ib] = sort(v,'descend');
M      = X(:,ib(1:1000));
%%%

%%% Build K-Nearest Neighbours graph.
K=5;
[Lap, ~] = build_knn_graph(M,K); % returns graph Laplacian matrix.
%%%

%%% GOP
[L,C,obj] = admm_algo_OP_on_graphs(M',1.6,2,Lap); % GOP algorithm
[U,~,~]   = svd(L);
r_gop(j)  = rank(L);
Z_GOP = U(:,1:r_gop(j))'*L;
[ cent_Z, pred_Z, ~, ~, ~] = kmeans_fast(Z_GOP', 2, 2, 0);
out_Z = find(pred_Z==2);
[TP_GOP, FP_GOP] = outlier_det_perf(out_Z, 60:65);
perf_GOP(j,:)=[TP_GOP,FP_GOP];

C = cov(Z_GOP');
a = (cent_Z(1,:)-cent_Z(2,:))';
separ_GOP(j) = sqrt(a'*inv(C)*a);


%%% PCA 
[coeff,scores] = pca(M,'NumComponents',r_gop(j));
[ cent_PCA, pred_PCA, ~, ~, ~] = kmeans_fast(scores, 2, 2, 0);
out_PCA = find(pred_PCA==2);
[TP_PCA, FP_PCA] = outlier_det_perf(out_PCA, 60:65);
perf_PCA(j,:) = [TP_PCA,FP_PCA];
 
 C = cov(scores);
 a = (cent_PCA(1,:)-cent_PCA(2,:))';
 separ_PCA(j) = sqrt(a'*inv(C)*a);



%%% Outlier Pursuit
[L_hat, C_hat] = OUTLIER_PERSUIT(M', 0.74); % OP algorithm
r_op(j) = rank(L_hat);
[U,~,~] = svd(L_hat);
Z_op = U(:,1:r_op(j))'*M';
[ cent_Zop, pred_Zop, ~, ~,~] = kmeans_fast(Z_op', 2, 2, 0);
out_Zop = find(pred_Zop==2);
[TP_OP, FP_OP] = outlier_det_perf(out_Zop, 60:65);
perf_OP(j,:)=[TP_OP,FP_OP];
 
C = cov(Z_op');
a = (cent_Zop(1,:)-cent_Zop(2,:))';
separ_OP(j) = sqrt(a'*inv(C)*a);



%%%% t-SNE
y = tsne(M,'NumDimensions',r_gop(j));
[cent_tsne, pred_tsne, ~, ~,~] = kmeans_fast(y, 2, 2, 0);
out_tsne = find(pred_tsne==2);
[TP_tsne, FP_tsne] = outlier_det_perf(out_tsne, 60:65);
perf_tsne(j,:)=[TP_tsne,FP_tsne];

C = cov(y);
a = (cent_tsne(1,:)-cent_tsne(2,:))';
separ_tsne(j) = sqrt(a'*inv(C)*a);

    

end

%% Compute F score 
for i = 1:30
TP   = perf_PCA(i,1);
FP   = perf_PCA(i,2);
Rec  = TP/5;
Prec = TP/(TP+FP);
F_PCA(i) = 2*(Rec * Prec)/( Rec + Prec);
end


for i = 1:30
TP = perf_GOP(i,1);
FP = perf_GOP(i,2);
Rec = TP/5;
Prec = TP/(TP+FP);
F_GOP(i) = 2*(Rec * Prec)/( Rec + Prec);
end

for i = 1:30
TP = perf_OP(i,1);
FP = perf_OP(i,2);
Rec = TP/5;
Prec = TP/(TP+FP);
F_OP(i) = 2*(Rec * Prec)/( Rec + Prec);
end


for i = 1:30
TP = perf_tsne(i,1);
FP = perf_tsne(i,2);
Rec = TP/5;
Prec = TP/(TP+FP);
F_tsne(i) = 2*(Rec * Prec)/( Rec + Prec);
end





%% Plots

% Separability plot
figure (1)
boxplot([separ_OP, separ_GOP, separ_PCA, separ_tsne],  'labels' ,{'OP','GOP','PCA','t-SNE'} )
ylabel('Separability')
grid on
title('Separability Single Cell Dataset')

% F_score plot
figure (2)
boxplot([F_OP,F_GOP,F_PCA,F_tsne],'labels',{'OP','GOP','PCA','t-SNE'} )
ylabel('F Score')
title('F-Score Single Cell Dataset')
%% Visualization of GOP, OP, PCA and t-SNE
 
j = 10; % choosing index of specific dataset

%%% Constructing dataset with 6 G2M cells and 59 G1 cells
ind_G2M = find(true_labs==3);
ind_G2M_6 = ind_G2M(ii_mat(j,:));
ind_G1 = find(true_labs==1);
idx = [ind_G1;ind_G2M_6];
X = in_X(idx,:);
%%%
%%% Retain most variable genes across samples
v = var(X);
[b,ib] = sort(v,'descend');
M = X(:,ib(1:1000));
%%%


%%%% t-SNE

y = tsne(M);
figure (3)
plot(y(1:59,1),y(1:59,2),'ko')
hold on 
plot(y(60:65,1),y(60:65,2),'kx')
title('t-SNE')
legend('G1','G2M')

%%% GOP

%%% Build K-Nearest Neighbours graph.
K=5;
[Lap, ~] = build_knn_graph(M,K); % returns graph Laplacian matrix.
%%%

[L, C, obj] = admm_algo_OP_on_graphs(M', 1.6, 2, Lap); % GOP algorithm
[U,~,~] = svd(L);
r = rank(L);
Z = U(:,1:r)'*L;
figure (4)
plot(Z(1,1:59),Z(2,1:59),'ko')
hold on 
plot(Z(1,60:65),Z(2,60:65),'kx')
legend('G1','G2M')
title('Graph Regularized Outlier Pursuit')
xlabel('PC1')
ylabel('PC2')


%%%PCA
[coeff,scores] = pca(M,'NumComponents',r);
figure (5)
plot(scores(1:59,1),scores(1:59,2),'ko')
hold on 
plot(scores(60:65,1),scores(60:65,2),'kx')
title('PCA')
legend('G1','G2M') 
title('PCA')
xlabel('PC1')
ylabel('PC2')



%%% Outlier pursuit
[L_hat, C_hat] = OUTLIER_PERSUIT(M', 0.74);
r_op = rank(L_hat);

[U,S,V] = svd(L_hat);
s_op(:,j) = diag(S);
Z_op = U(:,1:r_op)'*M';
figure (6)
plot(Z_op(1,1:59),Z_op(2,1:59),'ko')
hold on 
plot(Z_op(1,60:65),Z_op(2,60:65),'kx')
title('Outlier Pursuit')
legend('G1','G2M')
xlabel('PC1')
ylabel('PC2')
 