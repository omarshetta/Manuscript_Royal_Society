%%% This code uses GSPBOX for graph construction. PLEASE DO NOT FORGET to download GSPBOX! as described in the READ_ME file in parent directory.

%%% Download breast cancer TCGA dataset from Xena browser following
%%% instructions in 'Download Breast cancer TCGA dataset.txt' file. Download file then convert to
%%% excel and rename as 'BRCA_TCGA.xlsx'
clc
clear all
disp(['Loading Breast Cancer Dataset...'])
[breast_TCGA,text2] = xlsread('BRCA_TCGA.xlsx');% need to download from TCGA repository and convert to excel spreadsheet 
id_sequed = text2(1,2:end);


% Initialize gspbox library and add path for functions used in this script
cd ..
addpath('./fast_kmeans/');
addpath('./utils')
addpath('./gspbox/');
gsp_start;
cd Breast_Cancer_Dataset

% Load BRCA clinical data, find ER+ and ER- patients 
[num_clinical,t_clin] = xlsread('BRCA_clinical.xlsx');
patient_ids = t_clin(:,1);
ii = find(strcmp(t_clin(1,:),'ER_Status_nature2012'));
i_posi = find(strcmp(t_clin(:,8),'Positive')); %% index of ER postive patients 
i_neg = find(strcmp(t_clin(:,8),'Negative'));
pos_patients = patient_ids(i_posi);
neg_patients = patient_ids(i_neg);

 ind_pos = zeros(601,1);
 for j = 1:601
 ind_pos(j) = find(strcmp(id_sequed,pos_patients(j)));
 end
 
 ind_neg = zeros(179,1);
 for j = 1:179
 ind_neg(j) = find(strcmp(id_sequed,neg_patients(j)));
 end
 X = breast_TCGA;


%% Compare GOP, OP, PCA and t-SNE F-score and separability

load('i_pos') % index of 100 ER+ samples in the total of 600 ER+.
load('ind_outliers') % indexes of 5 ER- samples in the total of 179 ER-.   
perf_GOP   = zeros(30,2);
perf_PCA   = zeros(30,2);
perf_tsne  = zeros(30,2);
perf_OP    = zeros(30,2);
r2         = zeros(30,1);
r_OP       = zeros(1,30);
separ_GOP  = zeros(30,1);
separ_PCA  = zeros(30,1);
separ_OP   = zeros(30,1);
separ_tsne = zeros(30,1);

F_PCA  = zeros(30,1);
F_GOP  = zeros(30,1);
F_tsne = zeros(30,1);
F_OP   = zeros(30,1);

Pos = 101:105;

for j = 1:30

X_tst = [ X(:,ind_pos(i_pos)) , X(:,ind_neg(ind_mat1(j,:))) ]; % constructiong 100 ER+ and 5 ER- dataset.

%%% Filter dataset to 2000 most variable genes
v          = var(X_tst,0,2);
[b,ib]     = sort(v,'descend');
diff_genes = ib(1:2000); 
X_ts       = X_tst(diff_genes,:); 


%Quantile normalize dataset
qq_var = quantilenorm(X_ts);

% Set k-Nearest Neighbour graph parameters. Check GSPBOX documentation for more details
param_graph.use_flann = 0; % to use the fast C++ library for construction of huge graphs.
param_graph.k = 5; % setting k=5
param_graph.type = 'knn'; % specifying type of graph

% Find W (square and symmetric similarity matrix) and find Laplacian matrix
G = gsp_nn_graph(qq_var',param_graph); % create k-NN graph
w = full(G.W);
d = sum(w,2);
D = diag(d);
Lap_graph = D-w;

[L_hat2,C_hat2,obj] = admm_algo_OP_on_graphs(qq_var, 0.84, 1, Lap_graph); % GOP algorithm 
r2(j) = rank(L_hat2);

% Project L_hat onto its first r2(j) principal directions.
[U2,~,~] = svd(L_hat2);
Z_GOP = U2(:,1:r2(j))'*L_hat2;

% Find two clusters with k-means then : 1) Find the Mahalanobis distance between
% Cluster centers to measure separability. 2) Find F_score of outlier detection using the minority cluster as the outliers

[ cent_GOP, predGOP, ~,~,~ ] = kmeans_fast(Z_GOP', 2, 2, 0);% k-means with two cluster centers

%%%% Separability GOP
C = cov(Z_GOP');
y = (cent_GOP(1,:)-cent_GOP(2,:))';
separ_GOP(j) = sqrt(y'*inv(C)*y);
%%%%

%%%% F_score GOP 
out_GOP = find(predGOP==2);
[TP_GOP, FP_GOP] = outlier_det_perf(out_GOP, Pos);
perf_GOP(j,:) = [TP_GOP,FP_GOP]; 
TP = perf_GOP(j,1);
FP = perf_GOP(j,2);
Rec = TP/5;
Prec = TP/(TP+FP);
F_GOP(j) = 2*(Rec * Prec)/( Rec + Prec);
%%%%

%%% PCA 
for i = 1:size(qq_var,1)
qq_var_cent(i,:) = qq_var(i,:)-mean(qq_var(i,:));    
end
[U,~,~] = svd(qq_var_cent);
z = U(:,1:r2(j))'*qq_var_cent;
%%%

[cent_PCA, pred_pca, ~, ~, ~] = kmeans_fast(z', 2, 2, 0);  % k-means with two cluster centers

%%% Separability for PCA
C = cov(z');
y = (cent_PCA(1,:)-cent_PCA(2,:))';
separ_PCA(j) = sqrt(y'*inv(C)*y);
%%%
 
%%% F_score PCA 
out_pca = find(pred_pca==2);
[TP_pca, FP_pca] = outlier_det_perf(out_pca, Pos);
perf_PCA(j,:) = [TP_pca,FP_pca];
TP = perf_PCA(j,1);
FP = perf_PCA(j,2);
Rec = TP/5;
Prec = TP/(TP+FP);
F_PCA(j) = 2*(Rec * Prec)/( Rec + Prec); 
%%% 


%%% OP
[L_hat,C_hat] = OUTLIER_PERSUIT(qq_var,0.38);
r_OP(j) = rank(L_hat);
%%% 
% Project L_hat onto its first r_OP(j) principal directions.
[U,~,~] = svd(L_hat);
z_OP = U(:,1:r_OP(j))'*qq_var;

[cent_OP, pred_OP, ~, ~, ~] = kmeans_fast(z_OP', 2, 2, 0); % % k-means with two cluster centers

%%% Separabiliy OP
C = cov(z_OP');
y = (cent_OP(1,:)-cent_OP(2,:))';
separ_OP(j) = sqrt(y'*inv(C)*y);
%%%

%%% F_score OP
out_OP = find(pred_OP==2);
[TP_OP, FP_OP] = outlier_det_perf(out_OP, Pos);
perf_OP(j,:) = [TP_OP,FP_OP];
TP = perf_OP(j,1);
FP = perf_OP(j,2);
Rec = TP/5;
Prec = TP/(TP+FP);
F_OP(j) = 2*(Rec * Prec)/( Rec + Prec); 
%%%

%%% t-SNE
y = tsne(qq_var', 'NumDimensions', r2(j));
%%% 

[cent_tsne, pred_tsne, ~, ~, ~] = kmeans_fast(y, 2, 2, 0);  % k-means with two cluster centers

%%%  Separability t-SNE
C = cov(y);
a = (cent_tsne(1,:)-cent_tsne(2,:))';
separ_tsne(j) = sqrt(a'*inv(C)*a);
%%%

%%% F_score t-SNE
out_tsne = find(pred_tsne==2);
[TP_tsne, FP_tsne] = outlier_det_perf(out_tsne, Pos);
perf_tsne(j,:) = [TP_tsne,FP_tsne];
TP = perf_tsne(j,1);
FP = perf_tsne(j,2);
Rec = TP/5;
Prec = TP/(TP+FP);
F_tsne(j) = 2*(Rec * Prec)/( Rec + Prec);
%%% 

fprintf('dataset = %d \n',j)

end


figure (1)
boxplot([separ_OP,separ_GOP,separ_PCA, separ_tsne],'labels',{'OP','GOP','PCA','t-SNE'})
grid on
title('Separabilty Breast Cancer Dataset')

figure (2)
boxplot([F_OP,F_GOP,F_PCA,F_tsne],'labels',{'OP','GOP','PCA','t-SNE'})
ylabel('F-score')
grid on
title('F score Breast Cancer Dataset')


%% Visualization of GOP, OP, PCA and t-SNE.

j = 10;
X_tst = [ X(:,ind_pos(i_pos)) , X(:,ind_neg(ind_mat1(j,:))) ]; % constructiong 100 ER+ and 5 ER- dataset.

%%% Retain 200 most variable genes
v          = var(X_tst,0,2);
[b,ib]     = sort(v,'descend');
diff_genes = ib(1:2000); 
X_ts       = X_tst(diff_genes,:);

qq_var = quantilenorm(X_ts);% Quantile normalize

%%% Initlilze k-Nearest Neighbour graph
param_graph.use_flann = 0;
param_graph.k = 5; 
param_graph.type='knn';


Pos = 101:105;

% Find W (square and symmetric similarity matrix) and find Laplacian matrix
G = gsp_nn_graph(qq_var',param_graph);
w = full(G.W);
d = sum(w,2);
D = diag(d);
Lap_graph = D-w;

%%% OP
[L_hat, C_hat, cnt] = OUTLIER_PERSUIT(qq_var, 0.40);
[U,~,~] = svd(L_hat);
z_OP = U(:,1:2)'*qq_var;

figure (3)
plot(z_OP(1,1:100),z_OP(2,1:100),'ko')
hold on 
plot(z_OP(1,Pos),z_OP(2,Pos),'kx')
title('Outlier Pursuit')
xlabel('PC1')
ylabel('PC2')
legend('ER+','ER-')

%%% GOP
[L_hat2, C_hat2, obj]=admm_algo_OP_on_graphs(qq_var, 0.84, 1, Lap_graph);
r2 = rank(L_hat2);
[U2, S2, V2] = svd(L_hat2);
Z_GOP = U2(:,1:r2)'*L_hat2;

figure (4)
plot(Z_GOP(1,1:99),Z_GOP(2,1:99),'ko')
hold on 
plot(Z_GOP(1,Pos),Z_GOP(2,Pos),'kx')
title('GOP')
xlabel('PC1')
ylabel('PC2')
legend('ER+','ER-')



%%% t-SNE
figure (5)
y = tsne(qq_var');
plot(y(1:100,1),y(1:100,2),'ko')
hold on 
plot(y(101:105,1),y(101:105,2),'kx')
title('t-SNE')
legend('ER+','ER-')


%%%PCA
qq_var_cent = zeros(size(qq_var,1),size(qq_var,2));
for i = 1:size(qq_var,1)
qq_var_cent(i,:) = qq_var(i,:)-mean(qq_var(i,:));    
end
[U,~,~] = svd(qq_var_cent);
z = U(:,1:r2)'*qq_var_cent;

figure (6)
plot(z(1,1:99),z(2,1:99),'ko')
hold on 
plot(z(1,Pos),z(2,Pos),'kx')
title('PCA')
xlabel('PC1')
ylabel('PC2')
legend('ER+','ER-')

