%%% This code uses GSPBOX for graph construction. PLEASE DO NOT FORGET to download GSPBOX! as described in the READ_ME file.

%% Preparing data and adding relevant paths.
clc
% Initialize gspbox library and add path for functions used in this script
clear all
addpath('./fast_kmeans/');
addpath('./gspbox/');
addpath('./utils')
gsp_start;

% Load colon cancer data. Use only tumor samples.
load('colon.mat')
load('tissues.mat')
clearvars -except colon tissues 
ind_cancer = find(tissues<0);
T = colon(:,ind_cancer);
T_outliers = [2,30,33,36,37];

% Retaining 700 most variable genes
[v,idx] = sort(var(T,0,2),'descend');
features = idx(1:700);                                                                                       
M = T(features,:);

M_new = quantilenorm(M,'display',0); % Quantile normalize
%% Tuning lambda for OP. Using grid search method

n=50;
lambda_vec = linspace(0.1,0.8,n);

for i = 1:n  
    [L_hat, C_hat, cnt]=OUTLIER_PERSUIT(M_new, lambda_vec(i));% OP algorithm
    r_OP(i) = rank(L_hat);
end

figure (1)
plot(lambda_vec,r_OP,'ko')
xlabel('\lambda')
ylabel('rank(L)')
grid on 

%% Refine lambda search space

n=25;
lambda_vec2 = linspace(0.2,0.5,n); 
predicted_labels_ref = zeros(40,n);
index_outliers_ref = zeros(40,n);
num_outliers_ref = zeros(1,n);

for i =1:n
    [L_hat, C_hat, cnt]=OUTLIER_PERSUIT(M_new, lambda_vec2(i));% OP algorithm
    r_OP_ref(i) = rank(L_hat); % store rank of L_hat
    
    % Project L_hat onto first r_OP_ref(i) principal directions.
    [U_hat, S_hat, V_hat] = svd(L_hat);
    Z = U_hat(:,1:r_OP_ref(i))'*L_hat;
    
    [~, predicted_labels_ref(:,i), mindist1, ~, ~] = kmeans_fast(Z', 2, 2, 0); % k-means clustering with two clusters
    num_outliers_ref(i) = length(find(predicted_labels_ref(:,i)==2)); % Find number of outliers
end

figure (2)
plot(lambda_vec2, num_outliers_ref, 'ko')
 for i = 1:length(lambda_vec2)    
     x1 = lambda_vec2(i);
     y1 = num_outliers_ref(i);
     text(x1,y1,num2str(r_OP_ref(i)),'VerticalAlignment','bottom','FontSize',14)
 end
xlabel('\lambda')
ylabel('number of outliers')
title(' the number on each circle is the rank of recovered L at that specific \lambda')
grid on 

%% Using column sparse matrix to find outliers

[L_hat, C_hat, cnt]=OUTLIER_PERSUIT(M_new, 0.46); % 0.46 is optimal lambda
c_hat_norms = sqrt(sum(C_hat.^2)).^2; % Find l_{2} norm of each column of C_hat
figure (3)
title('Outlier Pursuit ||C_i|| L-2 norms')
grid on 
hold on 
plot(c_hat_norms,'ko');
xlabel('Sample Index i');
ylabel('||C_{i}||_{2}');
hold on 
anom = [2,30,33,36,37];
dx = 0.3;
for i = 1:5
    hold on 
    text(anom(i)+dx,c_hat_norms(anom(i)),[num2str(anom(i))],'FontSize',12)
end

 %%  Graph regualrized Outlier Pursuit (GOP)

%  Set k-Nearest Neighbour graph parameters. Check GSPBOX documentation for more details 
param_graph.use_flann = 0; % to use the fast C++ library for construction of huge graphs.
param_graph.k = 5;
param_graph.type = 'knn';   
G1 = gsp_nn_graph(M_new',param_graph); % construct graph

% Calculate Laplacian matrix
w = full(G1.W);
d = sum(w,2);
Lap_graph = diag(d)-w;

%% Tune lambda for GOP. Using grid search method

n = 25;
lambda_vec = linspace(0.1,3,n);
predicted_labels_gr = zeros(40,n);
index_outliers_gr = zeros(40,n);
num_outliers_gr = zeros(1,n);

for i = 1:n
[L_hat2, C_hat2, obj] = admm_algo_OP_on_graphs(M_new, lambda_vec(i), 1, Lap_graph);% GOP algorithm
r_gop(i) = rank(L_hat2);

% Project L_hat2 onto first r_gop(i) principal directions.
[U_hat,~,~] = svd(L_hat2);
Z = U_hat(:,1:r_gop(i))'*L_hat2;

[~, predicted_labels_gr(:,i), ~, ~, ~] = kmeans_fast(Z' , 2, 2, 0); % k-means clustering with two clusters
num_outliers_gr(i) = length(find(predicted_labels_gr(:,i)==2)); % Find number of outliers
end

figure (4)
plot(lambda_vec,num_outliers_gr,'ko')
for i = 1:length(lambda_vec)
    
    x1 = lambda_vec(i);
    y1 = num_outliers_gr(i);
  text(x1,y1,num2str(r_gop(i)),'VerticalAlignment','bottom','FontSize',14)
  
 end
grid on
ylim([0 6])
xlabel('\lambda')
ylabel('number of outliers')
title(' the number on each circle is the rank of recovered L at that specific \lambda')

%% Robustness of GOP to parameter gamma with optimal lambda
n = 30;
gamma_vec = linspace(0.5,10,n);
predicted_labels2 = zeros(40,n);
index_outliers2 = zeros(40,n);
num_outliers2 = zeros(1,n);

for i = 1:n
[L_hat2, C_hat2, obj] = admm_algo_OP_on_graphs(M_new, 1.1875, gamma_vec(i), Lap_graph); % 1.1875 is optimal lambda
[~, predicted_labels2(:,i), ~, ~, ~] = kmeans_fast(L_hat2', 2, 2, 0);
r(i) = rank(L_hat2);
ind_outliers2 = find(predicted_labels2(:,i)==2);
l = length(ind_outliers2);
index_outliers2(1:l,i) = ind_outliers2;
num_outliers2(i) = length(ind_outliers2);
end

figure (5)
plot(gamma_vec,num_outliers2,'-o')
title('Robustness of GOP to \gamma')
xlabel('\gamma')
ylabel('number of outliers')
figure (6)
plot(gamma_vec,r,'-o')
xlabel('\gamma')
ylabel('rank of L')
title('Robustness of GOP to \gamma')


 %% Using column sparse matrix to find outliers
[L_hat2, C_hat2, obj] = admm_algo_OP_on_graphs(M_new, 1.1875, 1, Lap_graph); 
c_hat_norms2 = sqrt(sum(C_hat2.^2)).^2;

figure (7)
grid on 
hold on 
plot(c_hat_norms2,'ko');
xlabel('Sample Index i');
ylabel('||C_{i}||_{2}');
hold on 
anom=[2,30,33,36,37];
dx=0.3;
for i=1:5
    hold on 
text(anom(i)+dx,c_hat_norms2(anom(i)),[num2str(anom(i))],'FontSize',12)
end 

%% Finding two dimensional visualization for GOP, OP, PCA and t-SNE.

[U,~,~] = svd(L_hat2);
Z2 = U(:,1:2)'*L_hat2;


non_out = setdiff(1:40,T_outliers);
figure (8)
plot(Z2(1,non_out),Z2(2,non_out),'ko')
xlabel('PC1')
ylabel('PC2')
[ centers, predicted_labelsZ, mindist,~,~ ] = kmeans_fast(Z2(1:2,:)', 2, 2, 0);
hold on 
plot(Z2(1,T_outliers),Z2(2,T_outliers),'ro')
title('Graph Regularized Outlier Pursuit, MD=2.8906')
hold on 
plot(centers(1,1),centers(1,2),'k+')
hold on 
plot(centers(2,1),centers(2,2),'k+')
legend('non-outliers','outliers','center 1','center 2')

%%% Mahalanobis distance GOP
C = cov(Z2');
a = (centers(1,:)-centers(2,:));
mahal_GOP = sqrt(a*inv(C)*a');
%%%




%%%%  OP projection
[U,~,~] = svd(L_hat);
Z = U(:,1:2)'*L_hat;
[centers_OP, pred_OP, ~, ~, ~] = kmeans_fast(Z', 2, 2, 0);
figure (9)
plot(Z(1,non_out),Z(2,non_out),'ko')
hold on 
plot(Z(1,T_outliers),Z(2,T_outliers),'ro')
hold on 
plot(centers_OP(1,1),centers_OP(1,2),'k+')
hold on 
plot(centers_OP(2,1),centers_OP(2,2),'k+')
xlabel('PC1')
ylabel('PC2')
title(' Outlier Pursuit, MD=1.8973')
legend('non-outliers','outliers','center 1','center 2')


%%% Mahalanobis distance OP
C = cov(Z');
a = (centers_OP(1,:)-centers_OP(2,:));
mahal_OP = sqrt(a*inv(C)*a');
%%%



figure (10)
y = tsne(M_new');
[ centers_tsne, pred_tsne, ~, ~,~] = kmeans_fast(y, 2, 2, 0);
plot(y(non_out,1),y(non_out,2),'ko')
hold on 
plot(y(T_outliers,1),y(T_outliers,2),'ro')
hold on 
plot(centers_tsne(1,1),centers_tsne(1,2),'k+')
hold on 
plot(centers_tsne(2,1),centers_tsne(2,2),'k+')
title('t-SNE, MD=1.6777')
legend('non-outliers','outliers','center 1','center 2')
%%% Mahalanobis distance t-SNE
C = cov(y);
a = (centers_tsne(1,:)-centers_tsne(2,:));
mahal_tsne = sqrt(a*inv(C)*a');
%%%


%%%PCA
M_new_cent = zeros(size(M_new,1),size(M_new,2));
for i = 1:size(M_new,1)
M_new_cent(i,:) = M_new(i,:)-mean(M_new(i,:));    
end
[U,~,~] = svd(M_new_cent);
z_PCA = U(:,1:2)'*M_new_cent;
[ centers_PCA, pred_PCA, ~, ~, ~] = kmeans_fast(z_PCA', 2, 2, 0);
figure (11)
plot(z_PCA(1,non_out),z_PCA(2,non_out),'ko')
hold on 
plot(z_PCA(1,T_outliers),z_PCA(2,T_outliers),'ro')
hold on 
plot(centers_PCA(1,1),centers_PCA(1,2),'k+')
hold on 
plot(centers_PCA(2,1),centers_PCA(2,2),'k+')
title('PCA, MD=1.7839')
xlabel('PC1')
ylabel('PC2')
legend('non-outliers','outliers','center 1','center 2')
%%% Mahalanobis distance OP
C = cov(z_PCA');
a = (centers_PCA(1,:)-centers_PCA(2,:));
mahal_PCA = sqrt(a*inv(C)*a');
%%%

clc
fprintf( ' Separability GOP = %d \n Separability OP = %d \n Separability PCA = %d \n Separability t-SNE = %d', mahal_GOP,mahal_OP, mahal_PCA, mahal_tsne)


%% F_score using clustering on low dimensional embedding to find outliers.

%%% GOP
[U,~,~] = svd(L_hat2);
r = rank(L_hat2);
Z_GOP = U(:,1:r)'*L_hat2;
[ centers_GOP, pred_GOP, ~, ~,~] = kmeans_fast(Z_GOP', 2, 2, 0);

out_GOP = find(pred_GOP==2);
[TP_GOP, FP_GOP]=outlier_det_perf(out_GOP, T_outliers);
perf_GOP=[TP_GOP,FP_GOP];
%%% F_score 
Rec = TP_GOP/length(T_outliers);
Prec = TP_GOP/(TP_GOP+FP_GOP);
F_GOP = 2*(Rec * Prec)/( Rec + Prec);


%%%  OP
[U,~,~] = svd(L_hat);
r = rank(L_hat);
Z_OP = U(:,1:r)'*L_hat;
[ centers_OP, pred_OP, ~, ~, ~] = kmeans_fast(Z_OP', 2, 2, 0);

out_OP = find(pred_OP==2);
[TP_OP, FP_OP]=outlier_det_perf(out_OP, T_outliers);
perf_OP = [TP_OP,FP_OP];
%%% F_score
Rec = TP_OP/length(T_outliers);
Prec = TP_OP/(TP_OP+FP_OP);
F_OP = 2*(Rec * Prec)/( Rec + Prec);





%%% PCA
[ centers_PCA, pred_PCA, ~, ~, ~] = kmeans_fast(z_PCA', 2, 2, 0);
out_PCA = find(pred_PCA==2);
[TP_PCA, FP_PCA]=outlier_det_perf(out_PCA, T_outliers);
perf_PCA=[TP_PCA,FP_PCA];

%%% F_score
Rec = TP_PCA/length(T_outliers);
Prec = TP_PCA/(TP_PCA+FP_PCA);
F_PCA = 2*(Rec * Prec)/( Rec + Prec);




%%%t-SNE
[ centers_tsne, pred_tsne, ~, ~, ~] = kmeans_fast(y, 2, 2, 0);

out_tsne = find(pred_tsne==2);
[TP_tsne, FP_tsne]=outlier_det_perf(out_tsne, T_outliers);
perf_tsne = [TP_tsne,FP_tsne];

%%% F_score
Rec = TP_tsne/length(T_outliers);
Prec  =TP_tsne/(TP_tsne+FP_tsne);
F_tsne = 2*(Rec * Prec)/( Rec + Prec);

fprintf(' \n F-Score GOP = %d \n F-Score OP = %d \n F-Score PCA = %d \n F-Score t-SNE = %d', F_GOP,F_OP, F_PCA, F_tsne)


