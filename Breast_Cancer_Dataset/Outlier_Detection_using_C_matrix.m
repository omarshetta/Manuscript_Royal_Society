%%% This code uses GSPBOX for graph construction. PLEASE DO NOT FORGET to download GSPBOX! as described in the READ_ME file in parent directory.

%%% Download breast cancer TCGA dataset from Xena browser following
%%% instructions in 'Download Breast cancer TCGA dataset.txt' file. Download file then convert to
%%% excel and rename as 'BRCA_TCGA.xlsx'
clc
clear all
disp(['Loading Breast Cancer Dataset...'])
[breast_TCGA,text] = xlsread('BRCA_TCGA.xlsx'); % load dataset
id_sequed = text(1,2:end);

% Initialize gspbox library and add path for functions used in this script
cd ..
addpath('./gspbox/');
addpath('./utils')
gsp_start;
cd Breast_Cancer_Dataset

% Load BRCA clinical data, find ER+ and ER- patients 
[~,t_clin] = xlsread('BRCA_clinical.xlsx'); 
patient_ids = t_clin(:,1);
ii = find(strcmp(t_clin(1,:),'ER_Status_nature2012'));
i_posi = find(strcmp(t_clin(:,8),'Positive')); %% index of ER positive patients
i_neg = find(strcmp(t_clin(:,8),'Negative'));
pos_patients = patient_ids(i_posi);
neg_patients = patient_ids(i_neg);

ind_pos = zeros(601,1);
for j = 1:601
    ind_pos(j) = find(strcmp(id_sequed,pos_patients(j)));  
end

ind_neg = zeros(179,1);
for j=1:179
    ind_neg(j) = find(strcmp(id_sequed,neg_patients(j)));
end
X = breast_TCGA;


load('ind_outliers') % indexes of 5 ER- samples in the total of 179 ER-.   
load('i_pos') % index of 100 ER+ samples in the total of 600 ER+.

% Initialize number of dimensions to be used in experiment
num_dim = [25,50,80,95,100,200];

%% Mahalanobis distance for outlier detection (same as fitting a Gaussian density) 
Pos           = 101:105;
mahal         = zeros(105,30);
FP_mahal_test = zeros(30,numel(num_dim));

for k = 1:numel(num_dim)
    
    for j = 1:30
        
        X_tst = [ X(:,ind_pos(i_pos)) , X(:,ind_neg(ind_mat1(j,:))) ]; % constructiong 100 ER+ and 5 ER- dataset.

        %%% filter dataset to most varaible genes
        v          = var(X_tst,0,2);
        [b,ib]     = sort(v,'descend');
        diff_genes = ib(1:num_dim(k));
        X_ts       = X_tst(diff_genes,:);
        
        % Find mean vector and inverse covariance matrix to calculate Mahalanobis distance
        mu    = mean(X_ts,2);
        C     = cov(X_ts');
        C1    = C+(1e-9)*eye(numel(diff_genes),numel(diff_genes)); % Need to regularize the covariance matrix to overcome small numerical artefacts
        invC1 = inv(C1);
        
        for i = 1:size(X_ts,2)
            y = X_ts(:,i)-mu;
            mahal(i,j) = sqrt(y'*invC1*y);
        end
        
        % Find number of False positives detected to find all 5 known outliers
        [c,ic] = sort(mahal(:,j),'descend');
        h = find(c==min(mahal(Pos,j)));
        FP_mahal_test(j,k) = length(setdiff(ic(1:h),Pos));
        
        fprintf('dataset number = %d, number of genes = %d \n', j,num_dim(k))
    end
    
end


%% OP and GOP choosing most variable genes
n           = numel(num_dim);
Pos         = 101:105;
r_opg       = zeros(30,n);
rop         = zeros(30,n);
FP_OPG_test = zeros(30,n);
FP_OP_test  = zeros(30,n);

 for k = 1:n

    for j = 1:30
        X_tst = [ X(:,ind_pos(i_pos)) , X(:,ind_neg(ind_mat1(j,:))) ]; % constructiong 100 ER+ and 5 ER- dataset.
        %%% retain highest varaince genes across samples
        v          = var(X_tst,0,2);
        [vv,iv]    = sort(v,'descend');
        diff_genes = iv(1:num_dim(k));
        X_ts       = X_tst(diff_genes,:);
        M          = quantilenorm(X_ts);
        
        % Set k-Nearest Neighbour graph parameters. Check GSPBOX documentation for more details
        param_graph.use_flann = 0; % to use the fast C++ library for construction of huge graphs.
        param_graph.k = 5; % specifying the value of k
        param_graph.type = 'knn'; % specifying type of graph
        %%%
        
        % Find W (square and symmetric similarity matrix) and find Laplacian matrix
        [G,sigma2] = gsp_nn_graph(M',param_graph); % create k-NN graph
        w = full(G.W);
        d = sum(w,2);
        D = diag(d);
        Lap_graph = D-w;

       
       [L_hat2, C_hat2] = admm_algo_OP_on_graphs(M, 0.70, 1, Lap_graph); % GOP algorithm 
       r_opg(j,k) = rank(L_hat2);
       
       % Find number of False positives detected to find all 5 known outliers
       c_norms_OPG = sqrt(sum(C_hat2.^2));
       [c,ic] = sort(c_norms_OPG,'descend');
       h = find(c==min(c_norms_OPG(Pos)));
       FP_OPG_test(j,k) = length(setdiff(ic(1:h),Pos));
       
       
       
       lambda = 0.38;
       [L_hat, C_hat] = OUTLIER_PERSUIT(M, lambda);% OP algorithm
       rop(j,k) = rank(L_hat);
       c_OP = sqrt(sum(C_hat.^2));

       % Find number of False positives detected to find all 5 known outliers
       [c, ic] = sort(c_OP,'descend');
       h = find(c==min(c_OP(Pos)));
       FP_OP_test(j,k) = length(setdiff(ic(1:h),Pos));

       fprintf('dataset number = %d, number of genes = %d \n', j,num_dim(k))
    end

 end


 %% MAD outlier detection method on 200 most variable genes
Pos = 101:105;
MAD = zeros(105,30);
FP_MAD_test = zeros(30,1);
k = 6;

for j = 1:30
    X_tst = [ X(:,ind_pos(i_pos)) , X(:,ind_neg(ind_mat1(j,:))) ]; % constructiong 100 ER+ and 5 ER- dataset.
    %%% retain highest varaince genes across samples
    v          = var(X_tst,0,2);
    [b,ib]     = sort(v,'descend');
    diff_genes = ib(1:num_dim(k)); 
    X_ts       = X_tst(diff_genes,:); 
 
    for i = 1:105
       MAD(i,j) = mad(X_ts(:,i),1);
    end

    % Find number of False positives detected to find all 5 known outliers
    [c,ic] = sort(MAD(:,j),'descend');
    h = find(c==min(MAD(Pos,j)));
    FP_MAD_test(j) = length(setdiff(ic(1:h),Pos));
    
    fprintf('dataset number = %d, number of genes = %d \n', j,num_dim(k))
end

  
 
%% Boxplot method on 200, 1000 and 2000 most variable genes. 
Pos = 101:105;
num_dim = [25,50,80,95,100,200,1000,2000];

%%% In this case outlyingness score is number of outliers. 

Num_out = zeros(105,30);
FP_Box = zeros(30,4);

for k = 6:8
    for j = 1:30
    X_tst = [ X(:,ind_pos(i_pos)) , X(:,ind_neg(ind_mat1(j,:))) ]; % constructiong 100 ER+ and 5 ER- dataset.
    %%% Retain highest varaince genes across samples
    v          = var(X_tst,0,2);
    [b,ib]     = sort(v,'descend');
    diff_genes = ib(1:num_dim(k)); 
    X_ts       = X_tst(diff_genes,:); 
 
        for i = 1:105
            outliers = outliers_boxplot(X_ts(:,i)); 
            Num_out(i,j) = length(outliers);
        end
    
    % Find number of False positives detected to find all 5 known outliers    
    [c,ic] = sort(Num_out(:,j),'descend');
    h = find(c==min(Num_out(Pos,j)));
    FP_Box(j,k) = length(setdiff(ic(1:h(end)),Pos));
    
    fprintf('dataset number = %d, number of genes = %d \n', j,num_dim(k))
    end
end   

%% Plot figures of False Positives (%) for the different dimensions for different methods

figure(1)
boxplot([FP_OPG_test(:,1),FP_OP_test(:,1),FP_mahal_test(:,1),FP_OPG_test(:,2),FP_OP_test(:,2),FP_mahal_test(:,2),FP_OPG_test(:,3),FP_OP_test(:,3),FP_mahal_test(:,3),FP_OPG_test(:,4),FP_OP_test(:,4),FP_mahal_test(:,4),FP_OPG_test(:,5),FP_OP_test(:,5),FP_mahal_test(:,5),FP_OPG_test(:,6),FP_OP_test(:,6),FP_mahal_test(:,6),FP_MAD_test,FP_Box(:,6)],'labels',{'GOP','OP','Mahal','GOP','OP','Mahal','GOP','OP','Mahal','GOP','OP','Mahal','GOP','OP','Mahal','GOP','OP','Mahal','MAD','BP'},'labelorientation','inline')
ylabel('False Positives (%)')
title('Breast Cancer Dataset at Different Dimensions')
   
figure (2)
boxplot([FP_Box(:,6),FP_Box(:,7),FP_Box(:,8)],'labels',{'Boxplot 200','Boxplot 1000','Boxplot 2000'})
ylabel('False Positives (%)')
title('(Breast Cancer Dataset) Boxplot Method at Increasing Dimensions')
   
