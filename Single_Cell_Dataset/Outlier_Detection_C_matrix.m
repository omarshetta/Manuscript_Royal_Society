%%% This code uses GSPBOX for graph construction. DO NOT FORGET to download GSPBOX! as described in the READ_ME file in parent directory.


% Initialize gspbox library and add path for functions used in this script
clc
clear all
cd ..
addpath('./gspbox/');
addpath('./utils')
gsp_start;
cd Single_Cell_Dataset
% Load single cell dataset
disp(['Loading Single Cell Data... '])
load('Test_1_mECS.mat')
X = in_X;

for i = 1:max(unique(true_labs))
p(i) = length(find(true_labs==i));
end

load('ind_outliers.mat') % load indexes of the 6 sampled G2M cells for all 30 datasets.

%% Mahalanobis distance for outlier detection (same as fitting a Gaussian density) 

num_genes = [2,20,30,50,60,70];
FP_mahal  = zeros(30,numel(num_genes));
mahal     = zeros(65,30); 
Pos       = 60:65; % index of 6 outlier samples in constructed dataset

for k = 1:numel(num_genes)
    
    for j = 1:30
        
        %%% Constructing dataset with 6 G2M cells and 59 G1 cells
        ind_G2M   = find(true_labs==3);
        ind_G2M_6 = ind_G2M(ii_mat(j,:));
        ind_G1    = find(true_labs==1);
        idx       = [ind_G1;ind_G2M_6]; % first 59 cells are the G1 cells and the last 6 cells are chosen from G2M.
        X         = in_X(idx,:);
        %%%
        %%% Retain most varaible genes across samples
        v          = var(X);
        [b,ib]     = sort(v,'descend');
        diff_genes = ib(1:num_genes(k));
        X_ts       = X(:,diff_genes);
        %%%
 
        % Find mean vector and inverse covariance matrix to calculate Mahalanobis distance
        mu    = mean(X_ts);
        C     = cov(X_ts);
        C1    = C+(1e-10)*eye(numel(diff_genes),numel(diff_genes)); % need to regularize the covaraince matrix to overcome small numerical artefacts
        invC1 = inv(C1);
        
        for i = 1:size(X_ts,1)
            y = X_ts(i,:)'-mu';
            mahal(i,j) = sqrt(y'*invC1*y);
        end
        
        % Find number of false positives before detecting all known outliers
        [c,ic] = sort(mahal(:,j),'descend');
        h = find(c==min(mahal(Pos,j)));
        FP_mahal(j,k) = length(setdiff(ic(1:h),Pos));
        fprintf('dataset number = %d, number of genes = %d \n',j,num_genes(k))
    end
    
end


%% OP and GOP (59G1 and 6G2M) detecting percentage of False positives to detect all known outliers for different dimensions.

num_genes = [2,20,30,50,60,70];
FP_c_OPG  = zeros(30,numel(num_genes));
FP_OP     = zeros(30,numel(num_genes));
r_OPG     = zeros(30,numel(num_genes));
rop       = zeros(30,numel(num_genes));
Pos       = 60:65;

for k = 1:numel(num_genes)
    
    for j = 1:30
        %%% Constructing dataset with 6 G2M cells and 59 G1 cells
        ind_G1    = find(true_labs==3);
        ind_G2M_6 = ind_G1(ii_mat(j,:));
        ind_G1    = find(true_labs==1);
        idx       = [ind_G1;ind_G2M_6];
        X         = in_X(idx,:);
        %%% Retain most variable genes across samples
        v          = var(X);
        [b,ib]     = sort(v,'descend');
        diff_genes = ib(1:num_genes(k));
        X_ts       = X(:,diff_genes);
        %%%
        
        % Initialize k-Nearest Neighbour graph using gspbox functions
        param_graph.use_flann = 0; % to use the fast C++ library for construction of huge graphs.
        param_graph.k = 3; % choosing value of k
        param_graph.type = 'knn'; % specifying type of graph

      
        % Find W (square and symmetric similarity matrix) and find Laplacian matrix
        [G,sigma2] = gsp_nn_graph(X_ts,param_graph);
        w = full(G.W);
        d = sum(w,2);
        D = diag(d);
        Lap_graph = D-w;

        if(k==1)
            lambda_opg = 0.35;  
        else
            lambda_opg = 0.70;            
        end
            
        [L, C, obj] = admm_algo_OP_on_graphs(X_ts', lambda_opg, 1, Lap_graph); % GOP algorithm
        r_OPG(j,k) = rank(L);
        % Find number of false positives before detecting all known outliers
        c_OPG = sqrt(sum(C.^2));
        [c, ic] = sort(c_OPG,'descend');
        h = find(c==min(c_OPG(Pos)));
        FP_c_OPG(j,k) = length(setdiff(ic(1:h),Pos));


        if(k==1)
            lambda_op = 0.16; 
        else
            lambda_op = 0.3;
        end
        
        [L_hat, C_hat] = OUTLIER_PERSUIT(X_ts', lambda_op);% OP algorithm
        rop(j,k) = rank(L_hat);
        c_OP = sqrt(sum(C_hat.^2));
        
        % Find number of false positives before detecting all known outliers
        [c, ic] = sort(c_OP,'descend');
        h= find(c==min(c_OP(Pos)));
        FP_OP(j,k) = length(setdiff(ic(1:h),Pos));
    end
    
end
 %% MAD outlier detection method on 70 most variable genes


MAD         = zeros(65,30);
num_genes   = [2,20,30,50,60,70];
FP_MAD_test = zeros(30,1);
Pos         = 60:65;
k           = 6;
 
for j = 1:30
    
    %%% Constructing dataset with 6 G2M cells and 59 G1 cells
    ind_G1    = find(true_labs==3);
    ind_G2M_6 = ind_G1(ii_mat(j,:));
    ind_G1    = find(true_labs==1);
    idx       = [ind_G1;ind_G2M_6];
    X         = in_X(idx,:);
    %%%
    %%% Retain most variable genes across samples
    v          = var(X);
    [b,ib]     = sort(v,'descend');
    diff_genes = ib(1:num_genes(k));
    X_ts       = X(:,diff_genes); 
    %%% 
    
        for i = 1:size(X_ts,1)
            MAD(i,j) = mad(X_ts(i,:),1);
        end
        
    % Find number of false positives before detecting all known outliers
    [c,ic] = sort(MAD(:,j),'descend');
    h = find(c==min(MAD(Pos,j)));
    FP_MAD_test(j) = length(setdiff(ic(1:h),Pos));
    fprintf('dataset number = %d, number of genes = %d \n',j,num_genes(k))
    
end

 
 
%% Boxplot method on 70, 400 and 1000 most variable genes. 
%%% outlyingness score is number of outliers.


num_genes = [2,20,30,50,60,70,400,1000];
Num_out   = zeros(65,30);
FP_Box    = zeros(30,4);
Pos       = 60:65;

for k = 6:8
    for j = 1:30
    %%% Constructing dataset with 6 G2M cells and 59 G1 cells    
    ind_G1    = find(true_labs==3);
    ind_G2M_6 = ind_G1(ii_mat(j,:));
    ind_G1    = find(true_labs==1);
    idx       = [ind_G1;ind_G2M_6];
    X         = in_X(idx,:);
    %%% Retain most variable genes across samples
    v          = var(X);
    [b,ib]     = sort(v,'descend');
    diff_genes = ib(1:num_genes(k));
    X_ts       = X(:,diff_genes); 
    %%%
    
        for i = 1:size(X_ts,1)
            outliers = outliers_boxplot(X_ts(i,:)); 
            Num_out(i,j) = length(outliers);
        end
    % Find number of false positives before detecting all known outliers
    [c,ic] = sort(Num_out(:,j),'descend');
    h = find(c==min(Num_out(Pos,j)));
    FP_Box(j,k) = length(setdiff(ic(1:h(end)),Pos));
    fprintf('dataset number = %d, number of genes = %d \n',j,num_genes(k))
    end
end   

%% Plot

figure (1)
boxplot([100*FP_c_OPG(:,1)./59,100*FP_OP(:,1)./59,100*FP_mahal(:,1)./59,100*FP_c_OPG(:,2)./59,100*FP_OP(:,2)./59,100*FP_mahal(:,2)./59,100*FP_c_OPG(:,3)./59,100*FP_OP(:,3)./59,100*FP_mahal(:,3)./59,100*FP_c_OPG(:,4)./59,100*FP_OP(:,4)./59,100*FP_mahal(:,4)./59,100*FP_c_OPG(:,5)./59,100*FP_OP(:,5)./59,100*FP_mahal(:,5)./59,100*FP_c_OPG(:,6)./59,100*FP_OP(:,6)./59,100*FP_mahal(:,6)./59,100*FP_MAD_test./59,100*FP_Box(:,6)./59],'labels',{'GOP','OP','Mahal','GOP','OP','Mahal','GOP','OP','Mahal','GOP','OP','Mahal','GOP','OP','Mahal','GOP','OP','Mahal','MAD','BP'},'labelorientation','inline');
ylabel('False Positives %')
title('Single Cell Data Outlier Detection at Different Dimensions')

figure (2)
boxplot([100*FP_Box(:,6)./59,100*FP_Box(:,7)./59,100*FP_Box(:,8)./59],'labels',{'Box 70','Box 400','Box 1000'});
ylabel('False Positives %')
title('(Single Cell Dataset) Boxplot Method at Increasing Dimensions')


