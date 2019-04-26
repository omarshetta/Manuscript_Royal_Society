%function demo_PCAG
%This code belongs to 2 models:
%1) "Robust PCA on graphs" (RPCAG) published in IEEE ICCV 2015. 
% 2) "Fast Robust PCA on graphs" (FRPCAG) published in IEEE Journal of
% Selected Topics in Signal Processing in 2016.
% The code provides a clustering demo for 30 subjects of the ORL
%dataset which is available at
%"http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html". 
% This dataset is also provided with this folder. We
%corrupt each of the 300 images belonging to 30 subjects (10 images each)
%with 10% random missing pixels and then apply the above two models to perform clustering:
% 1. robust PCA on graphs (RPCAG):
% min_{Lr} |X-Lr|_1 + lambda*|L|_* + gamma*tr(Lr L_1 Lr^T) 
% 2. fast robust PCA on graphs (FRPCAG): 
% min_{Lr} |X-L|_1 + gamma_1 tr(Lr^T L_1 Lr) + gamma_2 tr(Lr L_2 Lr^T)
% 3. PCA using graph total variation
% min_{Lr} |X-Lr|_1 + gamma_1 |Lr|_GTV + gamma_2 tr(Lr L_2 Lr^T)
% where X is the dataset with dimension N times Nx*Ny, and Lr is the
% low-rank, ||_GTV is the graph total variation
% N is the number of samples, Nx is the x dimension of the images and Ny is the y dimension of
% the images. For the non-image datasets, Nx should be set to 1 and Ny should be set to the number of features. 
%
% H_1 is the normalized graph Laplacian between the rows and 
% H_2 is the normalized graph Laplacian between the columns of X
%
% This code uses GSPBOX for graph construction and UNLOCBOX for the
% optimization part! DO NOT FORGET TO DOWNLOAD AND INSTALL THEM!

%%
clear
addpath('./utils/');
addpath('./gspbox/');
addpath('./gspbox/plotting');
addpath('./utils/fast_kmeans/');
addpath('./orl_faces/');
addpath('./algorithms/');
addpath('./unlocbox/');
gsp_start;
init_unlocbox;
%% get the breast cancer data
% [DNA_exp,text_exp]=xlsread('HiSeqV2_PANCAN.xlsx');
% [num_clin,text_clin]=xlsread('BRCA_clinicalMatrix.xlsx');

ER_pos=find(strcmp(text_clin(:,8),'Positive')==1);
patients_ER_pos=text_clin(ER_pos,1);

ER_neg=find(strcmp(text_clin(:,8),'Negative')==1);
patients_ER_neg=text_clin(ER_neg,1);

ind_pos=zeros(601,1);
for i=1:601
ind_pos(i)=find(strcmp(text_exp(1,2:1219),patients_ER_pos(i))==1);
end
DNA_exp_pos=DNA_exp(:,ind_pos);

ind_neg=zeros(179,1);
for i=1:179
ind_neg(i)=find(strcmp(text_exp(1,2:1219),patients_ER_neg(i))==1);
end
DNA_exp_neg=DNA_exp(:,ind_neg);

%% normalize
% normalize the dataset to zero mean and unit standard deviation along the
% features. this transformation should be applied after corrupting the
% images.
param_data = zero_means(param_data);

%% parameters for graph construction. (see GSPBOX for details)
param_graph.use_flann = 0; % to use the fast C++ library for construction of huge graphs.
param_graph.k = 7;

%% create the graphs
G1 = gsp_nn_graph(param_data.X,param_graph);
G2 = gsp_nn_graph(param_data.X',param_graph);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Robust PCA on Graphs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda = sqrt(max([param_data.N param_data.Nx*param_data.Ny]));  % nuclear norm 
gamma = 1;  % graph regularization
[L_rpcag, info_rpcag] = gsp_rpcag(param_data.X, lambda, gamma, G1,[]);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Fast Robust PCA on Graphs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.maxit = 500;
param.tol = 1e-4;
param.verbose = 2;

gamma1 = 1; %graph regularization along rows
gamma2 = 1; %graph regularization along columns
[L_frpcag, info_frpcag] = gsp_frpcaog_2g(param_data.X, gamma1, gamma2, G1, G2,param);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PCA using graph total variation
gamma1 = 3; %graph regularization along rows
gamma2 = 1; %graph regularization along columns
L_pcagtv = gsp_gpcatv_2g(param_data.X, gamma1, gamma2, G1, G2);

%% clustering comparison

[err_rpcag, err_frpcag, err_pcagtv] = clustering_quality(L_rpcag, L_frpcag, L_pcagtv , param_data)




