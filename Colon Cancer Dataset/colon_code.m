%% Preparing data and adding relvant paths.
clear all
addpath('./fast_kmeans/');
addpath('./gspbox/');
gsp_start;

load('colon.mat')
load('tissues.mat')
clearvars -except colon tissues 
ind_cancer=find(tissues<0);
ind_normal=find(tissues>0);
T=colon(:,ind_cancer);
T_outliers=[2,30,33,36,37];

[v,idx] = sort(var(T,0,2),'descend');

features=idx(1:700);
                                                                                            
M=T(features,:);
M_new=quantilenorm(M,'display',0);
%% Tuning lambda for OP. Perfroming a grid search 
n=50;
lambda_vec=linspace(0.1,0.8,n); % 0,2 to 0.47

for i =1:n  
 [L_hat,C_hat,cnt]=OUTLIER_PERSUIT(M_new,lambda_vec(i));
 r_OP(i)=rank(L_hat);
end

figure (1)
plot(lambda_vec,r_OP,'ko')
xlabel('\lambda')
ylabel('rank(L)')
grid on 

 %% refined lambda

 n=25;
lambda_vec2=linspace(0.2,0.5,n); % 0,2 to 0.47
predicted_labels_ref=zeros(40,n);
index_outliers_ref=zeros(40,n);
num_outliers_ref=zeros(1,n);
for i =1:n
 [L_hat,C_hat,cnt]=OUTLIER_PERSUIT(M_new,lambda_vec2(i));
r_OP_ref(i)=rank(L_hat);
[U_hat,S_hat,V_hat]=svd(L_hat);
 Z=U_hat(:,1:r_OP_ref(i))'*L_hat;
[ ~, predicted_labels_ref(:,i), mindist1,~,~ ] = kmeans_fast(Z',2,2,0);% input matrix needs to have samples in its rows
num_outliers_ref(i)=length(find(predicted_labels_ref(:,i)==2));
 
end

figure (2)
plot(lambda_vec2,num_outliers_ref,'ko')
 for i=1:length(lambda_vec2)
    
  x1=lambda_vec2(i);
  y1=num_outliers_ref(i);
  text(x1,y1,num2str(r_OP_ref(i)),'VerticalAlignment','bottom','FontSize',14)
  
 end
xlabel('\lambda')
ylabel('number of outliers')
grid on 
%% Using column sparse matrix to find outliers
[L_hat,C_hat,cnt]=OUTLIER_PERSUIT(M_new,0.46);
c_hat_norms=sqrt(sum(C_hat.^2)).^2;
figure (3)
title('Outlier Pursuit ||C_i|| L-2 norms')
grid on 
hold on 
plot(c_hat_norms,'ko');
xlabel('Sample Index i');
ylabel('||C_{i}||_{2}');
hold on 
anom=[2,30,33,36,37];
dx=0.3;
for i=1:5
    hold on 
text(anom(i)+dx,c_hat_norms(anom(i)),[num2str(anom(i))],'FontSize',12)
end
 %%  column sparse RPCA on graphs

param_graph.use_flann = 0; % to use the fast C++ library for construction of huge graphs.
param_graph.k = 5;
param_graph.type='knn';   
G1 = gsp_nn_graph(M_new',param_graph);

w=full(G1.W);
d=sum(w,2);
Lap_graph=diag(d)-w;

%% tune lambda for GOP. Using parameter search 
n=25;
lambda_vec=linspace(0.1,3,n);
predicted_labels_gr=zeros(40,n);
index_outliers_gr=zeros(40,n);
num_outliers_gr=zeros(1,n);
for i=1:n
[L_hat2,C_hat2,obj]=admm_algo_OP_on_graphs(M_new,lambda_vec(i),1,Lap_graph);%10
r_opg(i)=rank(L_hat2);
[U_hat,~,~]=svd(L_hat2);
Z=U_hat(:,1:r_opg(i))'*L_hat2;
[ ~, predicted_labels_gr(:,i), ~,~,~ ] = kmeans_fast(Z' ,2,2,0);

ind_outliers_gr=find(predicted_labels_gr(:,i)==2);
l=length(ind_outliers_gr);
index_outliers_gr(1:l,i)=ind_outliers_gr;
num_outliers_gr(i)=length(ind_outliers_gr);
end

figure (4)
plot(lambda_vec, num_outliers_gr,'ko')
for i=1:length(lambda_vec)
    
    x1=lambda_vec(i);
    y1=num_outliers_gr(i);
  text(x1,y1,num2str(r_opg(i)),'VerticalAlignment','bottom','FontSize',14)
  
 end
grid on
ylim([0 6])
xlabel('\lambda')
ylabel('number of outliers')


%% Robustness of GOP to gamma
n=30;
gamma_vec=linspace(0.5,10,n);
predicted_labels2=zeros(40,n);
index_outliers2=zeros(40,n);
num_outliers2=zeros(1,n);
for i=1:n
[L_hat2,C_hat2,obj]=admm_algo_OP_on_graphs(M_new,1.168,gamma_vec(i),Lap_graph);%10
[ ~, predicted_labels2(:,i), ~,~,~ ] = kmeans_fast(L_hat2' ,2,2,0);
r(i)=rank(L_hat2)
ind_outliers2=find(predicted_labels2(:,i)==2);
l=length(ind_outliers2);
index_outliers2(1:l,i)=ind_outliers2;
num_outliers2(i)=length(ind_outliers2);
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

[L_hat2,C_hat2,obj]=admm_algo_OP_on_graphs(M_new,1.168,1,Lap_graph);%10
rank(L_hat2)
[ ~, predicted_labels2, ~,~,~ ] = kmeans_fast(L_hat2' ,2,2,0);

 %% Using column sparse matrix to find outliers
[L_hat2,C_hat2,obj]=admm_algo_OP_on_graphs(M_new,1.168,1,Lap_graph); 
c_hat_norms2=sqrt(sum(C_hat2.^2)).^2;
m=median(c_hat_norms2);

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

%% Finding low dimensional projections for both GOP and OP 

[U,~,~]=svd(L_hat2);
Z2=U(:,1:2)'*L_hat2;

figure (8)
plot(Z2(1,:),Z2(2,:),'bo')
xlabel('PC1')
ylabel('PC2')
[ centers, predicted_labelsZ, mindist,~,~ ] = kmeans_fast(Z2(1:2,:)' ,2,2,0);
hold on 
plot(Z2(1,T_outliers),Z2(2,T_outliers),'ro')
title('Graph Regularized Outlier Pursuits')
hold on 
plot(centers(1,1),centers(1,2),'kx')
hold on 
plot(centers(2,1),centers(2,2),'kx')

%%% mahal dist OP
C=cov(Z2');
a=(centers(1,:)-centers(2,:));
mahal_OPG=sqrt(a*inv(C)*a')
%%%



%%%%  OP prjection
[U,~,~]=svd(L_hat);
Z=U(:,1:2)'*L_hat;
[ centers_OP, pred_OP, ~,~,~ ] = kmeans_fast(Z' ,2,2,0);
figure (9)
plot(Z(1,:),Z(2,:),'bo')
hold on 
plot(Z(1,T_outliers),Z(2,T_outliers),'ro')
hold on 
plot(centers_OP(1,1),centers_OP(1,2),'kx')
hold on 
plot(centers_OP(2,1),centers_OP(2,2),'kx')
xlabel('PC1')
ylabel('PC2')
title(' Outlier Pursuit')

%%% mahal distance OP
C=cov(Z');
a=(centers_OP(1,:)-centers_OP(2,:));
mahal_OP=sqrt(a*inv(C)*a')
%%%

%% F_score using clustering on low rank matrix projection to find outliers

%%% GOP
[U,~,~]=svd(L_hat2);
r=rank(L_hat2);
Z_OPG=U(:,1:r)'*L_hat2;
[ centers_OPG, pred_OPG, ~,~,~ ] = kmeans_fast(Z_OPG' ,2,2,0);

out_OPG1=find(pred_OPG==2);
[TP_OPG,FP_OPG]=outlier_det_perf(out_OPG1,T_outliers);
perf_OPG=[TP_OPG,FP_OPG];
%%% F_score 
Rec=TP_OPG/length(T_outliers);
Prec=TP_OPG/(TP_OPG+FP_OPG);
F1_OPG=2*(Rec * Prec)/( Rec + Prec);


%%%  OP
[U,~,~]=svd(L_hat);
r=rank(L_hat);
Z_OP=U(:,1:r)'*L_hat;
[ centers_OP, pred_OP, ~,~,~ ] = kmeans_fast(Z_OP' ,2,2,0);

out_OP=find(pred_OP==2);
[TP_OP,FP_OP]=outlier_det_perf(out_OP,T_outliers);
perf_OP=[TP_OP,FP_OP];
%%% F_score
Rec=TP_OP/length(T_outliers);
Prec=TP_OP/(TP_OP+FP_OP);
F1_OP=2*(Rec * Prec)/( Rec + Prec);





