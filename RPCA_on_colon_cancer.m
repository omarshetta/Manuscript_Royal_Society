clear
clear all
addpath('./fast_kmeans/');
addpath('./utils/');
addpath('./gspbox/');
addpath('./unlocbox/');
gsp_start;
init_unlocbox;
 

load('colon.mat')
load('tissues.mat')
clearvars -except colon tissues 
ind_cancer=find(tissues<0);
ind_normal=find(tissues>0);
T=colon(:,ind_cancer);
N=colon(:,ind_normal);
T_outliers=[2,30,33,36,37];
N_outliers=[8,12,34,36];


[v,idx] = sort(var(T,0,2),'descend');
%  plot(v,'bo')

 features=idx(1:800);
% features=feature_selection(T);
                                                                                            
  M=T(features,:);
%  M=T;
   M_new=quantilenorm(M,'display',0);
   
    [ ~, pred, ~,~,~ ] = kmeans_fast(M_new' ,2,2,0);

% M_new=M;
% for i=1:size(M_new,1)
%  M_new(i,:)=M_new(i,:)-mean(M_new(i,:)); % each gene to have zero mean
%  M_new(i,:)=M_new(i,:)/norm(M_new(i,:)); % each gene to have zero std
% end
% for i=1:size(M,2)
%  M_new(:,i)=M_new(:,i)-mean(M_new(:,i)); % normalize each sample to have zero mean
%  M_new(:,i)=M_new(:,i)/norm(M_new(:,i)); % each sample to have norm 1
% end


true_labels=ones(40,1);
true_labels(N_outliers)=2;
%%
n=50;
N=size(M_new,2);
lambda_vec=linspace(0.1,0.8,n); % 0,2 to 0.47
predicted_labels1=zeros(N,n);
ind_main=setdiff(1:N,N_outliers);
index_outliers=zeros(N,n);
omega=ones(size(M_new,1),size(M_new,2));
for i =1:n
 [L_hat,C_hat,cnt]=OUTLIER_PERSUIT(M_new,lambda_vec(i));
% [L_hat,C_hat]=mr_pca_part(M_new,omega,lambda_vec(i));
c_signal(i)=sum(sqrt(sum(C_hat.^2))); % ||C||1,2 
c_power(i)=norm(C_hat,'fro');
r_OP(i)=rank(L_hat);
[U_hat,S_hat,V_hat]=svd(L_hat);
% Z=U_hat(:,1:3)'*L_hat;
[ centers1, predicted_labels1(:,i), mindist1,~,~ ] = kmeans_fast(L_hat',2,2,0);% input matrix needs to have samples in its rows
[idx1,C1]=kmeans(L_hat',2);% input matrix needs to have samples in its rows

ind_outliers=find(predicted_labels1(:,i)==2);
l=length(ind_outliers);
index_outliers(1:l,i)=ind_outliers;

num_outliers(i)=length(find(predicted_labels1(:,i)==2));
% % TP(i)=length(find(true_labels(T_outliers)==predicted_labels1(T_outliers,i)));% outlier class is positve class and main class is negative class
% % ind_FP=true_labels(ind_main)~=predicted_labels1(ind_main,i);
% % ind_TN=true_labels(ind_main)==predicted_labels1(ind_main,i);
% % false_pos(i)=length(find(ind_FP));
% % TN(i)=length(find(ind_TN));
end


figure (1)
plot(svd(M_new),'-o')
ylabel('singular values')
title('scree plot of preprocessed data matrix')
% %  figure (2)
% %  plot(r_OP,num_outliers,'-o')
% %  xlabel('rank of L')
% %  ylabel('number of outliers kmeans')


 figure (3)
plot(lambda_vec,num_outliers,'-o');
 xlabel('\lambda')
 ylabel('num of outliers')
  hold on 
plot(ones(25,1)*0.4,1:25,'r')
hold on 
plot(ones(25,1)*0.5143,1:25,'r')

figure (5)
plot(lambda_vec(22:30) , num_outliers(22:30),'-o');
 xlabel('\lambda')
 ylabel('num of outliers')
% for i=1:length(lambda_vec(22:30))
%     
%     x1=lambda_vec(21+i);
%     y1=num_outliers(21+i);
%   text(x1,y1,num2str(r_OP(21+i)))  
%   
% end
figure (4)
 subplot (1,2,1)
 plot(lambda_vec,r_OP,'-o')
 xlabel('\lambda')
 ylabel('rank(L)')
 grid on 
 subplot(1,2,2)
 plot(svd(M_new),'-o')
ylabel('singular values')
title('scree plot of M')
 
 %% refined lambda

 n=30;
lambda_vec2=linspace(0.4,0.52,n); % 0,2 to 0.47
predicted_labels_ref=zeros(40,n);
index_outliers_ref=zeros(40,n);
num_outliers_ref=zeros(1,n);
for i =1:n
 [L_hat,C_hat,cnt]=OUTLIER_PERSUIT(M_new,lambda_vec2(i));
% [L_hat,C_hat]=mr_pca_part(M_new,omega,lambda_vec(i));
r_OP_ref(i)=rank(L_hat);
[U_hat,S_hat,V_hat]=svd(L_hat);
% Z=U_hat(:,1:3)'*L_hat;
[ centers1, predicted_labels_ref(:,i), mindist1,~,~ ] = kmeans_fast(L_hat',2,2,0);% input matrix needs to have samples in its rows
[idx1,C1]=kmeans(L_hat',2);% input matrix needs to have samples in its rows

ind_outliers_ref=find(predicted_labels_ref(:,i)==2);
l=length(ind_outliers_ref);
index_outliers_ref(1:l,i)=ind_outliers_ref;

num_outliers_ref(i)=length(find(predicted_labels_ref(:,i)==2));
 
end
figure (7)
plot(lambda_vec2,num_outliers_ref,'-o')
 for i=1:length(lambda_vec2)
    
    x1=lambda_vec2(i);
    y1=num_outliers_ref(i);
  text(x1,y1,num2str(r_OP_ref(i)))
  
 end
xlabel('\lambda')
ylabel('number of outliers')
title('refined \lambda')
 
 
 %%
% lambda_opt=lambda_vec(false_pos==min(false_pos));
acc=(TP+TN)./4;
diag=[TP;false_pos;TN;acc]
[L_hat,C_hat,cnt]=OUTLIER_PERSUIT(M_new,0.46);
% Z=U_hat(:,1:3)'*L_hat;
[ centers1_opt, predicted_labels1_opt, mindist1_opt,~,~ ] = kmeans_fast(L_hat',2,2,0);
% [idx,C]=kmeans(Z',2);
% omega=ones(size(M_new,1),size(M_new,2));
% [L_hat,C_hat]=mr_pca_part(M_new,omega,lambda);
figure (3)

plot(predicted_labels1_opt,'ko')
hold on 
o=ones(40,1);
o(T_outliers)=2;
 plot([T_outliers],o(T_outliers),'bx')
 title('predicted labels from kmeans clustering on RPCA subpace')
 legend('predicted labels from kmeans ','true outliers')
 axis([0 40 0 3  ])

%% find muscle index for all samples using the 4 smooth muscle EST's that are present in the 2000 gene colon cancer dataset of Uri. Alon 
load('names')
ESTs=names(features,2);

musc_ESTs={'T60155','J02854','X12369','H20709'};

for i=1:4  
ind_musc_ESTs(i)=find(strcmp(ESTs,musc_ESTs(i)));
end

musc_index=zeros(size(M_new,2),1);
for i=1:size(M_new,2)
musc_index(i)=mean(M_new(ind_musc_ESTs,i));    
end

musc_index_n=musc_index/max(musc_index);

figure (30)
plot(musc_index_n,'-o')


%%% muscle index of normal samples
N=colon(:,ind_normal);
M_N=quantilenorm(N,'display',0);
ESTs_N=names(:,2);
for i=1:4  
ind_musc_ESTs_N(i)=find(strcmp(ESTs_N,musc_ESTs(i)));
end

musc_index_N=zeros(size(N,2),1);
for i=1:size(N,2)
musc_index_N(i)=mean(M_N(ind_musc_ESTs_N,i));    
end
                                                                                     
musc_index_N=musc_index_N/max(musc_index_N);

%%
c_hat_norms=sqrt(sum(C_hat.^2)).^2;
m=median(c_hat_norms);

mad=1.483*median(abs(c_hat_norms-m));
threshold=m+1.7*mad;

figure (1)
title('column sparse RPCA ||C_i|| L-2 norms')
grid on 
hold on 
plot(c_hat_norms);
hold on 
plot(c_hat_norms,'ro');
xlabel('column index');
ylabel('column l2 norm');
hold on 
%  thresh=0.92*max(c_hat_norms);
plot(ones(40,1)*threshold,'k')
hold on 
% plot(ones(40,1)*thresh,'g')
outliers=find(c_hat_norms>threshold);
figure (20)
normplot(c_hat_norms)

 %%  column sparse RPCA on graphs

param_graph.use_flann = 0; % to use the fast C++ library for construction of huge graphs.
param_graph.k = 10;
param_graph.type='knn';
%  param_graph.k=param_graph.k+1;
% [indx, indy, dist, Xout, ~, epsilon]=gsp_nn_distanz(M_new,M_new,param_graph);
% sigma=mean(dist)^2;
% W=sparse(indx, indy,double(exp(-dist.^2/sigma)), 40, 40);
%  W(1:(40+1):end) = 0;
%    if (norm(W - W', 'fro') == 0)
%         disp('The matrix W is symmetric');
%    end
   
 G1 = gsp_nn_graph(M_new',param_graph);
 
A=full(G1.A);
w=full(G1.W);
d=sum(w,2);
Lap_graph=diag(d)-w;


% Lap=full(G1.L);


dist=sum(w.^2);
[~,h] = sort(dist ,'ascend') ;
figure (5)
caxis([0 1])
surf(w)
colorbar

% % figure (6)
% % caxis([0 1])
% % surf(Lap)
% % colorbar



T = repmat(sum(M_new.^2).^0.5 , size(M_new,1) , 1) ;
X = M_new./T ; 
G = X'*X ; G = G - diag(diag(G)) ;

d_coh=sum(G,2);
Lap_coh=diag(d_coh)-G;

p=sum(G.^2);
% p = p/max(p)
[~,b] = sort(p ,'ascend') ;
figure (8)
stem(p); title('The elements of vector p') ; grid on ; 
figure (7)
caxis([0 1])
surf(G)
colorbar


n=30;
gamma_vec=linspace(0.1,10,n);
predicted_labels2=zeros(40,n);
index_outliers2=zeros(40,n);
num_outliers2=zeros(1,n);
for i=1:n
[L_hat2,C_hat2,obj]=admm_algo_OP_on_graphs(M_new,0.46,gamma_vec(i),Lap_graph);%10
[ ~, predicted_labels2(:,i), ~,~,~ ] = kmeans_fast(L_hat2' ,2,2,0);

ind_outliers2=find(predicted_labels2(:,i)==2);
l=length(ind_outliers2);
index_outliers2(1:l,i)=ind_outliers2;
num_outliers2(i)=length(ind_outliers2);
end

figure (101)
plot(gamma_vec,num_outliers2,'-o')
title(' Laplacin from kNN graph ')
xlabel('\gamma')
ylabel('number of outliers')

[L_hat2,C_hat2,obj]=admm_algo_OP_on_graphs(M_new,0.46,1.466,Lap_graph);%10
[ ~, predicted_labels2, ~,~,~ ] = kmeans_fast(L_hat2' ,2,2,0);
figure (2)
c_hat_norms2=sqrt(sum(C_hat2.^2)).^2;
plot(c_hat_norms2,'ro')
hold on 
plot(c_hat_norms2,'b')



figure (4)

plot(predicted_labels2,'ko')
hold on 
o=ones(40,1);
o(T_outliers)=2;
plot([T_outliers],o(T_outliers),'bx')
 axis([0 40 0 3  ])
 title('predicted labels from kmeans clustering on RPCA on graphs subpace')
 legend('predicted labels from kmeans ','true outliers')
 
 
 [ ~, pred, ~,~,~ ] = kmeans_fast(M_new' ,2,2,0);
 
 
 

%% projections 

[U,~,~]=svd(L_hat2);
Z2=U(:,1:2)'*L_hat2;

figure (10)
plot(Z2(1,:),Z2(2,:),'bo')
xlabel('PC1')
ylabel('PC2')
[ centers, predicted_labelsZ, mindist,~,~ ] = kmeans_fast(Z2(1:2,:)' ,2,2,0);
hold on 
plot(Z2(1,T_outliers),Z2(2,T_outliers),'ro')
title('column sparse RPCA on graphs')
hold on 
plot(centers(1,1),centers(1,2),'kx')
hold on 
plot(centers(2,1),centers(2,2),'kx')

%%%%distance from second center
dist_2=sqrt(sum((Z2-centers(2,:)').^2));
[b,I]=sort(dist_2,'ascend');


z2=Z2(1,:);
% z2n=z2/std(z2);
FR2=(mean(z2(setdiff(1:40,[2,33,36,37])))-mean(z2([2,33,36,37])))^2/(var(z2(setdiff(1:40,[2,33,36,37]))) + var(z2([2,33,36,37])));

figure (5)
L_hat_unc=L_hat;
L_hat_unc(:,[2,33,36,37])=[];
[U,~,~]=svd(L_hat_unc);
Z1=U(:,1:3)'*L_hat;
[ centers1, predicted_labelsZ22, ~,~,~ ] = kmeans_fast(Z1' ,2,2,0);
plot(Z1(1,:),Z1(2,:),'bo')
hold on 
plot(Z1(1,T_outliers),Z1(2,T_outliers),'ro')
hold on 
plot(centers1(1,1),centers1(1,2),'kx')
hold on 
plot(centers1(2,1),centers1(2,2),'kx')
xlabel('PC1')
ylabel('PC2')
title(' column sparse RPCA')
figure (11)
[U,~,~]=svd(L_hat);
Z1=U(:,1:2)'*M_new;
[ centers1, pr, ~,~,~ ] = kmeans_fast(Z1' ,2,2,0);
plot(Z1(1,:),Z1(2,:),'bo')
hold on 
plot(Z1(1,T_outliers),Z1(2,T_outliers),'ro')
hold on 
plot(centers1(1,1),centers1(1,2),'kx')
hold on 
plot(centers1(2,1),centers1(2,2),'kx')
xlabel('PC1')
ylabel('PC2')
title(' column sparse RPCA')





z1=Z1(1,:);
z1n=z1/std(z1);
FR1=abs((mean(z1(setdiff(1:40,[2,33,36,37])))-mean(z1([2,33,36,37]))))/(var(z1(setdiff(1:40,[2,33,36,37]))) + var(z1([2,33,36,37])));


%%%%distance from second center
dist_1=sqrt(sum((Z1-centers1(2,:)').^2));
[b1,I1]=sort(dist_1,'ascend');
figure (6)

 c_hat_norms2=sqrt(sum(C_hat2.^2));

grid on 
hold on 
plot(c_hat_norms2);
hold on 
plot(c_hat_norms2,'ro');
xlabel('column index');
ylabel('column l2 norm');
hold on 
threshold2=0.96*max(c_hat_norms2);
plot(ones(40,1)*threshold2)

 outliers=find(c_hat_norms2>threshold2);
 
 %% finding outlier using graph method
 
 M_test=M_new;
for i=1:size(M_test,1)
 M_test(i,:)=M_test(i,:)-mean(M_test(i,:)); % each gene to have zero mean
 M_test(i,:)=M_test(i,:)/norm(M_test(i,:)); % each gene to have zero std
end
% for i=1:size(M,2)
%  M_test(:,i)=M_test(:,i)-mean(M_test(:,i)); % normalize each sample to have zero mean
%  M_test(:,i)=M_test(:,i)/norm(M_test(:,i)); % each sample to have norm 1
% end 
 
 l=param_graph.k;
dist=0;
s=zeros(size(M_test,2),1);
var=0.9;
for i=1:size(M_test,2)
  d=[]; 
ind=find(A(i,:)==1);
 for j=1:length(ind)  
     if(dist==1)
d(j)=exp(norm(M_test(:,i)-M_test(:,ind(j)))/var);
     else
         d(j)=norm(M_test(:,i)-M_test(:,ind(j)));
     end
 end    
[Y,~]=sort(d);
s(i)=Y(l);
    
end
figure (20)
plot(s,'-o')
 
 figure (21)
 plot(svd(M_test),'-o')
 figure(22)
 plot(sqrt(sum(M_test.^2)),'-o')
 


