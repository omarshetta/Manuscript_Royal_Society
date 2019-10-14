function  [Lap,W]=build_knn_graph(X,k)
%%%%%%%%%%%%
%%% This function computes the K-Nearest Neighbour graph. 
%%% Inputs:
%%% X, is the data matrix with dimension (n x m) n is the number of samples and m is the number of features. 
%%% k, is a scalar that defines the number of nearest neighbours.
%%%
%%% Outputs: 
%%% Lap, is the graph Laplacian matrix of the k-nearest neighbour graph. It is an (n x n) symmetric positive semi-definite matrix.
%%% W, is the Weight matrix. It is an (n x n) symmetric matrix. It holds the weight of each edge of the k-nearest neighbour graph.
%%%%%%%%%%%%

n = size(X,1);

K = k+1;
dist = zeros(n,n);
W = zeros(n,n);
for i=1:n
    for j = 1:n 
    dist(i,j) = norm(X(i,:)-X(j,:),2);  
    end
    
    [~ , di] = sort(dist(i,:),'ascend');
    index_near = di(1:K);
    ind_far = setdiff(1:n,index_near);
    dist(i,ind_far)=0;
end

[row,col] = find(dist~=0);
const = ( sum(dist(:))/(K*n) ).^2; % const is the square of the mean of all distances in the graph.
for i = 1:length(row)
  
     W(row(i),col(i)) = exp(  - dist( row(i),col(i) ).^2 / const );
     
end

% test if W is symmetric (sanity check)
 if (norm(W - W.', 'fro') == 0)
        disp('The matrix W is symmetric');
 else
     W=(W+W.')/2; % symmetrize
 end
 
 
% Compute Laplacian matrix
d = sum(W,2);
D = diag(d);
Lap = D-W;

 
end