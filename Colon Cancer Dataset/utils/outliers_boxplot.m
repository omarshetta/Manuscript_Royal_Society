function outliers = outliers_boxplot(x)
%%%%%%%%%%%%
%%% finds outliers using boxplot for input vector x.
%%% inputs: 
%%% x, is an input vector, we want to find its outlier elements.
%%%outputs:
%%% outliers, is a vector of outlier points of the input vector x.
%%%%%%%%%%%%
Q = quantile(x,[0.25, 0.5, 0.75]);
IQ = Q(3)-Q(1);
lower_fence = Q(1)-1.5*IQ ;
upper_fence = Q(3)+ 1.5*IQ ;
 
out_upper = find(x > upper_fence);
out_lower = find(x < lower_fence);
out=[out_lower ; out_upper];

outliers = x(out);

end