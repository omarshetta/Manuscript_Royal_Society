function [TP,FP]=outlier_det_perf(Pos_pred,Pos)
%%%%%%%%%%%%
%%% This function returns the True Positives and False positives of the predicted outliers.
%%% Inputs:
%%% Pos, is an array that contains the index of known outliers.
%%% Pos_pred, is an array that contains the index of points that are predicted as outliers.
%%%
%%% Outputs: 
%%% TP, True Positives.
%%% FP, False Positives.
%%%%%%%%%%%%
TP=0;
FP=0;
for i=1:length(Pos_pred)
    
if(~isempty(find(Pos==Pos_pred(i))))
TP=TP+1;

else
    FP=FP+1;
end

end
end