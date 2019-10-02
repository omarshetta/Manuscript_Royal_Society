function [TP,FP]=outlier_det_perf(Pos_pred,Pos)
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