function [C] = colmn_thresh(C,eta)

n1=size(C,2);

for i=1:n1
    
  if(norm(C(:,i),2)<=eta)  
    
    C(:,i)=0;
  else
      
      C(:,i)=C(:,i)-eta*C(:,i)/norm(C(:,i),2);
  end


end

end