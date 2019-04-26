
function [S]=soft_thresh(S,eta)

[n1,n2]=size(S);
n=min(n1,n2);
for i=1:n
    
   
   if(abs(S(i,i))<= eta)
   S(i,i)=0;
   else 
    S(i,i)=S(i,i)-eta*sign(S(i,i));
       
   end
   
  
   
end


end