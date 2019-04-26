function [X]=prox_nuclear_norm(X,neta)

[U,S,V]=svd(X,'econ');

[n1]=min(size(S));

for i=1:n1
    
   
   if(abs(S(i,i))<= neta)
   S(i,i)=0;
   else 
   S(i,i)=S(i,i)-neta*sign(S(i,i));
       
   end
   
  
   
end

X=U*S*V';

end