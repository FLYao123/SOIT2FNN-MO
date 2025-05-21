function [R_vector_up_mid,R_vector_lo_mid,sum_R] = Computing_Antecedent (X,I,M,rule_num,samp_num,ant_par,Mant)
% format long;
% Mant: 第一维是 M，第二维是I，第三维是哪�?个参�?
R_vector_up_mid = [];
R_vector_lo_mid = [];

%% Computing Antecedent 
for i = 1:samp_num  

  for ii = 1:rule_num
     % layer two ----->miu_up & miu_lo  
     [miu_up,miu_lo] = IT2MF(ant_par(:,:,1,ii), ant_par(:,:,2,ii),ant_par(:,:,3,ii), X(i,:),I); %miu_up 和miu_lo都是行向�?:1×I

%      % layer three --------> R_up & R_lo
%      R_up(ii) = max(prod(miu_up), 1e-16); % 确保不会因为累成变得无限小，被默认为0
%      R_lo(ii) = max(prod(miu_lo), 1e-17);
     % %******************取log,ji***************************
%      R_up(ii) = -1 ./log(prod(miu_up)); 
%      R_lo(ii) = -1 ./log(prod(miu_lo)); 


     for iii = 1 : M
         miu_mid = exp(-0.5*((X(i,:) - Mant(iii,:,1))./Mant(iii,:,2)).^2);
         R_up_mid(ii,iii) = -1 ./max(min((sum(log(miu_up))+sum(log(miu_mid))), -1e-3), -1e5);  % rule_num*M
         R_lo_mid(ii,iii) = -1 ./max(min((sum(log(miu_lo))+sum(log(miu_mid))),-1e-3), -1e5);
     end
  end
  R_vector_up_mid = cat(3, R_vector_up_mid, R_up_mid); % 第一维为rule,第二维度为M,第三维为samp_num
  R_vector_lo_mid = cat(3, R_vector_lo_mid, R_lo_mid);
end
%% Computing Consequent 
% sum_R = squeeze(sum(R_vector_up_mid,1))  + squeeze(sum(R_vector_lo_mid,1)); % 在第�?维上求和，结果为M*samp_num
sum_R = reshape(sum(R_vector_up_mid,1), [M, samp_num])  + reshape(sum(R_vector_lo_mid,1), [M, samp_num]); % 在第�?维上求和，结果为M*samp_num

end