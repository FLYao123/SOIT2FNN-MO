function [M_y] = IT2FNNP (X,ant_par,con_par,M_qlqr,I,Y,Mant, K_link, Mylyr)

%---------------------------------
% IT2FNNP1  : Single-optput Parallel prediction scheme
% IT2FNNP2 : Single-optput serial prediction scheme 
% IT2FNNP3  : Multi-optput  prediction scheme 
% ant_par: Antecedent parameters
% con_par, Mqlqr: Conquent parameters
% I: Input Dimension
% M: Output Dimension
%---------------------------------
rule_num = size(ant_par,4);
samp_num = size(X,1);
M = size(Y,2);

[R_vector_up,R_vector_lo,sum_R] = Computing_Antecedent (X,I,M,rule_num,samp_num,ant_par,Mant);
[M_y] = nonlinearFunction_(con_par, M_qlqr, X,rule_num,samp_num,R_vector_up,R_vector_lo,sum_R,I,M,K_link,Mylyr);

% M_y = zeros(M,samp_num);
% M_yl = zeros(M,samp_num);
% M_yr = zeros(M,samp_num);
% 
% for i = 1 : M      
%   [y_tmp,yl_tmp,yr_tmp] = nonlinearFunction_(con_par(:,i,:,:), Mqlqr(i,:), X,rule_num,samp_num,R_vector_up(:,i,:),R_vector_lo(:,i,:),sum_R(i,:),I);
%   M_y(i,:) = y_tmp; % M * samp_num
%   M_yl(i,:) = yl_tmp; % M * samp_num
%   M_yr(i,:) = yr_tmp; % M * samp_num
% 
% end

end

function [yy] = nonlinearFunction_(con_par, qlqr, X_batch,rule_num,batch_size,R_vector_up,R_vector_lo,sum_R,I,M,K_link,Mylyr)

yl_list= zeros(rule_num, batch_size);
yr_list= zeros(rule_num, batch_size);
R_vector_yl_list = [];
R_vector_yr_list = [];
for iii = 1:M
    for ii = 1:rule_num
      c = con_par(:,iii,1,ii)';
      s = con_par(:,iii,2,ii)';
      X_with_bias = [ones(1, batch_size); X_batch'];
      yl = c * X_with_bias - s * abs(X_with_bias);
      yr = c * X_with_bias + s * abs(X_with_bias);
      yl_list(ii, :) = yl;% rule_num * samp_num
      yr_list(ii, :) = yr;% rule_num * samp_num
    end

    R_vector_yl_list= cat(3, R_vector_yl_list, yl_list); % 第一维为rule,第二维度为M,第三维为samp_num
    R_vector_yr_list = cat(3, R_vector_yr_list, yr_list);   

    % Type Reduction -----> yleft, yright
    yleft(:,iii) = ((1-qlqr(iii,1))*sum(reshape(R_vector_lo(:,iii,:), [rule_num, batch_size]).* yl_list, 1) + qlqr(iii,1)*sum(reshape(R_vector_up(:,iii,:), [rule_num, batch_size]) .* yl_list,1))./sum_R(iii,:);%samp_num*M 注意：加'.',e.g. '. * ' �?'./'是指矩阵的对应元素相乘除，结果仍为原矩阵大小
    yright(:,iii) = ((1-qlqr(iii,2))*sum(reshape(R_vector_lo(:,iii,:), [rule_num, batch_size]) .* yr_list, 1) + qlqr(iii,2)*sum(reshape(R_vector_up(:,iii,:), [rule_num, batch_size]) .* yr_list,1))./sum_R(iii,:);%samp_num*M

end



%% Defuzzification -----> y
y = Mylyr.*yleft + (1-Mylyr).*yright; %samp_num*M

for iii = 1 :M
    if iii ==1
        yy(:,iii) = (1-K_link)*y(:,iii) + K_link*X_batch(:,end);
    else
        yy(:,iii) = (1-K_link)*y(:,iii) + K_link*yy(:,iii-1);
    end
end

end

