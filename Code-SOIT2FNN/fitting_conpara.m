function [optimizedParams,resnorm] = fitting_conpara(X,y_true,I,con_par,qlqr,rule_num,samp_num,M,K_link,Mylyr,ant_par,Mant)

format long;

initialparas = zeros(1, M*(2*rule_num*(I+1) +2 +2*I) +1); 
TMP_M = 2*rule_num*(I+1);
TMP_C = 2*(I+1);
c_num = I+1;

for i = 1 :M
    for ii = 1:rule_num
      initialparas((i-1)*TMP_M + (ii-1)*TMP_C + 1: (i-1)*TMP_M + (ii-1)*TMP_C + c_num) = con_par(:,i,1,ii);%C
      initialparas((i-1)*TMP_M + (ii-1)*TMP_C + c_num + 1 : (i-1)*TMP_M + (ii-1)*TMP_C + 2*c_num) = con_par(:,i,2,ii);%S
    end
end

for i = 1 :M
    initialparas(M*TMP_M + (i-1)*2+1) = qlqr(i,1);%ql
    initialparas(M*TMP_M + (i-1)*2+2) = qlqr(i,2);
%     initialparas(M*TMP_M + (i-1)*2+3) = Mylyr(i);
end

for i = 1 :M
    initialparas(M*(TMP_M + 2 ) + (i-1)*2*I + 1 : M*(TMP_M + 2 ) + (i-1)*2*I + I) = Mant(i,:,1);% mean of Mid
    initialparas(M*(TMP_M + 2 ) + (i-1)*2*I + I + 1 : M*(TMP_M + 2 ) + (i-1)*I*2 + 2*I) = Mant(i,:,2);% STD of Mid
end
    initialparas(M*(2*rule_num*(I+1) +2 +2*I) +1) = K_link;

options = optimoptions('lsqcurvefit', 'MaxIterations', 3000, 'MaxFunctionEvaluations', 25000); 

[R_vector_up_mid,R_vector_lo_mid,sum_R] = Computing_Antecedent (X,I,M, rule_num,samp_num,ant_par,Mant);

lb = [-100 * ones(1, M*TMP_M), zeros(1, 2 * M),0.01*ones(1,2*I*M), zeros(1, 1)]; 
ub = [100 * ones(1, M*TMP_M), ones(1, 2 * M),100*ones(1,2*I*M), ones(1, 1)]; 

[optimizedParams,resnorm] = lsqcurvefit(@(params, X) nonlinearFunction(params, X,rule_num,samp_num,R_vector_up_mid,R_vector_lo_mid,sum_R,I,M,K_link,Mylyr), initialparas, X, y_true, lb, ub, options);
end

function yy = nonlinearFunction(params, X,rule_num,samp_num,R_vector_up,R_vector_lo,sum_R,I,M,K_link,Mylyr)

TMP_M = 2*rule_num*(I+1);
TMP_C = 2*(I+1);
c_num = I+1;

yl_list= zeros(rule_num, samp_num);
yr_list= zeros(rule_num, samp_num);

for i = 1:M
    for ii = 1:rule_num
      c = params((i-1)*TMP_M + (ii-1)*TMP_C + 1: (i-1)*TMP_M + (ii-1)*TMP_C + c_num);
      s = params((i-1)*TMP_M + (ii-1)*TMP_C + c_num + 1 : (i-1)*TMP_M + (ii-1)*TMP_C + 2*c_num);
      X_with_bias = [ones(1, samp_num); X'];
      yl = c * X_with_bias - s * abs(X_with_bias);
      yr = c * X_with_bias + s * abs(X_with_bias);
      yl_list(ii, :) = yl;% rule_num * samp_num
      yr_list(ii, :) = yr;% rule_num * samp_num
    end

    % xxx0=R_vector_lo(:,i,:);
    % xxx2 =sum_R(i,:);
    % xxx1= (1-params(M*TMP_M + (i-1)*2+1))*sum(reshape(R_vector_lo(:,i,:), [rule_num, samp_num]) .* yl_list, 1);
    % xxx3= ((1-params(M*TMP_M + (i-1)*2+1))*sum(reshape(R_vector_lo(:,i,:), [rule_num, samp_num]) .* yl_list, 1) + params(M*TMP_M + (i-1)*2+1)*sum(R_vector_up(:,i,:) .* yl_list,1))./sum_R(i,:);;
    % Type Reduction -----> yleft, yright
    yleft(:,i) = ((1-params(M*TMP_M + (i-1)*2+1))*sum(reshape(R_vector_lo(:,i,:), [rule_num, samp_num]) .* yl_list, 1) + params(M*TMP_M + (i-1)*2+1)*sum(reshape(R_vector_up(:,i,:), [rule_num, samp_num]) .* yl_list,1))./sum_R(i,:);%samp_num*M 注意：加'.',e.g. '. * ' �?'./'是指矩阵的对应元素相乘除，结果仍为原矩阵大小
    yright(:,i) = ((1-params(M*TMP_M + (i-1)*2+2))*sum(reshape(R_vector_lo(:,i,:), [rule_num, samp_num]) .* yr_list, 1) + params(M*TMP_M + (i-1)*2+2)*sum(reshape(R_vector_up(:,i,:), [rule_num, samp_num]) .* yr_list,1))./sum_R(i,:);%samp_num*M
    y(:,i) = Mylyr(i)*yleft(:,i) + (1-Mylyr(i))*yright(:,i); %samp_num*M   
end

  %% Defuzzification -----> y
% y = Mylyr.*yleft + (1-Mylyr).*yright; %samp_num*M

for iii = 1 :M
    if iii ==1
        yy(:,iii) = (1-params(M*(2*rule_num*(I+1) +2 +2*I) +1))*y(:,iii) + params(M*(2*rule_num*(I+1) +2 +2*I) +1)*X(:,end);
    else
        yy(:,iii) = (1-params(M*(2*rule_num*(I+1) +2 +2*I) +1))*y(:,iii) + params(M*(2*rule_num*(I+1) +2 +2*I) +1)*yy(:,iii-1);
    end
end


end
