function [res_reducing, index_min, M_qlqr, con_par,Mylyr,Mant,K_link] = Rule_Reducing(X,Y, ant_par_before,con_par_before,Mqlqr,I,M,res_last,Mant, K_link,Mylyr)
% X �? sample_num * I 矩阵;
% Y �? sample_num * M 矩阵;
% ant_par_before �? 1*I *3 * rule_num;
% con_par_after �? (I +1)*M *2 *rule_num, 即前两维是数组，第三维是哪一个参数，�?后一维是哪一个（RULE�?
% Mqlqr �? M *2 矩阵，每�?行对应一个输出，第一列对应ql，第二列对于qr
% I 为输入维�?
% M 为输出维�?
rule_num = size(ant_par_before,4);
samp_num = size(X,1);
optimizedParams = [];
for r = 1:rule_num
    ant_par = ant_par_before;
    con_par = con_par_before;
    ant_par(:,:,:,r) =[];
    con_par(:,:,:,r) =[];

    %% Computing consequent parameters in stage one

    [optimizedParams_tmp, resnorm_tmp] = fitting_conpara(X, Y, I, con_par, Mqlqr, rule_num-1, samp_num, M, K_link,Mylyr,ant_par,Mant);
%     [optimizedParams,resnorm_tmp] = fitting_conpara(X,Y,I,con_par,Mqlqr,rule_num,samp_num,M,K_link,Mylyr,ant_par,Mant);

    res_tmp = resnorm_tmp/samp_num;
    res(r) = mean(res_tmp);   

    optimizedParams = vertcat(optimizedParams, optimizedParams_tmp);

end
delta_res = res - res_last;
[~, index_min] = min(delta_res);
res_reducing = res(index_min);

%*******************************
TMP_M = 2*(rule_num-1)*(I+1);
TMP_C = 2*(I+1);
c_num = I+1;

for i =1:M
    M_qlqr(i,1) = optimizedParams(index_min, M*TMP_M + (i-1)*2+1);%ql
    M_qlqr(i,2) = optimizedParams(index_min, M*TMP_M + (i-1)*2+2);
%     Mylyr(i) = optimizedParams(index_min, M*TMP_M + (i-1)*2+3);%Mylyr
  for ii = 1:rule_num-1
    con_par(:,i,1,ii) = optimizedParams(index_min, (i-1)*TMP_M + (ii-1)*TMP_C + 1: (i-1)*TMP_M + (ii-1)*TMP_C + c_num);%C
    con_par(:,i,2,ii) = optimizedParams(index_min, (i-1)*TMP_M + (ii-1)*TMP_C + c_num + 1 : (i-1)*TMP_M + (ii-1)*TMP_C + 2*c_num);%S    
  end

Mant(i,:,1) = optimizedParams(M*(TMP_M + 2 ) + (i-1)*2*I + 1 : M*(TMP_M + 2 ) + (i-1)*2*I + I);% mean of Mid
Mant(i,:,2) = optimizedParams(M*(TMP_M + 2 ) + (i-1)*2*I + I + 1 : M*(TMP_M + 2 ) + (i-1)*I*2 + 2*I);% STD of Mid
K_link = optimizedParams(M*(TMP_M + 2) + (i-1)*I*2 + 2*I + 1);% STD of Mid

end

end

