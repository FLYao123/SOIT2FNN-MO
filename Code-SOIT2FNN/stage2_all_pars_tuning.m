function [ant_par,Mant,con_par,M_qlqr,Mylyr, K_link, loss] = stage2_all_pars_tuning(X,Y, ant_par,Mant,con_par,M_qlqr,I,M,lr_init, ep_max,batch_size, K_link,Mylyr)
% format long;
rule_num = size(ant_par,4);
samp_num = size(X,1);

%% Computing Antecedent 

decay_rate = 0.999;
loss = [];

[R_vector_up,R_vector_lo,sum_R] = Computing_Antecedent (X,I,M, rule_num,samp_num,ant_par,Mant);    
[M_y_sum,~,~,~,~,~,~,~,~] = nonlinearFunction_(con_par, M_qlqr, X,rule_num,samp_num,R_vector_up,R_vector_lo,sum_R,I,M,K_link,Mylyr);   
loss1 = mean(mean((M_y_sum - Y)'.^2, 2)); % Y_batch' 和 M_y 为 M*batch_size； sum((Y_batch' - M_y).^2, 1)为1*batch_size; loss 为一个标量

for ep = 1 : ep_max

    lr = lr_init* decay_rate^ep;

    %%Sample a batch
    indices = randperm(samp_num, batch_size);
    X_batch = X(indices,:);
    Y_batch = Y(indices,:); % actual value  batch_size * M

    [R_vector_up,R_vector_lo,sum_R] = Computing_Antecedent (X_batch,I,M, rule_num,batch_size,ant_par,Mant);
    %R_vector_up,R_vector_lo: 第一维为rule_num,第二维度为M,第三维为samp_num or batch_size
    %sum_R: M*samp_num

    [M_y,R_yl,R_yr,yl_R_up,yl_R_lo,yr_R_up,yr_R_lo,y_Mylyr,y] = nonlinearFunction_(con_par, M_qlqr, X_batch,rule_num,batch_size,R_vector_up,R_vector_lo,sum_R,I,M,K_link,Mylyr);

    % 反向传播
    [delta_c, delta_s, delta_ql, delta_qr, delta_m1, delta_m2, delta_sig, delta_m_mid,delta_sig_mid,delta_Mylyr,delta_E_l] = backpropagation_delta(X_batch, Y_batch, M_y, R_vector_up,R_vector_lo,sum_R, M_qlqr,I,M,rule_num,R_yl,R_yr,yl_R_up,yl_R_lo,yr_R_up,yr_R_lo,ant_par,Mant,K_link,Mylyr,y_Mylyr,y);

    % 更新参数
    [ant_par, con_par,M_qlqr,Mant,Mylyr, K_link] = update_parameters(ant_par, con_par, M_qlqr,Mant, K_link, delta_c, delta_s, delta_ql, delta_qr, delta_m1, delta_m2, delta_sig, delta_m_mid,delta_sig_mid,lr,rule_num,I,delta_Mylyr,Mylyr,delta_E_l);
    
    %*****************Computing Loss***********************
     if  mod(ep, 50) == 0 || ep<10
        [R_vector_up,R_vector_lo,sum_R] = Computing_Antecedent (X,I,M, rule_num,samp_num,ant_par,Mant);      
        [M_y_sum,~,~,~,~,~,~,~,~] = nonlinearFunction_(con_par, M_qlqr, X,rule_num,samp_num,R_vector_up,R_vector_lo,sum_R,I,M,K_link,Mylyr); 
        loss(end+1) = mean(mean((M_y_sum - Y)'.^2, 2)); % Y_batch' 和 M_y 为 M*batch_size； sum((Y_batch' - M_y).^2, 1)为1*batch_size; loss 为一个标量
     end
    % disp(loss)

    %% extra跳出条件
    if ep > 500 % 500次迭代后再判断是否跳出
        min_loss =  min(loss(end-5:end));
        max_loss =  max(loss(end-5:end));
    
        if max_loss-min_loss < 1e-100
            break;
        end
    end
end

end

function [yy,R_yl,R_yr,yl_R_up,yl_R_lo,yr_R_up,yr_R_lo,y_Mylyr,y] = nonlinearFunction_(con_par, qlqr, X_batch,rule_num,batch_size,R_vector_up,R_vector_lo,sum_R,I,M,K_link,Mylyr)
% format long;
% format long;
% 
% R_vector_up = squeeze(R_vector_up)'; % samp_num *rule_num
% R_vector_lo = squeeze(R_vector_lo)';

yl_list= zeros(rule_num, batch_size);
yr_list= zeros(rule_num, batch_size);

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
    % Type Reduction -----> yleft, yright
    yleft(:,iii) = ((1-qlqr(iii,1))*sum(reshape(R_vector_lo(:,iii,:), [rule_num, batch_size]).* yl_list, 1) + qlqr(iii,1)*sum(reshape(R_vector_up(:,iii,:), [rule_num, batch_size]) .* yl_list,1))./sum_R(iii,:);%samp_num*M 注意：加'.',e.g. '. * ' 和'./'是指矩阵的对应元素相乘除，结果仍为原矩阵大小
    yright(:,iii) = ((1-qlqr(iii,2))*sum(reshape(R_vector_lo(:,iii,:), [rule_num, batch_size]) .* yr_list, 1) + qlqr(iii,2)*sum(reshape(R_vector_up(:,iii,:), [rule_num, batch_size]) .* yr_list,1))./sum_R(iii,:);%samp_num*M

    R_yl(iii,:) = (sum(reshape(R_vector_up(:,iii,:), [rule_num, batch_size]) .* yl_list,1) - sum(reshape(R_vector_lo(:,iii,:), [rule_num, batch_size]) .* yl_list, 1));% 1* batch_size, ql的分子项
    R_yr(iii,:) = (sum(reshape(R_vector_up(:,iii,:), [rule_num, batch_size]) .* yr_list,1) - sum(reshape(R_vector_lo(:,iii,:), [rule_num, batch_size]) .* yr_list, 1));% 1* batch_size, qr的分子项
    
    yl_R_up(iii,:,:) = qlqr(iii,1) .* yl_list - yleft(:,iii)'; % yl_R_up为rule_num * batch_size, 这是delta_yl_R_up的分子项；yl_list为rule_num * batch_size；yleft为1 * batch_size 
    yl_R_lo(iii,:,:) = (1-qlqr(iii,1)).* yl_list - yleft(:,iii)'; % yl_R_lo为rule_num * batch_size, 这是delta_yl_R_lo的分子项；
    
    yr_R_up(iii,:,:) = qlqr(iii,2).* yr_list - yright(:,iii)'; % yr_R_up为rule_num * batch_size, 这是delta_yr_R_up的分子项；yr_list为rule_num * batch_size；yright为1 * batch_size 
    yr_R_lo(iii,:,:) = (1-qlqr(iii,2)).* yr_list - yright(:,iii)'; % yr_R_lo为rule_num * batch_size, 这是delta_yr_R_lo的分子项；
    y_Mylyr(:,iii) = yleft(:,iii) -yright(:,iii); %samp_num*M

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
