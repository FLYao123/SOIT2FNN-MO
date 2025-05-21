function [loss, con_par, M_qlqr] = Rule_growing_adam(X,Y, ant_par,con_par, M_qlqr, I,M, ep_max,batch_size)
% function [ant_par,con_par,M_qlqr,loss] = stage2_all_pars_tuning(X,Y, ant_par,con_par,M_qlqr,I,M,lr, ep_max,batch_size)

rule_num = size(ant_par,4);
samp_num = size(X,1);

%% Hyperparameters for Adam
learning_rate = 0.001;
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;

% Initialize parameters
params = zeros(1, M*(2*rule_num*(I+1) +2));
grads = zeros(1, M*(2*rule_num*(I+1) +2));

m = zeros(size(params));
v = zeros(size(params));
t = 0;

for i = 1:M
    for ii = 1: rule_num
        params((i-1)*(2*rule_num*(I+1) +2)+(ii-1)*2*(I+1)+1:(i-1)*(2*rule_num*(I+1) +2)+((ii-1)*2 +1)*(I+1)) = con_par(:,i,1,ii);%c
        params((i-1)*(2*rule_num*(I+1) +2)+((ii-1)*2 +1)*(I+1)+1:(i-1)*(2*rule_num*(I+1) +2)+ii*2*(I+1)) = con_par(:,i,2,ii);%s
    end
    params((i-1)*(2*rule_num*(I+1) +2)+ii*2*(I+1)+1:(i-1)*(2*rule_num*(I+1) +2)+ii*2*(I+1)+2) = M_qlqr(i,:);
end
%% Computing Antecedent 
jump_out = 0;
for ep = 1 : ep_max

    %% Sample a batch
    indices = randperm(samp_num, batch_size);
    X_batch = X(indices,:);
    Y_batch = Y(indices,:); % actual value  batch_size * M

    [R_vector_up,R_vector_lo,sum_R] = Computing_Antecedent (X_batch,I,rule_num,batch_size,ant_par);% R_vector_up为batch_size * rule_num；sum_R 为 batch_size *1 矩阵
    
    M_y = zeros(M,batch_size);
    R_yl = zeros(M,batch_size);
    R_yr = zeros(M,batch_size);

    for i = 1 : M      
      [y_tmp,R_yl_tmp,R_yr_tmp] = nonlinearFunction_(con_par(:,i,:,:), M_qlqr(i,:), X_batch,rule_num,batch_size,R_vector_up,R_vector_lo,sum_R,I);
      M_y(i,:) = y_tmp; % M * batch_size
      R_yl(i,:) = R_yl_tmp;% M * batch_size
      R_yr(i,:) = R_yr_tmp;% M * batch_size
    end

    % 更新参数 using Adam
    %****preparation****
    [delta_c, delta_s, delta_ql, delta_qr] = stage1_backpropagation_delta(X_batch, Y_batch, M_y, R_vector_up,R_vector_lo,sum_R, M_qlqr,I,rule_num,R_yl,R_yr);
    for i = 1:M
        for ii = 1: rule_num
            grads((i-1)*(2*rule_num*(I+1) +2)+(ii-1)*2*(I+1)+1:(i-1)*(2*rule_num*(I+1) +2)+((ii-1)*2 +1)*(I+1)) = delta_c(ii,:,i);%c
            grads((i-1)*(2*rule_num*(I+1) +2)+((ii-1)*2 +1)*(I+1)+1:(i-1)*(2*rule_num*(I+1) +2)+ii*2*(I+1)) = delta_s(ii,:,i);%
        end
        grads((i-1)*(2*rule_num*(I+1) +2)+ii*2*(I+1)+1) = delta_ql(i);
        grads((i-1)*(2*rule_num*(I+1) +2)+ii*2*(I+1)+2) = delta_qr(i);
    end
    %%***Adam***
    t = t +1;

    m = beta1 * m + (1 - beta1) * grads;
    v = beta2 * v + (1 - beta2) * (grads.^2);

    % Correct bias in first and second moments
    m_hat = m / (1 - beta1^t);
    v_hat = v / (1 - beta2^t);

    % Update parameters
    params = params - learning_rate * m_hat ./ (sqrt(v_hat) + epsilon);

    % % Display loss for monitoring
    % fprintf('Epoch %d, Loss: %f\n', epoch, loss);

    for i = 1:M
        for ii = 1: rule_num
            con_par(:,i,1,ii) = params((i-1)*(2*rule_num*(I+1) +2)+(ii-1)*2*(I+1)+1:(i-1)*(2*rule_num*(I+1) +2)+((ii-1)*2 +1)*(I+1));%c
            con_par(:,i,2,ii) = params((i-1)*(2*rule_num*(I+1) +2)+((ii-1)*2 +1)*(I+1)+1:(i-1)*(2*rule_num*(I+1) +2)+ii*2*(I+1));%s
        end
        M_qlqr(i,:) = params((i-1)*(2*rule_num*(I+1) +2)+ii*2*(I+1)+1:(i-1)*(2*rule_num*(I+1) +2)+ii*2*(I+1)+2);
    end
   

%*****************Computing Loss***********************
    [R_vector_up,R_vector_lo,sum_R] = Computing_Antecedent (X,I,rule_num,samp_num,ant_par);% R_vector_up为batch_size * rule_num；sum_R 为 batch_size *1 矩阵
    
    for i = 1 : M      
      [y_tmp,~,~] = nonlinearFunction_(con_par(:,i,:,:), M_qlqr(i,:), X,rule_num,samp_num,R_vector_up,R_vector_lo,sum_R,I);
      M_y_sum(i,:) = y_tmp; % M * batch_size
    end
    xxx = mean((M_y_sum - Y').^2, 2);
    loss(ep) = mean(mean((M_y_sum - Y').^2, 2)); % Y_batch' 和 M_y 为 M*batch_size； mean((Y_batch' - M_y).^2, 2)为M*1; 
    % disp(loss)

    %% extra跳出条件
    if ep > 500 % 500次迭代后再判断是否跳出
        min_loss =  min(loss(end-20:end));
        max_loss =  max(loss(end-20:end));
    
        if max_loss-min_loss < 1e-5
            % jump_out =1;
            break;
        end
    end

end
figure(100)
plot(loss)


end


function [y,R_yl,R_yr] = nonlinearFunction_(con_par, qlqr, X_batch,rule_num,batch_size,R_vector_up,R_vector_lo,sum_R,I)

yl_list= zeros(rule_num, batch_size);
yr_list= zeros(rule_num, batch_size);
ql = qlqr(1);
qr = qlqr(2);
for ii = 1:rule_num
  c = con_par(:,:,1,ii)';
  s = con_par(:,:,2,ii)';
  X_with_bias = [ones(1, batch_size); X_batch'];
  yl = c * X_with_bias - s * abs(X_with_bias);
  yr = c * X_with_bias + s * abs(X_with_bias);
  yl_list(ii, :) = yl;% rule_num * batch_size
  yr_list(ii, :) = yr;% rule_num * batch_size
end
  % Type Reduction -----> yleft, yright
yleft = ((1-ql)*sum(R_vector_lo' .* yl_list, 1) + ql*sum(R_vector_up' .* yl_list,1))./sum_R';%1 * batch_size 注意：加'.',e.g. '. * ' 和'./'是指矩阵的对应元素相乘除，结果仍为原矩阵大小
yright= ((1-qr)*sum(R_vector_lo' .* yr_list, 1) + qr*sum(R_vector_up' .* yr_list,1))./sum_R';%1 * batch_size

  %% Defuzzification -----> y
y = (0.5*(yleft + yright))'; 
R_yl = (sum(R_vector_up' .* yl_list,1) - sum(R_vector_lo' .* yl_list, 1));% 1* batch_size
R_yr = (sum(R_vector_up' .* yr_list,1) - sum(R_vector_lo' .* yr_list, 1));% 1* batch_size

end