function     [delta_c, delta_s, delta_ql, delta_qr, delta_m1, delta_m2, delta_sig, delta_m_midd, delta_sig_midd,delta_Mylyr,delta_E_l] = backpropagation_delta(X_batch, Y_batch, M_y, R_vector_up,R_vector_lo,sum_R, M_qlqr,I,M,rule_num,R_yl,R_yr,yl_R_up,yl_R_lo,yr_R_up,yr_R_lo,ant_par,Mant,K_link,Mylyr,y_Mylyr,M_y_tmp)
% format long;
% X_batch �? batch_size * I 矩阵
% Y_batch �? batch_size * M 矩阵
% M_y �? samp_num*M 矩阵
% R_vector_up �? R_vector_lo 为rule_num *M * batch_size矩阵
% R_yl �? R_yr �? M*batch_size 矩阵
% sum_R �?  M*batch_size 矩阵
% M_qlqr�? M *2 矩阵，每�?行对应一个输出，第一列对应ql，第二列对于qr
% I 为输入维�?
% M 为输出维�?
% rule_num为截止到目前的规则数�?
% yl_R_up,yl_R_lo,yr_R_up,yr_R_lo 均为三维M*rule_num * batch_size
% ant_par �? 1*I *3 * rule_num; �? m_{j1}^i = ant_par(:,j,1,i),m_{j2}^i = ant_par(:,j,2,i),sig_{j}^i = ant_par(:,j,3,i)
% Mant  第一维是 M，第二维是I，第三维是哪�?个参�?
% x=R_vector_lo(1,:,:);

sum_R = sum_R';
batch_size = size(X_batch,1);

% delta_E_y = (M_y - Y_batch'); % M*batch_size

delta_E_y = zeros (M,batch_size);% M*batch_size
for t = 1:M
     for tt = t:M
         delta_E_y(t,:)  = delta_E_y(t,:) + (M_y(:,tt)-Y_batch(:,tt))'.*((1-K_link)*power(K_link, tt-t));
    end
end
delta_E_l = zeros (1,batch_size);% M*batch_size
delta_y_l = zeros (1,batch_size);
for t = 1:M
     for tt = 1:t
         if tt-1 ~= 0
             delta_y_l = delta_y_l + (t-tt+1)*power(K_link, t-tt).*(M_y_tmp(:,tt-1)-M_y_tmp(:,tt))';
         else
             delta_y_l = delta_y_l + (t-tt+1)*power(K_link, t-tt).*(X_batch(:,end)-M_y_tmp(:,1))';
         end
     end
     delta_E_l = delta_E_l + (M_y(:,t)-Y_batch(:,t))'.*delta_y_l;
end
delta_E_l = mean(delta_E_l);  % result dimension is 1*1
delta_y_yl = Mylyr';
delta_y_yr = (1-Mylyr)';

delta_y_Mylyr = y_Mylyr;%samp_num*M

delta_yl_ql =  R_yl./sum_R';% M* batch_size
delta_yr_qr =  R_yr./sum_R';% M* batch_size
delta_ql = mean(delta_E_y .*(delta_y_yl.*delta_yl_ql),2); % M* 1;其中delta_E_y .*(delta_y_yl.*delta_yl_ql)为M*batch_size
delta_qr = mean(delta_E_y .*(delta_y_yl.*delta_yr_qr),2); % M* 1;其中delta_E_y .*(delta_y_yl.*delta_yr_qr)为M*batch_size

delta_Mylyr = mean(delta_E_y .*delta_y_Mylyr',2); % M* 1;其中delta_E_y .*(delta_y_yl.*delta_yl_ql)为M*batch_size

for i = 1 : rule_num
    

    delta_yl_wl = ((1-M_qlqr(:,1)) .*reshape(R_vector_lo(i,:,:), [M, batch_size]) + M_qlqr(:,1) .*reshape(R_vector_up(i,:,:), [M, batch_size])) ./sum_R'; % M* batch_size, M_qlqr(:,1) �? 两个输出对应的ql
    delta_yr_wr = ((1-M_qlqr(:,2)) .*reshape(R_vector_lo(i,:,:), [M, batch_size]) + M_qlqr(:,2) .*reshape(R_vector_up(i,:,:), [M, batch_size])) ./sum_R';% M* batch_size, M_qlqr(:,2) �? 两个输出对应的qr

    delta_yl_R_up = reshape(yl_R_up(:,i,:), [M, batch_size])./sum_R';%delta_yl_R_up为M* batch_size；其中yl_R_up(:,i,:)为三维M*1 * batch_size，squeeze(yl_R_up(:,i,:))为二维M* batch_size
    delta_yl_R_lo = reshape(yl_R_lo(:,i,:), [M, batch_size])./sum_R';
    delta_yr_R_up = reshape(yr_R_up(:,i,:), [M, batch_size])./sum_R';
    delta_yr_R_lo = reshape(yr_R_lo(:,i,:), [M, batch_size])./sum_R';

    delta_E_R_up = delta_E_y .* (delta_y_yl .* delta_yl_R_up + delta_y_yr .* delta_yr_R_up); % M*batch_size
    delta_E_R_lo = delta_E_y .* (delta_y_yl .* delta_yl_R_lo + delta_y_yr .* delta_yr_R_lo); 
 
    
%     %% Here, 此时�?要将 M 个输出对应的返回梯度结合起来
%     delta_E_R_up_mean = sum(delta_E_R_up, 1); % 1*batch_size
%     delta_E_R_lo_mean = sum(delta_E_R_lo, 1);

    delta_E_R_up_mean = delta_E_R_up; % M*batch_size
    delta_E_R_lo_mean = delta_E_R_lo; % M*batch_size

    %% **********************************************

    delta_c (i,1,:) = mean(delta_E_y .* (delta_y_yl .* delta_yl_wl .* 1 + delta_y_yr .* delta_yr_wr .* 1),2); %M*1;完成后�?�共�? rule_num * (I+1)*M;其中delta_E_y .* (delta_y_yl .* delta_yl_wl .* 1 + delta_y_yr .* delta_yr_wr .* 1)为M * batch_size
    delta_s (i,1,:) = mean(delta_E_y .* (-delta_y_yl .* delta_yl_wl .* 1 + delta_y_yr .* delta_yr_wr .* 1),2); % M*1;完成后�?�共�? rule_num * (I+1)*M;其中delta_E_y .* (delta_y_yl .* delta_yl_wl .* 1 + delta_y_yr .* delta_yr_wr .* 1)为M * batch_size

    for j = 1: I

        for m = 1: M
            delta_R_up_m_mid(i,j,m,:) = ((squeeze(R_vector_up(i,m,:)).^2) .* (X_batch(:,j) - Mant(m,j,1)))' / (Mant(m,j,2).^2); %�?终为rule_num*I*M*batch_size
            delta_R_lo_m_mid(i,j,m,:) = ((squeeze(R_vector_lo(i,m,:)).^2) .* (X_batch(:,j) - Mant(m,j,1)))' / (Mant(m,j,2).^2); %�?终为rule_num*I*M*batch_size
            delta_Rup_sig_mid(i,j,m,:) = ((squeeze(R_vector_up(i,m,:)).^2) .* (X_batch(:,j) - Mant(m,j,1)).^2)' / (Mant(m,j,2).^3);
            delta_Rlo_sig_mid(i,j,m,:) = ((squeeze(R_vector_lo(i,m,:)).^2) .* (X_batch(:,j) - Mant(m,j,1)).^2)' / (Mant(m,j,2).^3);
    
            delta_m_mid(i,j,m,:) = delta_E_R_up(m,:) .*  squeeze(delta_R_up_m_mid(i,j,m,:))'  + delta_E_R_lo(m,:) .*  squeeze(delta_R_lo_m_mid(i,j,m,:))'; % rule_num*I*M*batch_size
            delta_sig_mid(i,j,m,:) = delta_E_R_up(m,:) .*  squeeze(delta_Rup_sig_mid(i,j,m,:))'  + delta_E_R_lo(m,:) .*  squeeze(delta_Rlo_sig_mid(i,j,m,:))'; % rule_num*I*M*batch_size 
        end

        %% **************************************************

        delta_wl_c = X_batch(:,j); %batch_size * 1
        delta_wl_s = -abs(X_batch(:,j)); %batch_size * 1
        delta_wr_c = X_batch(:,j); %batch_size * 1
        delta_wr_s = abs(X_batch(:,j)); %batch_size * 1

        %% consequent parameters 
        delta_c(i,j+1,:) = mean(delta_E_y .* (delta_y_yl .* delta_yl_wl .* delta_wl_c' + delta_y_yr .* delta_yr_wr .* delta_wr_c'),2); % M*1;完成后�?�共�? rule_num * (I+1)*M; 其中delta_E_y .* (delta_y_yl .* delta_yl_wl .* delta_wl_c' + delta_y_yr .* delta_yr_wr .* delta_wr_c')为M * batch_size
        delta_s(i,j+1,:) = mean(delta_E_y .* (-delta_y_yl .* delta_yl_wl .* delta_wl_s' + delta_y_yr .* delta_yr_wr .* delta_wr_s'),2); % M*1;完成后�?�共�? rule_num * (I+1)*M; 其中delta_E_y .* (delta_y_yl .* delta_yl_wl .* delta_wl_s' + delta_y_yr .* delta_yr_wr .* delta_wr_s')为M * batch_size
        for m = 1: M
        for ii = 1: batch_size

          %******m1*************

          if X_batch(ii,j) <= ant_par(:,j,1,i)              
            delta_R_up_m1(i,j,ii) = R_vector_up(i,m,ii)^2 * (X_batch(ii,j) - ant_par(:,j,1,i)) / ant_par(:,j,3,i)^2; 
          else
            delta_R_up_m1(i,j,ii) = 0;
          end

          if X_batch(ii,j) > (ant_par(:,j,1,i)+ant_par(:,j,2,i))/2
            delta_R_lo_m1(i,j,ii) = R_vector_lo(i,m,ii)^2 * (X_batch(ii,j) - ant_par(:,j,1,i)) / ant_par(:,j,3,i)^2; 
          else
            delta_R_lo_m1(i,j,ii) = 0;
          end

          %******m2*************

          if X_batch(ii,j) > ant_par(:,j,2,i)
            delta_R_up_m2(i,j,ii) = R_vector_up(i,m,ii)^2 * (X_batch(ii,j) - ant_par(:,j,2,i)) / ant_par(:,j,3,i)^2; 
          else
            delta_R_up_m2(i,j,ii) = 0;
          end

          if X_batch(ii,j) <= (ant_par(:,j,1,i)+ant_par(:,j,2,i))/2
            delta_R_lo_m2(i,j,ii) = R_vector_lo(i,m,ii)^2 * (X_batch(ii,j) - ant_par(:,j,2,i)) / ant_par(:,j,3,i)^2; 
          else
            delta_R_lo_m2(i,j,ii) = 0;
          end

           %******sig*************

          if X_batch(ii,j) < ant_par(:,j,1,i)
            delta_R_up_sig(i,j,ii) = R_vector_up(i,m,ii)^2 * (X_batch(ii,j) - ant_par(:,j,1,i))^2 / ant_par(:,j,3,i)^3; 
          elseif X_batch(ii,j) > ant_par(:,j,2,i)
            delta_R_up_sig(i,j,ii) = R_vector_up(i,m,ii)^2 * (X_batch(ii,j) - ant_par(:,j,2,i))^2 / ant_par(:,j,3,i)^3; 
          else
            delta_R_up_sig(i,j,ii) = 0;
          end

          if X_batch(ii,j) <= (ant_par(:,j,1,i)+ant_par(:,j,2,i))/2
            delta_R_lo_sig(i,j,ii) = R_vector_lo(i,m,ii)^2 * (X_batch(ii,j) - ant_par(:,j,2,i))^2 / ant_par(:,j,3,i)^3; 
%           elseif X_batch(ii,j) > (ant_par(:,j,1,i)+ant_par(:,j,2,i))/2
          else
            delta_R_lo_sig(i,j,ii) = R_vector_lo(i,m,ii)^2 * (X_batch(ii,j) - ant_par(:,j,1,i))^2 / ant_par(:,j,3,i)^3; 
          end
        end 

%         delta_m1(i,j,:) = mean(delta_E_R_up_mean .*  squeeze(delta_R_up_m1(i,j,:))'  + delta_E_R_lo_mean .*  squeeze(delta_R_lo_m1(i,j,:))',2); % 完成后�?�共�? rule_num * I *1;  delta_R_lo_m1(i,:,:) �? 1*1*batch_size; squeeze(delta_R_lo_m1(i,:,:)�?1*batch_size
%         delta_m2(i,j,:) = mean(delta_E_R_up_mean .*  squeeze(delta_R_up_m2(i,j,:))' + delta_E_R_lo_mean .* squeeze(delta_R_lo_m2(i,j,:))',2);
%         delta_sig(i,j,:) = mean(delta_E_R_up_mean .*  squeeze(delta_R_up_sig(i,j,:))' + delta_E_R_lo_mean .*  squeeze(delta_R_lo_sig(i,j,:))',2);

            delta_m1(i,j,m,:) = delta_E_R_up_mean(m,:) .*  squeeze(delta_R_up_m1(i,j,:))'  + delta_E_R_lo_mean(m,:) .*  squeeze(delta_R_lo_m1(i,j,:))'; 
            delta_m2(i,j,m,:) = delta_E_R_up_mean(m,:) .*  squeeze(delta_R_up_m2(i,j,:))' + delta_E_R_lo_mean(m,:) .* squeeze(delta_R_lo_m2(i,j,:))';
            delta_sig(i,j,m,:) = delta_E_R_up_mean(m,:) .*  squeeze(delta_R_up_sig(i,j,:))' + delta_E_R_lo_mean(m,:) .*  squeeze(delta_R_lo_sig(i,j,:))';
        end

    end
end
        delta_m1 = reshape(mean(sum(delta_m1, 3),4), [rule_num, I]);% rule_num * I
        delta_m2 = reshape(mean(sum(delta_m2, 3),4), [rule_num, I]);% rule_num * I
        delta_sig = reshape(mean(sum(delta_sig, 3),4), [rule_num, I]);%rule_num * I

        delta_m_midd = reshape(mean(sum(delta_m_mid, 1),4), [I, M]);% I *M
        delta_sig_midd = reshape(mean(sum(delta_m_mid, 1),4), [I, M]);% I *M
end