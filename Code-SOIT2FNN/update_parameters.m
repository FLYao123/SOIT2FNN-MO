function [ant_par, con_par, M_qlqr,Mant,Mylyr, K_link] = update_parameters(ant_par, con_par, M_qlqr, Mant, K_link, delta_c, delta_s, delta_ql, delta_qr, delta_m1, delta_m2, delta_sig, delta_m_mid,delta_sig_mid,lr,rule_num,I,delta_Mylyr,Mylyr,delta_E_l)
% format long;
% ant_par 为 1*I *3 * rule_num; 则 m_{j1}^i = ant_par(:,j,1,i),m_{j2}^i = ant_par(:,j,2,i),sig_{j}^i = ant_par(:,j,3,i)
% con_par 为 (I +1)*M *2 *rule_num, 即，前两维是数组，第三维是哪一个参数，最后一维是哪一个（RULE）
% M_qlqr为 M *2 矩阵，每一行对应一个输出，第一列对应ql，第二列对于qr
% Mylyr 为 1*M
% delta_c, delta_s 为  rule_num * (I+1) * M;
% delta_ql, delta_qr, delta_Mylyr 为 M* 1
% % delta_m1, delta_m2, delta_sig 为 rule_num * I *1
% M_qlqr(:,1) = M_qlqr(:,1) - lr .* delta_ql;
% M_qlqr(:,2) = M_qlqr(:,2) - lr .* delta_qr;
% xx=con_par(1,:,1,1);
% yy=delta_c(1,1,:);
% yy=squeeze(delta_c(1,1,:));
% Mant  M*I*2  第一维是 M，第二维是I，第三维是哪一个参数(c_mid or sigma_mid)
% delta_m_mid,delta_sig_mid 为 I*M
% delta_E_l is 1*1

M_qlqr(:,1) = max(0, min(1, M_qlqr(:,1) - lr .* delta_ql));
M_qlqr(:,2) = max(0, min(1, M_qlqr(:,2) - lr .* delta_qr));
% Mylyr = max(0, min(1, Mylyr - lr .* delta_Mylyr'));
Mant(:,:,1) = max(0.01,min(100,Mant(:,:,1) - lr .* delta_m_mid'));
Mant(:,:,2) = max(0.01, min(100,Mant(:,:,2) - lr .* delta_sig_mid'));
K_link = max(0, min(1,K_link - lr.*delta_E_l));

for i = 1: rule_num
    con_par(1,:,1,i) = con_par(1,:,1,i) -lr .* squeeze(delta_c(i,1,:))';  % c0
    con_par(1,:,2,i) = con_par(1,:,2,i) -lr .* squeeze(delta_s(i,1,:))';  % s0
    for j = 1: I
        ant_par(1,j,1,i) = ant_par(:,j,1,i) - lr .* delta_m1(i,j,1);
        ant_par(1,j,2,i) = ant_par(:,j,2,i) - lr .* delta_m2(i,j,1);
        ant_par(1,j,3,i) = max(0.001, ant_par(:,j,3,i) - lr .* delta_sig(i,j,1));

        con_par(j+1,:,1,i) = con_par(j+1,:,1,i) -lr .* squeeze(delta_c(i,j+1,:))';  % c_1 - c_I
        con_par(j+1,:,2,i) = con_par(j+1,:,2,i) -lr .* squeeze(delta_s(i,j+1,:))';  % s_1 - s_I
    end
end

end