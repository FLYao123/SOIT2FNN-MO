function [res, con_par, M_qlqr,Mylyr,Mant, K_link] = Rule_growing(X,Y, ant_par,Mant, con_par, Mqlqr, I,M, K_link,Mylyr)
rule_num = size(ant_par,4);
samp_num = size(X,1);

%% Computing Antecedent 
%R_vector_up_mid,R_vector_lo_mid: 
%sum_R: M*samp_num

% [R_vector_up_mid,R_vector_lo_mid,sum_R] = Computing_Antecedent (X,I,M, rule_num,samp_num,ant_par,Mant);

%% Computing consequent parameters in stage one

[optimizedParams,resnorm_tmp] = fitting_conpara(X,Y,I,con_par,Mqlqr,rule_num,samp_num,M,K_link,Mylyr,ant_par,Mant);

% [optimizedParams,resnorm_tmp] = fitting_conpara(X,Y,I,con_par,Mqlqr,rule_num,samp_num,R_vector_up_mid,R_vector_lo_mid,sum_R,M,K_link,Mylyr);

    res = resnorm_tmp/samp_num;


TMP_M = 2*rule_num*(I+1);
TMP_C = 2*(I+1);
c_num = I+1;

for i =1:M
  for ii = 1:rule_num
    con_par(:,i,1,ii) = optimizedParams((i-1)*TMP_M + (ii-1)*TMP_C + 1: (i-1)*TMP_M + (ii-1)*TMP_C + c_num);%C
    con_par(:,i,2,ii) = optimizedParams((i-1)*TMP_M + (ii-1)*TMP_C + c_num + 1 : (i-1)*TMP_M + (ii-1)*TMP_C + 2*c_num);%S    
  end
end

for i = 1 :M
    M_qlqr(i,1) = optimizedParams(M*TMP_M + (i-1)*2+1);%ql
    M_qlqr(i,2) = optimizedParams(M*TMP_M + (i-1)*2+2);
%     Mylyr(i) = optimizedParams(M*TMP_M + (i-1)*2+3);% Mylyr
end

for i = 1 :M
    Mant(i,:,1) = optimizedParams(M*(TMP_M + 2 ) + (i-1)*2*I + 1 : M*(TMP_M + 2 ) + (i-1)*2*I + I);% mean of Mid
    Mant(i,:,2) = optimizedParams(M*(TMP_M + 2 ) + (i-1)*2*I + I + 1 : M*(TMP_M + 2 ) + (i-1)*I*2 + 2*I);% STD of Mid
end
K_link = optimizedParams(M*(2*rule_num*(I+1) +2 +2*I) +1);
%√É‚Äî√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚??
%√É‚Äî√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?? Not Finish √Ø¬º¬ç√Ø¬º¬ç√Ø¬º¬çE_significance  √¶≈°‚Äö√¶‚?î¬∂√ß‚?ù¬®√¶¬Æ‚?π√•¬∑¬Æres√§¬ª¬£√¶‚Ä∫¬ø√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚??
%√É‚Äî√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚?î√É‚??

% end

end

