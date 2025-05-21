%---------------------------------
% IT2FNNP1  : Single-optput Parallel prediction scheme (PM)
% IT2FNNP2 : Single-optput serial prediction scheme (SW)
% IT2FNNP3  : Multi-optput  prediction scheme (MO)
%---------------------------------

clc;
clear all;
close all;

% % æŒ‡å®šç�?�¸å¯¹è·¯å¾�?�çš�?�æ�?��?�ä»¶å�?
subfolder_data = 'IT2FNN_normalization_data';

% % æž„å»ºæ�?��?�ä»¶å�??

% filename_train = fullfile(subfolder_data, 'unmets_HWM_train.mat');
% load(filename_train)
% inputs = trainData(:, 4:15);
% 
% X = inputs(:,1:9);
% Y = inputs(:,10:12);

% æž„å»ºæ�?��?�ä»¶å�??
filename_train = fullfile(subfolder_data, 'Chaotic_train_9_3_0.300000.mat');
load(filename_train)
X = train_input';
Y = train_output';


% % 
% dataTable = readtable('Timestamp.csv');
% Timestamps = dataTable.Timestamps;
% save('Timestamps.mat', 'Timestamps');

% load('prs_30.mat');
% load('Timestamps.mat')
% inputs = prs(:, 1:12);
% 
% % plot(inputs(:,1))
% X = inputs(:,1:9);
% Y = inputs(:,10:12);
% 
% % % Feature Extraction
% X = Feature_extraction(X,Timestamps);
% 
% 
% min_vals = min(X);
% max_vals = max(X);
% 
% X = (X - min_vals) ./ (max_vals - min_vals);

%% Clustering 
%---------------------------------
% Cnum : the number of expected clusterings
% Ctype=1 : K-means
% Ctype=2 : Gaussian Mixture Model ï¼ˆGMMï¼?
% Ctype=3ï¼? Fuzzy C means (FCM)
% Ctype=4 : Possibilistic C-Means ï¼ˆPCMï¼?
% Ctype=5 : Possibilistic Fuzzy C-Meansï¼ˆPFCMï¼?
% Ctype=6 : Enhanced Possibilistic Fuzzy C-Meansï¼ˆEPFCMï¼?
% Ctype=7 : Interval Type 2 Possibilistic Fuzzy C-Meansï¼ˆIT2-EPFCMï¼?
%---------------------------------

Ctype = 3;
Cnum = 5; % 10, 15, 20, 25, 30, ...
% 
% % % % ******* CLUSTERING  & SAVE******
% [C] = clustering (X, Ctype, Cnum);
% save('pr_12_6_clustering_FCM_5.mat', 'C');

% %% ******* CLUSTERING  & LOADING******
% load('pr_9_3_clustering_FCM_8.mat');
% load('pr_9HWM_3_clustering_FCM_8.mat');
% % cl=ca(ctype,trndat,cnum); % do the cluster analysis

filename_cluster = fullfile(subfolder_data, 'Chaotic_9_3_clustering_FCM_5_0.3.mat');
filename_cluster_mid = fullfile(subfolder_data, 'Chaotic_9_3_clustering_FCM_mid_0.3.mat');
% filename_cluster = fullfile(subfolder_data, 'unmets_9_3_clustering_FCM_5.mat');
% filename_cluster_mid = fullfile(subfolder_data, 'unmets_9_3_clustering_FCM_10_mid.mat');
load(filename_cluster);
load(filename_cluster_mid);

I = size(X,2);
MM = 1;
M = 1;

uncertainty = 0.1;
cc_up = C_x.center * (1+uncertainty); % Cnum * I 
cc_lo = C_x.center * (1-uncertainty);
cc_mid = C_mid.center;% 1 * I 
ssigma = C_x.std; % Cnum * I 
ssigma_mid = C_mid.std; % 1 * I 

res_last = ones(1,MM)*1000;
tic;

for iii = 1 : MM
    c_up = cc_up;
    c_lo = cc_lo;
    sigma = ssigma;
    c_mid = cc_mid ;% 1 * I 
    sigma_mid  = ssigma_mid;

    %% Initialize the antecedent parameters (Convert the clustering parameters to interval MF)
    buffer_size = size(c_lo,1);
    Rule_num = 0; % åˆå§‹åŒ�?�ä¸�?0æ¡è§„åˆ™ï¼Œæ³¨æ„è¿™ä¸ªä¸èƒ½å’ŒRule_growingä¸­çš„rule_numç›¸æ�?�¿æ¢ï¼Œå®žé™…ä¸Šrule_num = Rule_num +1,å› ä¸ºåœ¨å�?�½æ�?�°é�?�Œé¢å®žé™…ä¸Šæ˜¯å�?�è®¾æ·»åŠ ä¸€æ¡è§„å�??
    ant_par = [];
    con_par = [];
    stage2_flag = 0; % åˆ¤æ–­æ˜¯ä¸æ˜¯è¿žç»­ä¸¤æ¬¡è¿�?�å�?�¥stage2ï¼Œå³stage 2 è°ƒå‚åŽï¼Œä»ç�?�¶æ²¡æœ�?�è§�?�åˆ™å¢žåŠ 
%     change_stage2 = 1; % ç»™change_stage1æ¬¡stage2è°ƒå‚æœºä¼šï¼Œå¦�?�æžœchange_stage1æ¬¡ä¹‹åŽä¾ç�?�¶æ²¡æœ�?�è§�?�åˆ™å¢žåŠ ï¼Œç»“æŸã€‚ä»æœªRule_num å·²æ”¶æ�???
%     Mqlqr = [0.5,0.5]; % ql and qr
%     res_last = 1000; %åˆå§‹åŒ�?�ä¸ºä�??ä¸ªå¾ˆå¤§çš„æ�???
    Mqlqr = zeros(M,2) + 0.5; %M*2
    Mylyr = zeros(1,M) + 0.5;

    M_c_mid = repmat(c_mid, M, 1); % M*I
    M_sigma_mid = repmat(sigma_mid, M, 1);% M*I
    Mant = cat(3, M_c_mid, M_sigma_mid); % ç¬¬ä¸€ç»´æ˜�? Mï¼Œç¬¬äºŒç»´æ˜¯Iï¼Œç¬¬ä¸‰ç»´æ˜¯å�?�ªä�??ä¸ªå‚æ�???

    threshold_growing = 0.0025; %0.5 
    threshold_reducing = threshold_growing*1; %0.5  

    K_link = 0.1;
    Flag = 0; % 0ä¸ºåˆå§‹�???; 1ä¸ºè§„åˆ™å¢žåŠ?; 2ä¸ºè§„åˆ™å‡å�??; 3ä¸ºè§„åˆ™ä¸å¢žä¸å‡è·³åˆ°stage2ï¼Œä¼˜åŒ–åŽå�?�ç»™ä¸?æ¬¡æœºä¼?; 4ä¸ºè¿žç»­ä¸¤æ¬¡ä¸å¢žä¸å‡è¡¨ç¤ºç¨³å®šçŠ¶æ€ï¼Œå³ç»“æ�??
    
    lr = 0.003; % learning rate for stage 2
    ep_max = 5000; % maximum episode for stage 2
    batch_size = 64; % batch size for stage 2

    replay_momory = [];% ç”¨äºŽå­˜æ�?�¾å�?� ä¸ºrule_growingè¢«åˆ é™¤çš�?�èšç±»ï¼Œä»¥ä¾¿åŽç»­rule_decreasingæ—¶å�?�æ¢å¤�?

    for epo = 1:100
        % loss = zeros(1,size(c_up,1));
        % res = zeros(1,size(c_up,1));
        con_par_stage1 = [];
        Mqlqr_growing = [];
        res = zeros(1,size(c_up,1));
        Mylyr_growing = [];
        Mant_growing = [];
        K_link_growing = [];

        if isempty(c_up)
            break;
        end
    
    % ****** First stage: Rule growing ******
        for i = 1:size(c_up,1) %é€ä¸ªåŠ è§�?�åˆ™ï¼Œç¬¬ä¸€æ¬¡ä¸€ä¸ªï¼Œç¬¬äºŒæ¬¡ä¸¤ä¸ªï¼Œç›´åˆ°èšç±»ä�??
            seed = 42; % ä½ å¯ä»¥é?‰æ�?�©ä»»ä½�?�æ�?�´æ�?�°ä½œä¸ºç§å­�?
            rng(seed);
            
            % ***********æŒ‰é¡ºåºå°±å¯ä»¥ï¼ŒéšæœºæŠ½æ ·ï¼Œè¿˜éº»çƒ¦Ã�?�Ã�?�Ã�?�Ã�?�Ã�?�Ã�?�Ã�?�Ã�?�Ã�?�Ã�?�Ã�?��??
            ant_tmp_tuple= cat(3, c_lo(i,:), c_up(i,:), sigma(i,:));
            c = randn(I + 1, M);
            s = randn(I + 1, M);
            con_tmp_tuple = cat(3, c,s); % ç¬¬ä¸€ç»´æ˜�? I+1ï¼Œç¬¬äºŒç»´æ˜¯Mï¼Œç¬¬ä¸‰ç»´æ˜¯å�?�ªä�??ä¸ªå‚æ�???
    
            if isempty(ant_par) ==1
                ant_par_tmp = ant_tmp_tuple;
                con_par_tmp = con_tmp_tuple;
            else
                ant_par_tmp = cat(4, ant_par,ant_tmp_tuple);%å›�?�ç»´æ�?�°æ®ï¼Œå�?�ä¸¤ç»´æ˜¯æ�?�°ç»�?�ï¼Œç¬¬ä¸�?�ç»´æ˜¯å�?�ªä¸€ä¸ªå‚æ�?�°ï¼Œæ�??åŽä¸€ç»´æ˜¯å�?�ªä¸€ä¸ªï¼ˆRULEï¼‰å³ï�??1*I *3 * rule_num
                con_par_tmp = cat(4, con_par,con_tmp_tuple);%å›�?�ç»´æ�?�°æ®ï¼Œå�?�ä¸¤ç»´æ˜¯æ�?�°ç»�?�ï¼Œç¬¬ä¸�?�ç»´æ˜¯å�?�ªä¸€ä¸ªå‚æ�?�°ï¼Œæ�??åŽä¸€ç»´æ˜¯å�?�ªä¸€ä¸ªï¼ˆRULEï¼‰å³ï�??(I +1)*M *2 *rule_num
            end
    
            % ***computing the consequent parameters using LSE
            [res_tmp, con_par_tmp, Mqlqr_tmp,Mylyr_tmp,Mant_tmp, K_link_tmp] = Rule_growing(X,Y(:,iii), ant_par_tmp,Mant,con_par_tmp,Mqlqr,I,M, K_link,Mylyr); %æŸå¤±å‡½æ�?��? å�?? è§„åˆ™é‡è¦æ€? ï¼ˆè¿™ä¸¤ä¸ªæš�?�æ�?�¶å�?�è®¾æ˜¯åŒä�??ä¸ªæ ‡å�?��?�score_tmpï¼?
            % loss(i) = loss_tmp;  
            res(i) = mean(res_tmp); 
            con_par_stage1 = cat(5,con_par_stage1, con_par_tmp); %ä¸´æ—¶å­˜å�?�¨æ�???æœ?
            Mqlqr_growing = cat(3,Mqlqr_growing,Mqlqr_tmp);
            Mylyr_growing = cat(3,Mylyr_growing,Mylyr_tmp); 
            Mant_growing  = cat(4,Mant_growing,Mant_tmp);
            K_link_growing  = cat(2,K_link_growing,K_link_tmp);

        end
        
        % *******é€�?�æ�?�©å½�?�å�?�epochç©¶ç«Ÿç•™ä¸‹å�?�ªä¸ªèšç±�?,ç„¶åŽåˆ é™¤è¯¥èšç±»ï¼Œå‰©ä¸�?�çš�?�èšç±»ç»§ç»­epoch*********
        [minres, index] = min(res); 
        rule_growing_res(epo) = res(index);
        % [maxscore, index] = max(score); % ????æœªç¡®å®šï¼ŒåŽç»­å†�??è??,å¯èƒ½ç›´æŽ¥ç�?�¨mseï¼Œä¹Ÿå¯èƒ½ç”¨é�?�è¦�??§æŒ‡æ �?�ï¼Œæš�?�æ�?�¶å�?�è®¾ä¸¤ä¸ªæ˜¯åŒä�??ä¸ªï¼Œæ­¤å¤–ç�?�®å�?�å�?�è®¾æ˜¯maxï¼Œå½“ç�?�¶ä¹Ÿå¯èƒ½æ˜¯minï¼Œè¦æ ¹æ®æ ‡å�?��?�ä¿®æ�?��?
    
        if res_last(iii) - minres >= threshold_growing % æ­¤æ—¶growing rule  ä¹Ÿå¯ä»¥æ¢æˆ? abs(res_last - minres)>= threshold_growingè¯•ä¸€è¯?
            ant_tuple= cat(3, c_lo(index,:), c_up(index,:), sigma(index,:));
            ant_par = cat(4, ant_par,ant_tuple);
            % con_par = cat(4, con_par,con_par_stage1(:,:,:,index));
            con_par = con_par_stage1(:,:,:,:,index);
            Mqlqr = Mqlqr_growing(:,:,index);
            Mylyr = Mylyr_growing(:,:,index);
            Mant = Mant_growing(:,:,:,index);
            K_link = K_link_growing(:,index);
            Rule_num = Rule_num + 1;
            Flag = 1;
            res_last(iii) = minres;
    
        else %æ­¤æ—¶è§�?�åˆ™ä¸å†å¢žåŠ ï¼Œå³ä½¿æŒ�?�é€‰äº�?�æ�??å¥½çš„å·²ç�?�¶ä¸å�?�å¢žåŠ ï¼Œåˆ™å¼?å§‹è°ƒæ�?�´å�?�æ�?�°ï¼Œè¿™æ¬¡è°ƒæ•´å�?�åŽä�??èµ·è°ƒæ�??
%             [ant_par,con_par,Mqlqr,loss] = stage2_all_pars_tuning(X,Y(:,iii), ant_par,con_par,Mqlqr,I,M,lr, ep_max,batch_size);   % Stage 2 è°ƒå�?
%             stage2_flag = stage2_flag +1;
%             res_last = loss;
            if size(ant_par,4) == 0 || size(ant_par,4) == 1
                % flag_rule_reducing = 1;%æ­¤æ—¶ï¼Œè§�?�åˆ™ä¸å¢žä¸å‡ï¼ˆå�?� ä¸ºå°±ä¸€ä¸ªè§„åˆ™ï¼‰ï¼Œå¯ä»¥åˆ°stage2 äº?
                if Flag == 3 
                    Flag = 4;% ä¼˜åŒ–ä�??æ¬¡åŽè¿˜æ˜¯ä¸å¢žä¸å‡ï¼Œå³è¿žç»­ä¸¤æ¬¡ä¸å¢žä¸å�?�ï¼Œåˆ™æŽ¨å‡ºï¼ˆFlag=4ï¼?
                else
                    Flag =3; % æ­¤æ—¶ä¸å¢žä¸å�?�ï¼Œéœ€è¦è¿›å�?�¥stage2 ä¼˜åŒ�?
                end
            else
                [res_reducing_tmp, rule_reducing_index, Mqlqr_red, con_par_red,Mylyr_red,Mant_red,K_link_red] = Rule_Reducing(X,Y(:,iii), ant_par,con_par,Mqlqr,I,M,res_last(iii),Mant, K_link,Mylyr); % æ³¨ï¼šæ¯æ¬¡åªå‡å°�?�ä¸€ä¸ªè§„å�??,åªè¿”å�?�žä¸€ä¸ªres_tmp
                
                if res_reducing_tmp - res_last(iii) >= threshold_reducing % åˆ é™¤è¿™ä¸ªè§„åˆ™å®žé™�?�ä¸Šå½±å�?�å¾ˆå¤§ï¼Œåˆ™ä¸å¯ä»¥åˆ é™�?
                     if Flag == 3 
                         Flag = 4;% ä¼˜åŒ–ä�??æ¬¡åŽè¿˜æ˜¯ä¸å¢žä¸å‡ï¼Œå³è¿žç»­ä¸¤æ¬¡ä¸å¢žä¸å�?�ï¼Œåˆ™æŽ¨å‡ºï¼ˆFlag=4ï¼?
                     else
                         Flag =3; % æ­¤æ—¶ä¸å¢žä¸å�?�ï¼Œéœ€è¦è¿›å�?�¥stage2 ä¼˜åŒ�?
                     end
                else % å¦åˆ™ï¼Œè¯¥è§�?�åˆ™ä¸é‡è¦ï¼Œæˆ�?��??…ä¼šèµ·åˆ°åä½œç�?�¨ï¼Œé�??è¦åˆ é™¤è¿™ä¸ªè§„å�??
                   c_up(end+1,:) =  replay_momory(:,:,1,rule_reducing_index);
                   c_lo(end+1,:) = replay_momory(:,:,2,rule_reducing_index);
                   sigma(end+1,:) = replay_momory(:,:,3,rule_reducing_index);
                   ant_par(:, :, :, rule_reducing_index) = [];
                   % con_par(:, :, :, rule_reducing_index) = [];
                   con_par = con_par_red;
                   Mqlqr = Mqlqr_red;
                   Mylyr = Mylyr_red;
                   Mant = Mant_red;
                   K_link = K_link_red;
    
                   res_last(iii) = res_reducing_tmp;
                   Rule_num = Rule_num - 1;
                   % stage2_flag = 0;
                   Flag = 2;
                end        
           end 
        end
        % if flag_rule_reducing == 1%æ­¤æ—¶è§�?�åˆ™ä¸å†å¢žåŠ ï¼Œä¹Ÿä¸å�?�å�?�å°�?�ï¼Œåˆ™å¼?å§‹è°ƒæ�?�´å�?�æ�?�°ï¼Œè¿™æ¬¡è°ƒæ•´å�?�åŽä�??èµ·è°ƒæ�?? 
        if Flag == 3 || Rule_num == Cnum%æ­¤æ—¶è§�?�åˆ™ä¸å†å¢žåŠ ï¼Œä¹Ÿä¸å�?�å�?�å°�?�ï¼Œåˆ™å¼?å§‹è°ƒæ�?�´å�?�æ�?�°ï¼Œè¿™æ¬¡è°ƒæ•´å�?�åŽä�??èµ·è°ƒæ�?? 
            [ant_par_all_op,Mant_op,con_par_all_op,Mqlqr_all_op,Mylyr_op,K_link_op,loss(iii,:)] = stage2_all_pars_tuning(X,Y(:,iii), ant_par,Mant,con_par,Mqlqr,I,M,lr, ep_max,batch_size, K_link,Mylyr);   % Stage 2 è°ƒå�?
            if loss(iii,end) > res_last(iii)
                msgbox('Stage2 does not work! Please have a check! Keep the previous record!', 'Warning!');
            else
                if  res_last(iii) - loss(iii,end)>threshold_growing
                    msgbox('Stage 2 does not work well! A little help! Maybe not better than add a new rule','Warning!');
                end
                ant_par = ant_par_all_op;
                con_par = con_par_all_op;
                Mant = Mant_op;
                Mqlqr = Mqlqr_all_op;
                Mylyr = Mylyr_op; 
                K_link = K_link_op; 
                res_last(iii) = loss(iii,end);

            end
        end
    
        if Flag == 1 % stage_flag == 1ï¼?2æ—¶ï¼Œä¸éœ€è¦åˆ é™¤èšç±»ï¼Œå�?�ç»™ä¸¤æ¬¡æœºä¼šï¼Œç»™ä¸¤æ¬¡æœºä¼šå�?�Œç»™ä¸?æ¬¡æœºä¼šæ²¡ä»?ä¹ˆåŒºåˆ«å‘€ï¼ŒåŒæ ·çš„åˆå§�?�å�?�æ�?�°ï¼ŒåŒæ ·çš�?�ç½�?�ç»œå�?�Œä¼˜åŒ�?�æ�?�¹å�??
            replay_momory_tmp = cat(3, c_up(index,:),  c_lo(index,:),sigma(index,:));
            c_up(index,:) = []; % ä»Žæ•°æ®ä¸­ç§»é™¤å·²ç»å–è¿�?�çš�?��??¼ï¼Œå³è¿›è¡Œä¸æ�?�¾å�?�žå�?��???
            c_lo(index,:) = []; % ä»Žæ•°æ®ä¸­ç§»é™¤å·²ç»å–è¿�?�çš�?��??¼ï¼Œå³è¿›è¡Œä¸æ�?�¾å�?�žå�?��???
            sigma(index,:) = []; % ä»Žæ•°æ®ä¸­ç§»é™¤å·²ç»å–è¿�?�çš�?��??¼ï¼Œå³è¿›è¡Œä¸æ�?�¾å�?�žå�?��???
            replay_momory = cat(4,replay_momory, replay_momory_tmp); %  äºŒç»´æ•°ç»�?? * å“ªä¸€ä¸ªå…ƒç�?? * rule_num
    
        elseif Flag == 4 % æ­¤æ—¶ï¼Œå³ä½¿stage2æ€»è°ƒå�??2æ¬¡ä¹‹åŽä¹Ÿä¾ç�?�¶ä¸èƒ½å�?�åŠ ruleï¼Œè®¤ä¸ºå·²ç»æ”¶æ�???
            break;
        end
    
    rule_number(iii,epo) = Rule_num;
    end  

    figure(100)    
    plot(rule_number(iii,:),'LineWidth',iii)
    hold on;
    figure(101)    
    plot(loss(iii,:),'LineWidth',iii)
    hold on;

    % æŒ‡å®šç�?�¸å¯¹è·¯å¾�?�çš�?�æ�?��?�ä»¶å�?
    subfolder_weights = 'Weights';
    
    % æž„å»ºæ�?��?�ä»¶å�??
%     filename_weights = fullfile(subfolder_weights, sprintf('IT2FNNP2_%d_unmets_weights_I%d_M%d_C%d_Tg%2f_Tr%2f.mat', iii,I, M,Cnum,threshold_growing,threshold_reducing));
     filename_weights = fullfile(subfolder_weights, sprintf('IT2FNNP2_%d_Chaotic_0.3_weights_I%d_M%d_C%d_Tg%2f_Tr%2f.mat', iii,I, M,Cnum,threshold_growing,threshold_reducing));   
    % ä½¿ç”�? save å‡½æ�?�°ä¿å­˜å¤šä¸ªæ�?�°ç»�??
    save(filename_weights, 'ant_par', 'con_par','Mqlqr','Mant','Mylyr','K_link','rule_number');


end

total_time = toc;
disp(['total training time costs: ', num2str(total_time)]);

%% IT2FNN Prediction(IT2FNNP)

%-------------------------------------------------
% IT2FNNP1  : Single-optput Parallel prediction scheme (PM)
% IT2FNNP2  : Single-optput serial prediction scheme  (SW)
% IT2FNNP3  : Multi-optput  prediction scheme (MO)
%-------------------------------------------------
clc;
close all;
clear all;
subfolder_data = 'IT2FNN_normalization_data';
I = 9;
M = 3;
% K_link = 0.05;
subfolder_weights = 'Weights';

% filename_test = fullfile(subfolder_data, 'Chaotic_test_9_3_0.000000.mat');
% % filename_test = fullfile(subfolder_data, 'Chaotic_train_9_3_0.300000.mat');
% load(filename_test)
% X_test = test_input';
% Y_test = test_output';
% % X_test = train_input';
% % Y_test = train_output';

% filename_test = fullfile(subfolder_data, 'unmets_HWM_train.mat');
filename_test = fullfile(subfolder_data, 'prs_HWM_test.mat');
load(filename_test)
inputs = testData(:, 4:15);
% inputs = trainData(:, 4:15);
X_test = inputs(:,1:9);
Y_test = inputs(:,10:12);

If_normalization = 1; % 1: Normalization' 0: not
MM = 1;
MMM = size(Y_test,2);

% filename_raw = '2012_Initial_data - unmet.csv';
filename_raw = '2012_Initial_data-pr.csv';
original_data = csvread(filename_raw);

min_vals = min(original_data);
max_vals = max(original_data);

for iii = 1: MM

%     filename_weight_loading = fullfile(subfolder_weights, sprintf('IT2FNNP2_%d_Chaotic_0.0_weights_I9_M1_C5_Tg0.002500_Tr0.002500.mat',iii));
    filename_weight_loading = fullfile(subfolder_weights, sprintf('IT2FNNP2_%d_prs_weights_I9_M1_C10_Tg0.001000_Tr0.001000.mat',iii));
    load(filename_weight_loading);

    for iiii = 1:MMM

        [My_Y_tmp] = IT2FNNP (X_test,ant_par,con_par,Mqlqr,I,Y_test(:,iiii),Mant, K_link,Mylyr);
    
        My_Y(:,iiii)= My_Y_tmp;  % 1* sam_num;å®ŒæˆåŽæ?»å…±ä¸ºMM* sam_num
    
        mse = mean((My_Y(:,iiii) - Y_test(:,iiii))'.^2, 2); % Y_batchå�?? M_y ä¸? batch_size*Mï¼? sum((Y_batch' - M_y).^2, 2)ä¸ºM*1; 
        disp('loss(mse): ');
        disp(mse)

        rmse(iiii) = sqrt(mse);
        disp('rmse:')
        disp(rmse(iiii))
    
        if If_normalization == 1
            Y_back(:,iiii) = Y_test(:,iiii)*(max_vals-min_vals) + min_vals;  
            My_y_back(:,iiii) = My_Y(:,iiii)*(max_vals-min_vals) + min_vals;
        else
            Y_back(:,iiii) = Y_test(:,iiii);  
            My_y_back(:,iiii) = My_Y(:,iiii);
        end
    
    
        figure(iiii+1)
        plot(Y_back(:,iiii), 'b', 'LineWidth', 2);  % 'r-' è¡¨ç¤ºçº¢è‰²å®žçº�?
        hold on; 
        plot(My_y_back(:,iiii), 'r--', 'LineWidth', 2);
        hold off;
        
        X_test(:, 1) = [];
        X_test = [X_test, My_Y_tmp];

    end
end


disp('mean rmse:')
disp(mean(rmse))

% %% SAVE For post_processing
% yp_sw = My_y_back;
% ya_sw = Y_back;
% 
% error_sw = ya_sw - yp_sw;
% r_error_sw = abs(ya_sw - yp_sw)./ya_sw;
% 
% 
% subfolder_weights = 'Post_processing';
% filename_results = fullfile(subfolder_weights, 'prs_results_SW.mat');
% % filename_results = fullfile(subfolder_weights, 'Chaotic_results_SW.mat');
% save(filename_results, 'error_sw','r_error_sw','yp_sw','ya_sw');