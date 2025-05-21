function XX = Feature_extraction(X,Timestamps)

   I = size(X,2);
   samp_num = size(X,1);

    % 将字符串数组转换为 MATLAB 日期时间对象
    datetimeArray = datetime(Timestamps, 'InputFormat', 'dd/MM/yyyy HH:mm:ss');
    
    % 提取月份,天数，周几，小时 特征
    months = month(datetimeArray);
    days = day(datetimeArray);
    weekdays = weekday(datetimeArray);
    hours = hour(datetimeArray);

    XX(:,1) = hours(1:samp_num);
    XX(:,2) = weekdays(1:samp_num);
    XX(:,3) = months(1:samp_num);
    XX(:,4:I+3) = X;
    
    % % 将新加特征转化为与X对应的时序序列
    % months = convert_to_sequence(months,I);
    % days = convert_to_sequence(days,I);
    % weekdays = convert_to_sequence(weekdays,I);
    % hours = convert_to_sequence(hours,I);

    % for i = 1 : 5
    %     XX(:,(i-1)*5 +1) = X(:,i);
    %     XX(:,(i-1)*5 +2) = months(:,i);
    %     XX(:,(i-1)*5 +3) = days(:,i);
    %     XX(:,(i-1)*5 +4) = weekdays(:,i);
    %     XX(:,(i-1)*5 +5) = hours(:,i);
    % end
end


function Se = convert_to_sequence(S,I)
    % 设置时序窗口的大小
    window_size = I;
    
    % 计算数据长度
    data_length = length(S);
    
    % 初始化矩阵，每行包含一个时序序列
    Se = zeros(data_length - window_size + 1, window_size);
    
    % 生成时序序列矩阵
    for i = 1:(data_length - window_size + 1)
        Se(i, :) = S(i:i+window_size-1);
    end
end