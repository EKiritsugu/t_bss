%% ��������źŽ�������

n_sources = 2;
algorithm_name = 'ilrma-IP';
save_path = ['../seped/', algorithm_name];
mix_info = ['../mixed/ch_',int2str(n_sources),'.xlsx'];

[num, txt, as] = xlsread(mix_info,1)