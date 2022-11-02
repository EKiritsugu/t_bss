%% 这个文件是关于ILRMA-IP算法的批量处理，同时也旨在构建一个信号混合后的批量处理的框架
clc;
clear all;

algorithm_name = 'ilrma-IP';
n_sources = 4
mixed_sig_path = ['../wped/',int2str(n_sources),'ch'];
save_path = ['../seped/',int2str(n_sources),'ch/', algorithm_name];
mkdir(save_path);
file_list = dir(fullfile(mixed_sig_path,'*.wav'));
[n_samples, ~] = size(file_list);

[sep_wav, ~] = audioread([file_list(1).folder, '\',file_list(1).name]);%没有任何实际意义，仅仅为了声明变量sep_wav以通过matlab的编译

for n_s = 1:n_samples
    fprintf('\b\b\b%3d', n_s);
    file_name = [file_list(n_s).folder, '\',file_list(n_s).name];
    [wav, fs] = audioread(file_name);
    if strcmp( algorithm_name, 'ilrma-IP') | strcmp( algorithm_name, 'WPE-ilrma-IP') 
        [sep_wav, ~] = ILRMA(wav, n_sources, fs, 2, 1024, 1024/4, 'hann', 100, 1);
    elseif strcmp( algorithm_name, 'ilrma-ISS') | strcmp( algorithm_name, 'WPE-ilrma-ISS') 
        [sep_wav, ~] = ILRMAISS(wav, n_sources, fs, 2, 1024, 1024/4, 'hann', 100, 1);
    end
    %ILRMAISS(mixSig, nSrc, sampFreq, nBases, fftSize, shiftSize, windowType, nIter, ilrmaType, refMic, applyNormalize, applyWhitening, drawConv)
    sep_wav = sep_wav./(max(abs(sep_wav)));

    wav_name = [save_path,'\',int2str(n_s-1),'.wav'];
    

    audiowrite(wav_name, sep_wav, fs);
    

    % [estSig, cost] = ILRMA(mixSig, nSrc, sampFreq, nBases, fftSize, shiftSize, windowType, nIter, ilrmaType, refMic, applyNormalize, applyWhitening, drawConv)

end
