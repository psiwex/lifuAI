%     if datainfo(irun).version == "1" %%Compare strings. If run 1 has version 1, based on examining the file name, above, then these are the onsets:
%         Version = 1;
%         taskonsets(irun).angry = [0 280 440];
%         taskonsets(irun).fear = [40 160 400];
%         taskonsets(irun).happy = [80 200 360];
%         taskonsets(irun).shapes = [20 60 100 140 180 220 260 300 340 380 420 460];
%         taskonsets(irun).sad = [120 240 320];
%     elseif datainfo(irun).version == "2"  %%Compare strings. If run 1 has version 2, based on examining the file name, above, then these are the onsets:
%         Version = 2;
%         taskonsets(irun).angry = [40 160 400];
%         taskonsets(irun).fear = [0 280 440];
%         taskonsets(irun).happy = [120 240 320];
%         taskonsets(irun).shapes = [20 60 100 140 180 220 260 300 340 380 420 460];
%         taskonsets(irun).sad = [80 200 360];
%     elseif datainfo(irun).version == "3"  %%Compare strings. If run 1 has version 3, based on examining the file name, above, then these are the onsets:
%         Version = 3;
%         taskonsets(irun).angry = [120 280 440];
%         taskonsets(irun).fear = [80 160 320];
%         taskonsets(irun).happy = [40 200 360];
%         taskonsets(irun).shapes = [20 60 100 140 180 220 260 300 340 380 420 460];
%         taskonsets(irun).sad = [0 240 400];
tr=2;
offset=8;

%% pre 
filenam='s6mm_sub-HIFUAE002_task-EFATpre_run-1_bold.nii';
aa=niftiread(filenam);
ab=niftiinfo(filenam);
[aaValues]=timeExtractor("3",aa,tr,offset);

filenam='s6mm_sub-HIFUAE002_task-EFATpre_run-1_bold.nii';
aa=niftiread(filenam);
ab=niftiinfo(filenam);
[aaValues2]=timeExtractor("2",aa,tr,offset);

[preValues]=timeCombiner(aaValues,aaValues2);


%% post
filenam='s6mm_sub-HIFUAE002_task-EFATpost_run-1_bold.nii';
aa=niftiread(filenam);
ab=niftiinfo(filenam);
[aaValues]=timeExtractor("1",aa,tr,offset);

filenam='s6mm_sub-HIFUAE002_task-EFATpost_run-2_bold.nii';
aa=niftiread(filenam);
ab=niftiinfo(filenam);
[aaValues2]=timeExtractor("2",aa,tr,offset);

[postValues]=timeCombiner(aaValues,aaValues2);


[signifier]=contrastEfat(preValues);
[signifier]=contrastEfat(postValues);