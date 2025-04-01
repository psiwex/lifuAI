function [aaInfo]=timeExtractor(version,aa,tr,offset)

%--------------------------------------------------------------------------
 % blockLoader.m

 % Last updated: Dec 2023, John LaRocco
 
 % Ohio State University
 
 % Details: EFAT extractor
 %--------------------------------------------------------------------------
 taskonsets=[];
 aaInfo=[];
    if version == "1" %%Compare strings. If run 1 has version 1, based on examining the file name, above, then these are the onsets:
        Version = 1;
        taskonsets.angry = [0 280 440];
        taskonsets.fear = [40 160 400];
        taskonsets.happy = [80 200 360];
        taskonsets.shapes = [20 60 100 140 180 220 260 300 340 380 420 460];
        taskonsets.sad = [120 240 320];
    elseif version == "2"  %%Compare strings. If run 1 has version 2, based on examining the file name, above, then these are the onsets:
        Version = 2;
        taskonsets.angry = [40 160 400];
        taskonsets.fear = [0 280 440];
        taskonsets.happy = [120 240 320];
        taskonsets.shapes = [20 60 100 140 180 220 260 300 340 380 420 460];
        taskonsets.sad = [80 200 360];
    elseif version == "3"  %%Compare strings. If run 1 has version 3, based on examining the file name, above, then these are the onsets:
        Version = 3;
        taskonsets.angry = [120 280 440];
        taskonsets.fear = [80 160 320];
        taskonsets.happy = [40 200 360];
        taskonsets.shapes = [20 60 100 140 180 220 260 300 340 380 420 460];
        taskonsets.sad = [0 240 400];
    elseif version == "4"   %%Compare strings. If run 1 has version 4, based on examining the file name, above, then these are the onsets:
        Version = 4;
        taskonsets.angry = [0 240 320];
        taskonsets.fear = [120 280 360];
        taskonsets.happy = [80 160 400];
        taskonsets.shapes = [20 60 100 140 180 220 260 300 340 380 420 460];
        taskonsets.sad = [40 200 440];
    end
taskonsets.angry=ceil((taskonsets.angry+offset)/tr);
taskonsets.fear=ceil((taskonsets.fear+offset)/tr);
taskonsets.happy=ceil((taskonsets.happy+offset)/tr);
taskonsets.shapes=ceil((taskonsets.shapes+offset)/tr);
taskonsets.sad=ceil((taskonsets.sad+offset)/tr);

aaAng=aa(:,:,:,taskonsets.angry);
aaInfo.angry=mean(aaAng,4);

aaAng=aa(:,:,:,taskonsets.fear);
aaInfo.fear=mean(aaAng,4);

aaAng=aa(:,:,:,taskonsets.happy);
aaInfo.happy=mean(aaAng,4);

aaAng=aa(:,:,:,taskonsets.shapes);
aaInfo.shapes=mean(aaAng,4);

aaAng=aa(:,:,:,taskonsets.sad);
aaInfo.sad=mean(aaAng,4);
aaInfo.taskonsets=taskonsets;
end