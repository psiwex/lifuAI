function [combinedInfo]=timeCombiner(aaValues,aaValues2)

%--------------------------------------------------------------------------
 % blockLoader.m

 % Last updated: March 2025, John LaRocco
 
 % Ohio State University
 
 % Details: EFAT combiner
 %--------------------------------------------------------------------------
combinedInfo=[];

combinedInfo.fear=(aaValues.fear+aaValues2.fear)./2;
combinedInfo.angry=(aaValues.angry+aaValues2.angry)./2;
combinedInfo.happy=(aaValues.happy+aaValues2.happy)./2;
combinedInfo.shapes=(aaValues.happy+aaValues2.shapes)./2;
combinedInfo.sad=(aaValues.happy+aaValues2.sad)./2;


end
