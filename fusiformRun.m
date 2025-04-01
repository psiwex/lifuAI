img=load('Fusiform_L_1_roi.mat');
T=img.roi.mat;

filenam='outL.nii';
filenam='fusiform_L.nii';
filenam='Thalamus_L_1';
filenam='N_Acc_L_AAL3_1mm_157.nii';
filenam='Amygdala_L_1.nii';
aa=niftiread(filenam);
ab=niftiinfo(filenam);


affine=aa;
ab.Transform.T=T';
ab.raw.srow_x = affine(1,:);
ab.raw.srow_y = affine(2,:);
ab.raw.srow_z = affine(3,:);

%niftiwrite(x3,'fur.nii')

niftiwrite(aa,'fusiform_L.nii',ab);

%niftiwrite(aa,'fusiform_R.nii');

% affine=x3;
% nii = struct('img', affine, 'hdr', struct());
% nii.hdr.dime.dim = [3 91 109 91 1 1 1 1];
% nii.hdr.hist.srow_x = affine(1,:);
% nii.hdr.hist.srow_y = affine(2,:);
% nii.hdr.hist.srow_z = affine(3,:);
% % 
% nii.hdr.Transform.T=T';
% % 
% niftiwrite(affine,'outL.nii');
% % 
% 
