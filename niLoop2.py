import nibabel as nib
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from nilearn import image, plotting
from nilearn.image import resample_to_img
from nilearn.datasets import load_mni152_template
from nibabel.processing import resample_to_output
import nilearn.image as nli

roiList=list(['./roi/Amygdala_L_1.nii','./roi/Amygdala_R_1.nii','./roi/Blackford_BNST_L_final_-6_2_0.nii','./roi/Blackford_BNST_R_final_6_2_0.nii','./roi/Insula_L_AAL3_1mm_33.nii','./roi/Insula_R_AAL3_1mm_34.nii'])

roiList=list(['./roi/Amygdala_L_1.nii','./roi/Amygdala_R_1.nii','./roi/SPM8_L_ANT_INS_-34_14_-1.nii','./roi/SPM8_R_ANT_INS_38_15_-3.nii','./roi/Thalamus_L_1.nii','./roi/Thalamus_R_1.nii'])

roiList=list(['./roi/Amygdala_L_1.nii','./roi/Amygdala_R_1.nii','./roi/SPM8_L_ANT_INS_-34_14_-1.nii','./roi/SPM8_R_ANT_INS_38_15_-3.nii','./roi/fusif_L.nii.nii','./roi/fusif_R.nii'])


roiList=list(['./roi/Amygdala_L_1.nii','./roi/Amygdala_R_1.nii','./roi/SPM8_L_ANT_INS_-34_14_-1.nii','./roi/SPM8_R_ANT_INS_38_15_-3.nii','Blackford_BNST_L_final_-6_2_0.nii','./roi/Blackford_BNST_R_final_6_2_0.nii'])

roiList=list(['./roi/Amygdala_L_1.nii','./roi/Amygdala_R_1.nii','./roi/SPM8_L_ANT_INS_-34_14_-1.nii','./roi/SPM8_R_ANT_INS_38_15_-3.nii','./roi/fusiform_L.nii.nii','./roi/fusiform_R.nii'])




mask = nib.load(roiList[0])
maskData = mask.get_fdata()

contrastFiles=list(['./sub-HIFUAE002pre/con_0001.nii','./sub-HIFUAE002pre/con_0002.nii','./sub-HIFUAE002pre/con_0003.nii','./sub-HIFUAE002pre/con_0004.nii','./sub-HIFUAE002pre/con_0005.nii','./sub-HIFUAE002pre/con_0006.nii','./sub-HIFUAE002pre/con_0007.nii','./sub-HIFUAE002pre/con_0008.nii','./sub-HIFUAE002pre/con_0009.nii','./sub-HIFUAE002pre/con_0010.nii','./sub-HIFUAE002pre/con_0011.nii','./sub-HIFUAE002post/con_0001.nii','./sub-HIFUAE002post/con_0002.nii','./sub-HIFUAE002post/con_0003.nii','./sub-HIFUAE002post/con_0004.nii','./sub-HIFUAE002post/con_0005.nii','./sub-HIFUAE002post/con_0006.nii','./sub-HIFUAE002post/con_0007.nii','./sub-HIFUAE002post/con_0008.nii','./sub-HIFUAE002post/con_0009.nii','./sub-HIFUAE002post/con_0010.nii','./sub-HIFUAE002post/con_0011.nii'])

looperLength=len(contrastFiles)
looperLength=len(roiList)

ii=21
img=nib.load(contrastFiles[ii])
print(contrastFiles[ii])



meanList=list([])
print(looperLength)
for iii in range(0,looperLength):
	try:
		mask = nib.load(roiList[iii])
		maskData = mask.get_fdata()

		brainData = img.get_fdata()
		rmask = image.resample_to_img(mask, img, copy_header=True, force_resample=True,interpolation='nearest')
		maskData = rmask.get_fdata()
		masked_data = brainData * (maskData == 1)
		mimg = nib.Nifti1Image(masked_data, img.affine)
		mdata=mimg.get_fdata()
		arr = mdata
		mask = np.isfinite(arr)
		arr = arr[mask]
		non_zero_indices = np.nonzero(arr)
		non_zero_elements = arr[non_zero_indices]
		roiMean = np.mean(non_zero_elements)
		roiStd = np.std(non_zero_elements)
	except:
		roiMean=0
	print(roiMean)
	meanList.append(roiMean)

print(meanList)

