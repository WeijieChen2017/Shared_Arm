import os
code1 = "python3 yoshi.py --dir_pet breast1_pet --dir_mri breast1_water --blur_method skimage_gaus --blur_para "
code2 = " --slice_x 3 --batch_size 5 --id breast_x3_sg"

for idx in range(7):
	para_sg = idx/2 + 1
	code = code1+str(para_sg)+code2+str(para_sg)
	os.system(code)

