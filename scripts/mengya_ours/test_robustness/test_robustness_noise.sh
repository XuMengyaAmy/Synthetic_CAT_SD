# for type in gaussian_noise, shot_noise, impulse_noise speckle_noise
# bash /home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/scripts/endo_instru_lr0.01_0.001/test_robustness/test_script_robustness_corruption_elastic_transform/FT_endo_instru_17-18.sh
# bash /home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/scripts/endo_instru_lr0.01_0.001/test_robustness/test_script_robustness_corruption_elastic_transform/ILT_endo_instru_17-18_not_train_step0.sh
# bash /home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/scripts/mengya_ours/test_robustness/real_with_temp/elastic_transform/ILT_24_4_temperature_apply_customized_endo_instru_17-18_not_train_step0.sh
# bash /home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/scripts/mengya_ours/test_robustness/synthetic_with_tem/elastic_transform/ILT_24_4_temperature_apply_customized_endo_instru_17-18_not_train_step0.sh


for type in gaussian_noise shot_noise impulse_noise # speckle_noise
do 
    bash /home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/scripts/endo_instru_lr0.01_0.001/test_robustness/test_script_robustness_corruption_$type/FT_endo_instru_17-18.sh
    bash /home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/scripts/endo_instru_lr0.01_0.001/test_robustness/test_script_robustness_corruption_$type/ILT_endo_instru_17-18_not_train_step0.sh
    bash /home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/scripts/mengya_ours/test_robustness/real_with_temp/$type/ILT_24_4_temperature_apply_customized_endo_instru_17-18_not_train_step0.sh
    bash /home/ren2/data2/mengya/mengya_code/CVPR2021_PLOP/scripts/mengya_ours/test_robustness/synthetic_with_tem/$type/ILT_24_4_temperature_apply_customized_endo_instru_17-18_not_train_step0.sh
done