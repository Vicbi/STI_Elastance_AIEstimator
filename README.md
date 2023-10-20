# STI_Elastance_AIEstimator
### AI-Based Estimation of End-Systolic Elastance From Arm-Pressure and Systolic Time Intervals

This repository contains Python scripts including different machine learning models to derive end-systolic elastance (Ees) from non-invasive blood pressure and systolic timing intervals data.

**Abstract**

Left ventricular end-systolic elastance (Ees) is a major determinant of cardiac systolic function and ventricular-arterial interaction. Previous methods for the Ees estimation require the use of the echocardiographic ejection fraction (EF). However, given that EF expresses the stroke volume as a fraction of end-diastolic volume (EDV), accurate interpretation of EF is attainable only with the additional measurement of EDV. Hence, there is still need for a simple, reliable, noninvasive method to estimate Ees. This study proposes a novel artificial intelligence - based approach to estimate Ees using the information embedded in clinically relevant systolic time intervals, namely the pre-ejection period (PEP) and ejection time (ET). We developed a training/testing scheme using virtual subjects (n=4645) from a previously validated in-silico model. Extreme Gradient Boosting regressor was employed to model Ees using as inputs arm cuff pressure, PEP, and ET. Results showed that Ees can be predicted with high accuracy achieving a normalized RMSE equal to 9.15 % (r = 0.92) for a wide range of Ees values from 1.2 to 4.5 mmHg/mL. The proposed model was found to be less sensitive to measurement errors (±10% to 30% of the actual value) in blood pressure, presenting low test errors for the different levels of noise (RMSE did not exceed 0.32 mmHg/mL). In contrast, high sensitivity was reported for measurement errors in the systolic timing features. It was demonstrated that Ees can be reliably estimated from the traditional arm pressure and echocardiographic PEP and ET. This approach constitutes a step towards the development of an easy and clinically applicable method for assessing left ventricular systolic function.

<img width="1040" alt="Screenshot at Oct 19 18-07-16" src="https://github.com/Vicbi/STI_Elastance_AIEstimator/assets/10075123/97a25de1-e291-45b3-8282-1302fd7c70bc">


**Original Publication**

For a comprehensive understanding of the methodology and background, please refer to the original publication: Bikia, V., Adamopoulos, D., Pagoulatou, S., Rovas, G., & Stergiopulos, N. (2021). AI-based estimation of end-systolic elastance from arm-pressure and systolic time intervals. Frontiers in Artificial Intelligence, 4, 579541.

**Citation**

If you use this code in your research, please cite the original publication:

Bikia, V., Adamopoulos, D., Pagoulatou, S., Rovas, G., & Stergiopulos, N. (2021). AI-based estimation of end-systolic elastance from arm-pressure and systolic time intervals. Frontiers in Artificial Intelligence, 4, 579541.

**License**

This project is licensed under the Apache License 2.0 - see the LICENSE.md file for details.

This work was developed as part of a research project undertaken by the Laboratory of Hemodynamics and Cardiovascular Technology at EPFL (https://www.epfl.ch/labs/lhtc/).

Feel free to reach out at vickybikia@gmail.com if you have any questions or need further assistance!

