# STI_Elastance_AIEstimator
### AI-Based Estimation of End-Systolic Elastance From Arm-Pressure and Systolic Time Intervals

**Abstract**

Left ventricular end-systolic elastance (Ees) is a crucial factor in evaluating cardiac systolic function and ventricular-arterial interaction. Previous methods for Ees estimation often rely on the echocardiographic ejection fraction (EF), which requires additional measurements of end-diastolic volume (EDV) for accurate interpretation. This study introduces an innovative artificial intelligence-based approach to estimate Ees using clinically relevant systolic time intervals, specifically the pre-ejection period (PEP) and ejection time (ET).

We devised a training/testing scheme using virtual subjects (n=4,645) from a previously validated in-silico model. An Extreme Gradient Boosting regressor was employed to model Ees using arm cuff pressure, PEP, and ET as inputs. Results demonstrate high accuracy in predicting Ees, achieving a normalized RMSE of 9.15% (r=0.92) across a broad range of Ees values (1.2 to 4.5 mmHg/ml). The model exhibits robustness against measurement errors (±10–30% of actual value) in blood pressure, yielding low test errors even with varying levels of noise (RMSE ≤ 0.32 mmHg/ml).

However, the model displays high sensitivity to measurement errors in systolic timing features. This study establishes that Ees can be reliably estimated using traditional arm-pressure and echocardiographic PEP and ET measurements. This approach marks a significant step towards developing an easily applicable method for assessing left ventricular systolic function.

**Original Publication**
For a comprehensive understanding of the methodology and background, please refer to the original publication: Bikia, V., Adamopoulos, D., Pagoulatou, S., Rovas, G., & Stergiopulos, N. (2021). AI-based estimation of end-systolic elastance from arm-pressure and systolic time intervals. Frontiers in Artificial Intelligence, 4, 579541.

**Citation**

If you use this code in your research, please cite the original publication:

Bikia, V., Adamopoulos, D., Pagoulatou, S., Rovas, G., & Stergiopulos, N. (2021). AI-based estimation of end-systolic elastance from arm-pressure and systolic time intervals. Frontiers in Artificial Intelligence, 4, 579541.

**License**

This project is licensed under the Apache License 2.0 - see the LICENSE.md file for details.
