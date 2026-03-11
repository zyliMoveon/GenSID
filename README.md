# GenSID: A versatile generative framework for systematic imaging biomarker discovery
This is a code implementation of "**A versatile generative framework for systematic imaging biomarker discovery**"

## Introduction
Systematically discovering imaging biomarkers from medical images remains a central yet unresolved challenge in computational medicine. 
Despite the widespread use of medical imaging in healthcare, imaging biomarker discovery still relies largely on disease-specific pipelines, handcrafted features, and intensive expert interpretation, limiting the ability to uncover  fine-grained, heterogeneous manifestations of disease at scale. 
Here we introduce GenSID, a computational framework that **repurposes generative artificial intelligence (AI) as an engine for systematic imaging biomarker discovery**. 
GenSID learns disentangled latent trajectories that continuously span healthy and diseased states, enabling structured exploration of disease-related image variations. 
By tracing these variations, GenSID automatically isolates, disentangles, and organizes subtle imaging abnormalities into candidate biomarkers, which can be efficiently refined through expert-in-the-loop validation.
Across multiple imaging modalities and representative diseases, GenSID recovers established imaging biomarkers while revealing **previously unreported** imaging patterns with strong visual disentanglement and high expert agreement. 
The framework further enables biomarker-level personalized assessment, reveals disease subtypes, and supports clinically meaningful patient stratification.
Together, our results demonstrate that generative AI can be transformed from a tool for image synthesis into a **computational paradigm for systematic biomarker discovery**, enabling the construction of dense and interpretable atlases of clinically actionable imaging biomarkers and opening new opportunities for scalable disease characterization across biomedical imaging.

## Overview of the framework

 <img src="./readme_files/overall_framework.png" alt="overall_framework" width="800"> 

# Datasets
3D grayscale magnetic resonance imaging for Alzheimer's disease: [ADNI](https://adni.loni.usc.edu/)  

2D grayscale chest radiography for pneumonia: [Mendeley](https://data.mendeley.com/datasets/rscbjbr9sj/2)  

2D RGB fundus imaging for diabetic retinopathy: [Eyepacs](https://www.eyepacs.com/) or from [Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection)  

# Usage