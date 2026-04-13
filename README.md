# Temporal Symbolic Data for IDS in ICS

Implementation of a temporal-symbolic representation with adaptive pooling for intrusion detection in IT (CICIDS2017) and OT (SWaT, WADI) environments.

## Repository Structure

```
Temporal_symbolic_data/
│
├── cicids_models/                 
│   # Trained models on CICIDS2017 (weights, checkpoints)
│
├── swat_model/                    
│   # Trained models on the SWaT industrial dataset
│
├── wadi_model/                    
│   # Trained models on the WADI industrial dataset
│
├── preprocessed_swat/             
│   # Preprocessed SWaT data (cleaning, normalization, formatting)
│
├── preprocessed_wadi/             
│   # Preprocessed WADI data
│
├── X_input_split_train_smote/     
│   # Training data (CICIDS) transformed into 2D matrices
│   # Includes SMOTE augmentation
│
├── X_input_split_test/            
│   # Test data (CICIDS) transformed into 2D matrices
│
├── preprocessing_and_2D_transformation_CICIDS.ipynb
│   # Full pipeline:
│   # - CICIDS cleaning
│   # - normalization
│   # - temporal 2D matrix construction (sliding window)
│
├── preprocessing_swat.ipynb       
│   # SWaT preprocessing pipeline (cleaning + normalization + structuring)
│
├── preprocessing_wadi.ipynb       
│   # WADI preprocessing pipeline
│
├── training_dynamic_pooling_cicids.ipynb
│   # Training CNN with mask-guided adaptive pooling (CICIDS)
│
├── train_dynamic_swat.ipynb       
│   # Training with adaptive pooling on SWaT
│
├── train_dynamic_wadi.ipynb       
│   # Training with adaptive pooling on WADI
│
├── test_dynamic_pooling_cicids.ipynb
│   # Evaluation on CICIDS (adaptive pooling)
│
├── test_dynamic_swat.ipynb        
│   # Evaluation on SWaT
│
├── test_dynamic_wadi.ipynb        
│   # Evaluation on WADI
│
├── training_avgpooling_cicids.ipynb     
│   # Baseline: training with average pooling
│
├── training_custom_pooling_cicids.ipynb 
│   # Previous approach: custom/weighted pooling
│
├── test_avgpooling_cicids.ipynb         
│   # Evaluation with average pooling
│
├── test_custom_pooling_cicids.ipynb     
│   # Evaluation with custom pooling
```
