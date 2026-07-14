import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

def dynamically_resample_and_fit(fold_pipeline, X_train_fold, y_train_fold, X_val_fold, clf):
    temp_pipeline_steps = [('preprocessor', fold_pipeline.named_steps['preprocessor'])]
    
    if len(y_train_fold) > 0:
        counts = pd.Series(y_train_fold).value_counts()
        avg_samples = int(counts.mean())
        
        downsample_strategy = {cls: avg_samples for cls, count in counts.items() if count > avg_samples}
        upsample_strategy = {cls: avg_samples for cls, count in counts.items() if count < avg_samples}
        
        if downsample_strategy:
            temp_pipeline_steps.append(('under_sampler', RandomUnderSampler(sampling_strategy=downsample_strategy, random_state=42)))
        
        if upsample_strategy:
            min_minority_samples = counts.min()
            k_neighbors = max(1, min(5, min_minority_samples - 1)) if min_minority_samples > 1 else 1
            temp_pipeline_steps.append(('smote_over_sampler', SMOTE(sampling_strategy=upsample_strategy, k_neighbors=k_neighbors, random_state=42)))
            
    temp_pipeline_steps.append(('clf', clf))
    execution_pipeline = ImbPipeline(temp_pipeline_steps)
    execution_pipeline.fit(X_train_fold, y_train_fold)
    return execution_pipeline