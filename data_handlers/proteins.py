import pennylane.numpy as pnp
import pandas as pd

def CAPRI_label(metric_df:pd.DataFrame,thresholds:dict={'high':0.5,'medium':1.0,'acceptable':5.0}, return_tags:bool=False) -> pd.Series:
    '''Assigns CAPRI quality labels based on provided metrics DataFrame.
    Args:
        metric_df (pd.DataFrame): DataFrame with columns 'fnat', 'l_rms', 'i_rms'.
        thresholds (dict): Thresholds for high, medium, acceptable quality.
        return_tags (bool): Whether to return the tags along with labels.
    Returns:
        List[str]: List of CAPRI quality tags if return_tags is True.
        pnp.ndarray: Array of CAPRI quality labels.
    '''
    labels = []
    tags = []
    for _, row in metric_df.iterrows():
        fnat = row['fnat']
        l_rms = row['l_rms']
        i_rms = row['i_rms']
        
        if fnat >= thresholds['high'] and l_rms <= 1.0 and i_rms <= 1.0:
            tags.append('high')
            labels.append(1.0)
        elif fnat >= thresholds['medium'] and l_rms <= 5.0 and i_rms <= 2.0:
            tags.append('medium')
            labels.append(1.0)
        elif fnat >= thresholds['acceptable'] and l_rms <= 10.0 and i_rms <= 4.0:
            tags.append('acceptable')
            labels.append(1.0)
        else:
            tags.append('incorrect')
            labels.append(0.0)
    
    # Convert to numpy array with gradient tracking and then return
    if return_tags:
        return pnp.array(labels, requires_grad=True), tags
    return pnp.array(labels, requires_grad=True)

def DockQ_score(metric_df:pd.DataFrame, d1:float = 8.5, d2:float = 1.5) -> pnp.ndarray:
    '''Calculates DockQ scores based on provided metrics DataFrame.
    First defined in Basu and Wallner, "DockQ: A Quality Measure for Protein-Protein Docking Models", PLoS ONE, 2016.
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0161879#abstract0
    Args:
        metric_df (pd.DataFrame): DataFrame with columns 'fnat', 'l_rms', 'i_rms'.
        d1 (float): Normalization constant for l_rms.
        d2 (float): Normalization constant for i_rms.
    Returns:
        pnp.ndarray: Array of DockQ scores.
    '''
    dockq_scores = []
    for _, row in metric_df.iterrows():
        fnat = row['fnat']
        l_rms = row['l_rms']
        i_rms = row['i_rms']
        
        score = (fnat / (1 + (l_rms / d1)**2 + (i_rms / d2)**2))
        dockq_scores.append(score)
    
    # Convert to numpy array with gradient tracking and then return
    return pnp.array(dockq_scores, requires_grad=True)