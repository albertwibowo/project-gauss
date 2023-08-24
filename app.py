import streamlit as st 
import pandas as pd 
from src.algorithms.knn_compressor import KnnCompressor
from src.algorithms.ce_compressor import CeCompressor
from src.utils.general_utils import save_data

# sidebar elements
st.sidebar.write("Settings")

distance_metric = st.sidebar.selectbox(label="Select distance metric",
                                     options=['normalised compressed distance'])
classification_algo = st.sidebar.selectbox(label="Select classification algorithm",
                                     options=['knn + compressor',
                                              'cross entropy + compressor'])

if classification_algo == 'knn + compressor':
    n_neighbours = st.sidebar.number_input(label="Input number of neighbours", min_value=1,
                                           value=5, step=1)

# main page
st.title("Compressor based text classification")
st.markdown(
    """
The following app can be used to classify text.
Two dataframes must be uploaded in order for the app to function:

* base dataframe: this is the "training" dataframe
* to-predict dataframe: this is the dataframe to be predicted

""")


tab0, tab1, tab2 = st.tabs(["Guidelines", "Data", "Theory"])

# Tab 0 - Guidelines

tab0.subheader("Algorithm setting")
tab0.markdown(
    """
* **Distance Metric**: Distance metric used to measure how different the compressed texts are - only support normalised compressed distance at the moment.
* **Classification algorithm**: Classification algorithm used to classify texts.
* **Input number of neighbours**: Number of neighbours if KNN + compressed is chosen.
"""
)

tab0.subheader("Workflow")
tab0.markdown(
    """
1. Toggle to the **Data** tab and upload two dataframes to be investigated
2. Toggle to the **Analysis** tab to see the result of the detection
"""
)

tab0.subheader("Notes")
tab0.markdown(
    """
* Ensure each column in a dataframe has the right data format before being uploaded - the data type detection in the app is not perfect
* Do not upload datetime columns - this will cause error 
* The algorithm will ignore any column that **IS NOT** present in the reference dataframe
"""
)



# Tab 1 - Data 

tab1.markdown("###### Upload dataframes")
to_predict_file = tab1.file_uploader("Choose to-predict dataframe")
base_file = tab1.file_uploader("Choose base dataframe")

if to_predict_file is not None and base_file is not None:
    to_predict_df = pd.read_csv(to_predict_file)
    base_df = pd.read_csv(base_file)

    tab1.markdown("##### Base dataframe")
    tab1.dataframe(base_df)

    tab1.markdown("##### To predict dataframe")
    tab1.dataframe(to_predict_df)

    text_col = tab1.selectbox(label="Please select a text column", 
                              options=base_df.columns)
    target_col = tab1.selectbox(label="Please select a class column", 
                                options=base_df.columns)
    sample_frac = tab1.slider(label="Sample base dataframe?", 
                               min_value=0.05, 
                               max_value=1.0, 
                               step=0.05,
                               value=0.2)

    if text_col and target_col:

        run_button = tab1.button(label="Run algorithm", key="session_button")
        if run_button:
            
            if classification_algo == 'cross entropy + compressor':
                algo = CeCompressor(base_df=base_df,
                        to_predict_df=to_predict_df)
                result_df = algo.run(text_col=text_col, 
                                    target_col=target_col,
                                    sample_frac=sample_frac)
            
            elif classification_algo == 'knn + compressor':
                algo = KnnCompressor(base_df=base_df,
                        to_predict_df=to_predict_df)
                result_df = algo.run(text_col=text_col, 
                                     target_col=target_col,
                                     k=n_neighbours,
                                     sample_frac=sample_frac)
            
            else:
                pass

            tab1.markdown("##### Prediction result")
            tab1.dataframe(result_df)

            tab1.download_button(label="download", 
                                       data=result_df.to_csv(index=False), 
                                       file_name="prediction.csv",
                                       mime="text/csv")
                

# Tab 2 - Theory

if to_predict_file is not None and base_file is not None:
    tab2.markdown("###### Prediction result")
    # tab2.dataframe(algo.final_predict_result)


    