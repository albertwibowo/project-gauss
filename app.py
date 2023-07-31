import streamlit as st 
import pandas as pd 

# sidebar elements
st.sidebar.write("Settings")

distance_metric = st.sidebar.selectbox(label="Select distance metric",
                                     options=['normalised compressed distance',
                                              'lempel-ziv jaccard distance'])
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
The following app can be used to detect simple data drift.
Two dataframes must be uploaded in order for the app to function:

* reference dataframe: this is the source of truth
* target dataframe: this is the dataframe to be investigated

""")


tab0, tab1, tab2 = st.tabs(["Guidelines", "Data", "Analysis"])

# Tab 0 - Guidelines

tab0.subheader("Algorithm setting")
tab0.markdown(
    """
* **P-value threshold**: This threshold is used to decide the drift threshold - any p-value below the threshold will result in the features being flagged
* **Categorical column algorithm**: This will choose the algorithm used to detect drift in categorical columns- it can only use chi-square test for now.
* **Numerical column algorithm**: This will choose the algorithm used to detect drift in numerical columns- it can only use kolmogorov-smirnov test for now.
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
target_file = tab1.file_uploader("Choose a target dataframe")
reference_file = tab1.file_uploader("Choose a reference dataframe")

if target_file is not None and reference_file is not None:
    target_df = pd.read_csv(target_file)
    reference_df = pd.read_csv(reference_file)


    tab1.markdown("###### Target dataframe")

    tab1.markdown("###### reference dataframe")


    tab1.markdown("###### Column types")

# Tab 2 - Analysis

if target_file is not None and reference_file is not None:
    tab2.markdown("###### Analysis result")

    tab2.markdown("###### Visualisation")

    