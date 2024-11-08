import streamlit as st
import pandas as pd
from src.algorithms.knn_compressor import KnnCompressor
from src.algorithms.ce_compressor import CeCompressor
from src.utils import calculate_f1_score

# sidebar elements
st.sidebar.write("Settings")

distance_metric = st.sidebar.selectbox(
    label="Select distance metric", options=["normalised compressed distance"]
)
classification_algo = st.sidebar.selectbox(
    label="Select classification algorithm",
    options=["knn + compressor", "cross entropy + compressor"],
)

if classification_algo == "knn + compressor":
    n_neighbours = st.sidebar.number_input(
        label="Input number of neighbours", min_value=1, value=5, step=1
    )

# main page
st.title("Compressor based text classification")
st.markdown(
    """
The following app can be used to classify text.
Two dataframes must be uploaded in order for the app to function:

* base dataframe: this is the "training" dataframe
* to-predict dataframe: this is the dataframe to be predicted

"""
)


tab0, tab1, tab2, tab3 = st.tabs(["Guidelines", "Example", "Data", "Theory"])

# Tab 0 - Guidelines

tab0.subheader("Algorithm setting")
tab0.markdown(
    """
* **Distance Metric**: Distance metric used to measure how different the compressed texts are - only support normalised compressed distance at the moment
* **Classification algorithm**: Classification algorithm used to classify texts
* **Input number of neighbours**: Number of neighbours if KNN + compressed is chosen
* **Sample fraction**: Fraction to sample data points for EACH class - 1.00 sample fraction means no sampling
"""
)

tab0.subheader("Workflow")
tab0.markdown(
    """
1. Toggle to the **Data** tab and upload two dataframes to be investigated
2. Select text and target columns
3. Select sample fraction if required - default is to sample 0.2 data points from EACH class
4. Run algorithm and download prediction result if required
"""
)

tab0.subheader("Notes")
tab0.markdown(
    """
* Ensure there is no missing value in a text column
* Ensure all values in a text column are string
* If the algorithm is too slow, use sample fraction
* If sample fraction does not work, consider chunking the dataset to be predicted
* The algorithm uses *gzip* compressor
"""
)

tab0.subheader("Disclaimer")
tab0.markdown(
    """
* I DO NOT own nor create the original algorithms
* I DO NOT earn any money from this app - it is a hobby project
"""
)

tab0.subheader("Future Improvements")
tab0.markdown(
    """
* Add different distance metrics
* Add different algorithms
* Add different compressors
* Add methods to validate result in the app
"""
)

# Tab 1 - Data

tab1.markdown("###### Dataframe example")
tab1.markdown(
    """
To use this app, the dataframe must be in a specific format. This tab
contains an example of the format of the dataframe. In general, the dataframe
must consist of text and class columns.
"""
)


@st.cache_data
def read_data(path: str):
    df = pd.read_csv(path)
    return df


tab1.markdown("###### Training dataframe example")
tab1.dataframe(read_data("data/Poem_classification - train_data.csv"))

tab1.markdown("###### Test dataframe example")
tab1.dataframe(read_data("data/Poem_classification - test_data.csv"))

# Tab 2 - Data

tab2.markdown("###### Upload dataframes")
to_predict_file = tab2.file_uploader("Choose to-predict dataframe")
base_file = tab2.file_uploader("Choose base dataframe")

if to_predict_file is not None and base_file is not None:
    to_predict_df = pd.read_csv(to_predict_file)
    base_df = pd.read_csv(base_file)

    tab2.markdown("##### Base dataframe")
    tab2.dataframe(base_df)

    tab2.markdown("##### To predict dataframe")
    tab2.dataframe(to_predict_df)

    text_col = tab2.selectbox(
        label="Please select a text column", options=base_df.columns
    )
    target_col = tab2.selectbox(
        label="Please select a class column", options=base_df.columns
    )
    sample_frac = tab2.slider(
        label="Sample base dataframe?",
        min_value=0.05,
        max_value=1.0,
        step=0.05,
        value=0.2,
    )

    if text_col and target_col:

        run_button = tab2.button(label="Run algorithm", key="session_button")
        if run_button:

            if classification_algo == "cross entropy + compressor":
                algo = CeCompressor(
                    base_df=base_df, to_predict_df=to_predict_df
                )
                result_df = algo.run(
                    text_col=text_col,
                    target_col=target_col,
                    sample_frac=sample_frac,
                )

            elif classification_algo == "knn + compressor":
                algo = KnnCompressor(
                    base_df=base_df, to_predict_df=to_predict_df
                )
                result_df = algo.run(
                    text_col=text_col,
                    target_col=target_col,
                    k=n_neighbours,
                    sample_frac=sample_frac,
                )

            else:
                pass

            tab2.markdown("##### Prediction result")

            # show metrics
            tab2.markdown("F1 score")
            col1, col2, col3 = tab2.columns(3)

            f1_micro = round(
                calculate_f1_score(
                    y_true=result_df[target_col],
                    y_pred=result_df["prediction"],
                    type="micro",
                ),
                2,
            )
            f1_macro = round(
                calculate_f1_score(
                    y_true=result_df[target_col],
                    y_pred=result_df["prediction"],
                    type="macro",
                ),
                2,
            )
            f1_weighted = round(
                calculate_f1_score(
                    y_true=result_df[target_col],
                    y_pred=result_df["prediction"],
                    type="weighted",
                ),
                2,
            )

            col1.metric("F1 Score Micro", f1_micro)
            col2.metric("F1 Score Macro", f1_macro)
            col3.metric("F1 Score Weighted", f1_weighted)

            tab2.markdown("Prediction")
            tab2.dataframe(result_df)

            tab2.download_button(
                label="download",
                data=result_df.to_csv(index=False),
                file_name="prediction.csv",
                mime="text/csv",
            )


# Tab 3 - Theory

tab3.subheader("Summary")
tab3.markdown(
    """
The algorithm centres around two ideas:
* A compressor
* A distance metric

A compression algorithm compresses a particular data - usually images, into smaller bits. In this case however,
we will be compressing texts instead. In general, compression algorithms can be categorised into
two different groups:

* A lossy compression that aims to minimise the size of the resulting compression at the cost of quality of the result
* A losless compression that  aims to preserve quality of the resulting compression at the cost of size of the result

In the context of a classification, a lossless compression algorithm is more suitable because it "preserves"
the information in a text as much as possible. We can think of the compression algorithm as a preprocessing
step prior to classification. In ML based classification task, usually we have to define a cost function
that will be used by the ML algorithm to "learn" information about the data. The algorithm does this by
trying different values of parameters that minimise a cost function. In the context of compressor based
classification however, there are no parameters to be learned and cost functions. The classification is
purely based on a distance metric.

The intuition is quite simple - consider the following scenario where we have three lines of text:

* Text A -> topic A
* Text B -> topic B
* Text C -> topic ?

To classify "Text C", we must get a sense of how "similar" it is to the other two texts. This can be done through
a distance metric. In this particular algorithm, we are using a distance metric called the
*Normalised Compression Distance*. This distance metric measures the extra "effort" needed to compress a particular
line of text if it was combined with another line of text - how difficult it is to compress "Text A + Text C"
compared to compressing "Text C" on its own? The lower the effort needed (or the distance), the more similar
the two texts are. Simple KNN algorithm can then be used to classify the line of text once distance for every
possible permutation has been calculated.

Given we must iterate through a list of text to be predicted and a list of
texts with known target class, the algorithm can be quite slow. It is in fact has a complexity of O(n^2). To tackle
this challenge, this application uses the concept of multiprocessing.

"""
)

tab3.subheader("Research Paper")
tab3.markdown(
    """
* [“Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors](https://aclanthology.org/2023.findings-acl.426) (Jiang et al., Findings 2023)
"""
)
