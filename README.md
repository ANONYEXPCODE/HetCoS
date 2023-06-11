# HetCoS
This is the implementation of the HetCoS approach.

The main contributions of this work include the novel HCG and the proposed model:

- The HCG is built based on the AST through two steps: (1) remove some non-leaf AST nodes by considering both the layout and syntactic features of source code, and (2) specify eight types of edges between graph nodes. The first step largely reduces the depth of HCG, thus making it easier to propagate the message from the bottom nodes in the tree to the top nodes in neighborhood aggregation of GNNs. After the second step, the structural and sequential information of source code can be unified into HCG comprehensively and the structural uniqueness of HCG can be well maintained. **We feel that it is a novel direction to reduce the depth of code syntax structure while investigating its structural heterogeneity for automatic code summarization.

- The proposed code summarization model is composed of an HCG encoder, a summary decoder, and a copying mechanism. In the HCG encoder, the HCGNN is designed to extract the heterogeneous features of HCG through layer-by-layer neighborhood aggregations. Each HCGNN layer contains a two-step aggregation mechanism and a residual connection with a graph normalization. The decoder is a Vallina Transformer decoder. The copying mechanism adopts multi-head attention to generate the summary token by copying it from HCG nodes as well as selecting it from the summary vocabulary.

# Runtime Environment
- 4 NVIDIA 2080 Ti GPUs 
- Ubuntu 20.04.5
- CUDA 11.3 (with CuDNN of the corresponding version)
- Python 3.9
- PyTorch 1.11.0
- PyTorch Geometric 2.0.4 
- Specifically, install our package with ```pip install my-lib-0.1.2.tar.gz```. The package can be downloaded from [Google Drive](https://drive.google.com/file/d/1eXCJEkCWuxi8xqMa_XAjg6OH4DNknT35/view?usp=share_link)  
# Dataset
The whole datasets of Python and Java can be downloaded from [Google Drive](https://drive.google.com/file/d/1eXCJEkCWuxi8xqMa_XAjg6OH4DNknT35/view?usp=share_link).

We provide the **statistics of the main Java and Python datasets** below. Note that, the AST size and HCG size mean the numbers of nodes in the original AST and proposed HCG, respectively. It can be observed from the boxplots that the sizes and depths of the ASTs can be evidently reduced after removing the non-leaf nodes based on the rules designed in the paper. For example, the depths are averagely reduced by at least 1/2.

<img src="https://github.com/ANONYEXPCODE/IMG/blob/main/HETCOS/Statistics_of_Java_dataset.png" width="100%" height="100%" alt="JavaStat">

<img src="https://github.com/ANONYEXPCODE/IMG/blob/main/HETCOS/Statistics_of_Python_dataset.png" width="100%" height="100%" alt="PythonStat">

**Note that:** 
- We provide 100 samples for train/valid/test datasets in the directory `data/python/raw_data/`, `data/java/raw_data/`, and `data/python_GypSum/raw_data/`. 
- The python_GypSum dataset was originally built by the work [GypSum: Learning Hybrid Representations for Code Summarization](https://arxiv.org/pdf/2204.12916.pdf) for the model evaluation on the cleaned testing set, which is different from the python dataset in `data/python/raw_data/`.
- To run on the whole datasets,
please download them from [Google Drive](https://drive.google.com/file/d/1eXCJEkCWuxi8xqMa_XAjg6OH4DNknT35/view?usp=share_link) for usage.

# Experiment on the Python(Java/Python_GypSum) Dataset
1. Step into the directory `src_code/python(java,python_GypSum)/code_sum_41`:
    ```angular2html
    cd src_code/python/code_sum_41
    ```
    or
    ```angular2html
    cd src_code/java/code_sum_41
    ```
    or
    ```angular2html
    cd src_code/python_GypSum/code_sum_41
    ```
2. Pre-process the train/valid/test data:
   ```angular2html
   python s1_preprocessor.py
    ```
    **Note that:**
    It will take hours for pre-processing the whole dataset.

3. Run the model for training, validation, and testing:
    ```angular2html
   python s2_model.py
   ```
  
After running, the console will display the performances on the whole testing of the python/java datasets and the performance on the cleaned testing set of the python_GypSum dataset. The predicted results of testing data, along with the ground truth and source code, will be saved in the path `data/python(java,python_GypSum)/result/codescriber_v41a1_6_8_512.json` for observation.

We have provided the results of the whole testing sets. The user can get the evaluation results on the whole python/java testing sets directly by running 
```angular2html
python s3_eval_whole_test_set.py"
```
The user can also get the evaluation results on the cleaned java/python_GypSum testing sets by directly run
```angular2html
python s3_eval_cleaned_test_set.py"
```

**Note that:** 
- All the parameters are set in `src_code/python(java,python_GypSum)/config.py`.
- If a model has been trained, you can set the parameter "train_mode" in `config.py` to "False". Then you can predict the testing data directly by using the model that has been saved in `data/python/model/`.
- We have provided in [Google Drive](https://drive.google.com/file/d/1eXCJEkCWuxi8xqMa_XAjg6OH4DNknT35/view?usp=share_link) all the files including the trained models as well as the log files of training processes (in `data/python(java,python_GypSum)/log/`). The user can download them for reference and model evaluation without running `s1_preprocessor.py` and model training. Still, don't forget to set the parameter "train_mode" in `config.py` to "False" for direct prediction and evaluation with these files.

# More Implementation Details.
## Parameter Settings
- The main parameter settings for HetCoS are shown as:

<img src="https://github.com/ANONYEXPCODE/IMG/blob/main/HETCOS/parameter_setting_new.png" width="50%" height="50%" alt="params">

In the experiment, the number of nodes of an HCG is subject to both the maximum HCG size and maximum source code length. The two maximum values, as well as the maximum summary text length are set based on the statistics of HCG sizes, code lengths, and summary lengths, which are presented in the boxplots above. For each boxplot, we select the value at the maximum point as the corresponding maximum value, which is reasonable considering the statistics property.
For example, in a Java HCG, the number of its nodes should not be more than 473, and the number of its leaf nodes (source code tokens) cannot be more than 321. If not, the graph should be truncated.
The maximum summary lengths for Java and Python datasets are respectively set to 38 and 22, according to the distributions displayed in the boxplots. One may argue that the summary lengths for the two datasets are lower than those (50 and 30) in the baselines, such as Dual Model and CopyTrans. In fact, by setting the summary lengths to 50 and 30 on the Java and Python datasets, the experimental results show that the performance of our HetCoS is not affected. For example, the performance is **36.52/26.39/52.46%** on the Python dataset. We provide the data including the training and testing results and logs in [Google Drive](https://drive.google.com/file/d/1-gqUlaehrEkUOhNIRFL7AnmfC_Vpj4JM/view?usp=share_link).

## Resource Consumption
- The time used per epoch and the memory usage are provided as:

<img src="https://github.com/ANONYEXPCODE/IMG/blob/main/HETCOS/memory_usage.png" width="50%" height="50%" alt="usage">

# Clarification about the Metrics
Following the baseline works, we introduce the ROUGE-L, S-BLEU, and METEOR as evaluation metrics. ROUGE-L was initially proposed to assess summarization systems. The S-BLEU and METEOR metrics are originally used in translation-like tasks.

Considering there have been many BLEU variants that may affect the evaluation, we clarify that the BLEU metric is fair in our experiment. Specifically, the BLEU used in our work is averaged sentence-level BLEU (S-BLEU) with smoothing function (SmoothingFunction().method4), which is provided by NLTK 3.3 and is the same as those used in the baselines (such as Dual Model,  CopyTrans, and CODESCRIBE). In fact, the [README file](https://github.com/Bolin0215/CSCGDual) of Dual Model has highlighted that NLTK version is 3.3 and the source code can be found at this [link](https://github.com/Bolin0215/CSCGDual/blob/f13a8d8050b2f7df926626c7ecf13658469d4e59/nmt/nmt.py). The similar code also exists in the implementation of [CopyTrans](https://github.com/wasiahmad/NeuralCodeSum/blob/0e197513446958b10dcb53ba3a93f1adddc5be1c/c2nl/eval/bleu/nltk_bleu.py) and [CODESCRIBE](https://github.com/GJCEXP/CODESCRIBE/blob/master/src_code/java/config.py)(The implementation of `get_sent_bleu` function calls `sentence_bleu` in NLTK 3.3). Besides, in case of the unfairness in BLEU, the additional METEOR and ROUGE-L still prove the outperformance of our approach.

# Experimental Result:
We provide part of our experimental result below.
- Comparison with 16 state-of-the-arts, including 12 models without pre-training and 4 pre-training-based models.

<img src="https://github.com/ANONYEXPCODE/IMG/blob/main/HETCOS/SOTA_new.png" width="80%" height="80%" alt="Comparison with the baselines">

<font color=FireBrick><em>

- The detailed results of ablation study are presented below. 
In the ablation study, if an investigated component is removed or changed, it is a rule that the other components should not be modified.
The performances of R-CodeEdge, R-SibEdge, R-DFG, R-Het and V-AST illustrate the effectiveness of the components in HCG structure. The scores of R-Copy, R-EncRes, V-HetGCN, V-HetGAT, V-HetGT demonstrate the effectiveness of the modules in HetCoS model. 

  To be specific, The performances of R-CodeEdge, R-SibEdge, R-DFG prove the effectiveness of the incorporations of plain code sequence, sibling relations, and DFG, respectively. R-Het's result shows the effectiveness of HCG's heterogeneous features. V-AST keeps all the AST nodes in HCG, of which the result indicates that reducing the AST depth indeed helps for code summarization. R-Copy and R-EncRes is designed to validate the effectiveness of the multi-head attention-based copying mechanism and the residual connection in the HCGNN layer, respectively.

  **Note that:**
  (1) The residual connection with normalization is important in multi-layer computing for deep learning models, since it mitigates the problems of vanishing gradient and high vector offset. In practice, such problems indeed affect the deep models a lot. In our work, the number of HCG encoding layers is set to 6. The result of R-EncRes demonstrates that there would be vanishing gradient and high vector offset issues without residual connection, and the integration of residual connection helps our model solve the problems.
  (2) In R-Het variant, since the HCG becomes a homogeneous graph, we only use one-step aggregation. The aggregation function is "mean" (not "sum"), which is the same as that in the HetCoS's first-step aggregation. It is reasonable because the homogeneous graph can be regarded as a special heterogeneous graph with only one edge type. In fact, we have also conducted the experiment by replacing the "mean" function in R-Het with "sum". The result shows that the performance of R-Het drops significantly with "sum" used (only **48.73/32.50/59.86%** and **34.28/23.94/47.75%** on the Java and Python datasets).
</em></font>

<img src="https://github.com/ANONYEXPCODE/IMG/blob/main/HETCOS/ablation_study.png" width="70%" height="70%" alt="Ablation Study">

- Detailed results of study on the model size

<img src="https://github.com/ANONYEXPCODE/IMG/blob/main/HETCOS/performance_wrt_enc_layer.png" width="70%" height="70%" alt="HCG_layers">

<img src="https://github.com/ANONYEXPCODE/IMG/blob/main/HETCOS/performance_wrt_dec_layer.png" width="70%" height="70%" alt="sum_layers">

<img src="https://github.com/ANONYEXPCODE/IMG/blob/main/HETCOS/performance_wrt_emb_size.png" width="70%" height="70%" alt="emb_sizes">

- Qualitative examples on the Java and Python datasets.
    
    **Note that:**
    All the words in summary texts were lemmatized in the pre-processing, which is a common operation to reduce the vocabulary size in the NLP community. For example, the word "describing" and "specified" became "describe" and "specify" after lemmatization. As a result, some tokens in the processed and generated texts might not fit the context exactly. Practically, such minor grammatical issues do not affect the human understanding of natural language summary texts. To facilitate the observation, we fixed the issues manually in the displayed cases. For example in the first Java case, the word "describe" generated by HETCOS,CODESCRIBE, GypSum, and CopyTrans was revised as "describing" due to the context "return a string". The word "specify" was revised as "specified" in the fourth Java case. Besides, we removed the space between the words and their following punctuation marks to make the summary texts more natural.
    We make sure that all the minor revisions do not affect the meanings of the generated summaries.

![case_java](https://github.com/ANONYEXPCODE/IMG/blob/main/HETCOS/case_java.png)

![case_python](https://github.com/ANONYEXPCODE/IMG/blob/main/HETCOS/case_python.png)

***This paper is still under review, please do not distribute it.***
