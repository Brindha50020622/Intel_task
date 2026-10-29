Problem Statement

I was  give a task related to my Internship project. The task was when given a audio call of some business case where agent and customer are conversing,
I was asked to take the time frames from the audio, extract voice features, and then to train a custom model to predict the escalations in the cutomer side.
The output contains a JSON file having the conversation between speakers with each segment indicating a escalation probablity. I built a pipeline where when a 
audio file is given, it generates transcripts using "Wispher" model(ASR), then call feature_extraction, then the trained and saved model would be used to predict 
the escalations probabilities in the customer segment of the call.

------------------------------------------------------------------------------------------------------------------------------------------------------------------

Challenges

  1.I was asked to take Bank calls(like emi payment calls, remainder calls) in which I have worked during internship, but the datset was not available for that, many datsets
asked to pay. So I took a dataset called "apptek_callcenter_dialogues" from hugging face - https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues/viewer/default/test?p=1.
AppTek Call-Center Dialogues is a long-form English ASR benchmark consisting of spontaneous, role-played agent–customer conversations across 14 accent groups and 16 service-oriented domains.
The dataset is designed to evaluate ASR systems under realistic conversational conditions, including extended interactions with disfluencies, repairs, and domain-specific language.

  2.The diarization using Pyannote took around 10 mins for each file, due to time contraint I decided not to include diarization in the pipeline, I have done it for one file 
which you can look it in predicted_segments.json

  3.Not every complaint or negative statement indicates escalation. Distinguishing ordinary conversations from potentially escalated segments was challenging.

  4.Selecting a suitable threshold was challenging because a higher threshold could miss potential escalations, while a lower threshold could flag too many 
segments for review.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

Implementation

I colelcted 30 audio files from this dataset and with 15 escalations and 15 non-escalations(as the time was limited, I had to go woth only 30 calls).Used gemini-2.5-flash model to label the escalations in the dataset.Although the selected calls were balanced, the segment-level dataset was imbalanced because it contained many more non-escalated segments than escalated segments.

**Feature Extraction**

    Features included:
        Duration of the segment.
        RMS energy, which represents the loudness of the audio.
        Zero-crossing rate, which measures how often the audio signal changes direction across zero.
        Mean pitch.
        Spectral centroid, which describes the brightness of the sound.
        MFCC features, which capture important characteristics of the audio signal.
        
    Combined these features with the segment labels to prepare the dataset for model training.

**Models**

    Random Forest: Trained as the primary model to predict the likelihood of escalation for each segment.

    LSTM: Trained as an experimental model to compare its performance with Random Forest on the sequential audio segments.

**Models Evaluation**

    Used 5-fold cross-validation grouped by call file, helping prevent segments from the same call from appearing in both the training and validation sets.
    Compared the models using precision, recall, F1-score, and average precision.

Used a threshold of 0.14(because the dataset contained significantly fewer escalated segments than non-escalated segments. A lower threshold makes the model more likely to flag potential escalations for review. However, this can also increase false positives. Therefore, the threshold is a practical choice for identifying candidate segments, rather than a guarantee that every flagged segment is a true escalation) to flag segments for review. Segments with predicted scores at or above this threshold are flagged as potential escalations.

You can refer model_comparision.csv for different threhold with different scores for the metrics.

The system saves segment-level predictions, timestamps, transcripts, and estimated escalation scores in output files.

<img width="972" height="170" alt="image" src="https://github.com/user-attachments/assets/c7cdff62-6376-41da-9009-7b9f24894598" />




Model output:

  You can find the test folds output in escalation_output.json

  i have attached how the pieplien works when given a audio call, complete transcript with probabilites get saved under predictions and the escalation segments exceeding the threshold would be displayed in the terminal
  <img width="1246" height="332" alt="Screenshot 2026-10-29 084405" src="https://github.com/user-attachments/assets/4d534335-fd63-4202-b6ee-fec8553e2360" />

  




