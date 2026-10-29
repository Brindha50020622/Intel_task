Problem Statement

I was  give a task related to my Internship project. The task was when given a audio call of some business case where agent and customer are conversing,
I was asked to take the time frames from the audio, extract voice features, and then to train a custom model to predict the escalations in the cutomer side.
The output contains a JSON file having the conversation between speakers with each segment indicating a escalation probablity. I built a pipeline where when a 
audio file is given, it generates transcripts using "Wispher" model(ASR), then call feature_extraction, then the trained and saved model would be used to predict 
the escalations probabilities in the customer segment of the call.

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

