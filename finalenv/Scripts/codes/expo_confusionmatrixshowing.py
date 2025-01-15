from expo_confusionmatrixcreating import confusing, plot_cm, plot_perc_cm
from deepface.models.demography.Emotion import load_model
import matplotlib.pyplot as plt

model = load_model('finetuned_model.h5')
cm2 = confusing(model,'Custom Model')
plot_perc_cm(cm2,'Custom Model')


cm1 = confusing('deepface','DeepFace')
plot_perc_cm(cm1, 'DeepFace')

plt.show()