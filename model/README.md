# model/

Place trained Keras model files here:

```
model/
  brain_tumor_model.keras   ← trained with training/train_model.py --disease brain_tumor
  pneumonia_model.keras     ← trained with training/train_model.py --disease pneumonia
  skin_cancer_model.keras   ← trained with training/train_model.py --disease skin_cancer
```

If no `.keras` files are found, the API automatically builds and uses a **demo MobileNetV2**
model with random weights (for testing the pipeline end-to-end).

## Download pre-trained weights (example)

```bash
# Hugging Face Hub (once uploaded)
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('yourname/medical-ai-models', 'brain_tumor_model.keras', local_dir='model/')
"
```
